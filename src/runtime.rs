use crate::error::EngineError;
use crate::ops::{KernelIo, MultiKernel, SingleKernel};
use crate::ops::{
    OpMeta, OperatorRegistry, ScratchProfile, SharedFamily, TsBivariateMomentsView,
    TsUnivariateMomentsView,
};
use crate::plan::{LogicalParam, PhysicalNode, PhysicalPlan, MAX_NODE_INPUTS};
use crate::state::{EngineState, RuntimeScratchConfig};
use crate::types::{
    AdvancePolicy, BorrowedFeatureFrame, EventEnvelope, FeatureFrame, FeatureFrameBuffers, Payload,
    SourceKind, QUALITY_FORCED_ADVANCE,
};
use std::collections::HashMap;
use std::sync::Arc;

const CONST_FIELD_PREFIX: &str = "__const__";
const VAR_NUM_EPS: f64 = 1e-12;

#[derive(Debug, Clone)]
struct FieldRoute {
    field_slot: usize,
    accessor: FieldAccessor,
}

#[derive(Debug, Clone)]
enum FieldAccessor {
    BarOpen,
    BarHigh,
    BarLow,
    BarClose,
    BarVolume,
    TradePrice,
    TradeSize,
    QuoteBidPrice,
    QuoteAskPrice,
    QuoteBidSize,
    QuoteAskSize,
}

impl FieldAccessor {
    fn from_binding(source_kind: SourceKind, field: &str) -> Option<Self> {
        match source_kind {
            SourceKind::Bar => match field {
                "open" => Some(Self::BarOpen),
                "high" => Some(Self::BarHigh),
                "low" => Some(Self::BarLow),
                "close" => Some(Self::BarClose),
                "volume" => Some(Self::BarVolume),
                _ => None,
            },
            SourceKind::TradeTick => match field {
                "price" => Some(Self::TradePrice),
                "size" => Some(Self::TradeSize),
                _ => None,
            },
            SourceKind::QuoteTick => match field {
                "bid_price" => Some(Self::QuoteBidPrice),
                "ask_price" => Some(Self::QuoteAskPrice),
                "bid_size" => Some(Self::QuoteBidSize),
                "ask_size" => Some(Self::QuoteAskSize),
                _ => None,
            },
            SourceKind::OrderBookSnapshot => None,
            SourceKind::Data => None,
        }
    }

    fn read_bar(&self, bar: &crate::types::BarLite) -> Option<f64> {
        match self {
            Self::BarOpen => Some(bar.open),
            Self::BarHigh => Some(bar.high),
            Self::BarLow => Some(bar.low),
            Self::BarClose => Some(bar.close),
            Self::BarVolume => Some(bar.volume),
            _ => None,
        }
    }

    fn read_trade(&self, trade: &crate::types::TradeTickLite) -> Option<f64> {
        match self {
            Self::TradePrice => Some(trade.price),
            Self::TradeSize => Some(trade.size),
            _ => None,
        }
    }

    fn read_quote(&self, quote: &crate::types::QuoteTickLite) -> Option<f64> {
        match self {
            Self::QuoteBidPrice => Some(quote.bid_price),
            Self::QuoteAskPrice => Some(quote.ask_price),
            Self::QuoteBidSize => Some(quote.bid_size),
            Self::QuoteAskSize => Some(quote.ask_size),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SingleStep {
    io: KernelIo,
    output_field_slot: Option<usize>,
    params: LogicalParam,
    kernel: SingleKernel,
}

#[derive(Debug, Clone, Copy)]
enum BarrierKernel {
    Batch(MultiKernel),
    PerInstrument(SingleKernel),
}

#[derive(Debug, Clone, Copy)]
struct BarrierStep {
    node_id: usize,
    io: KernelIo,
    output_field_slot: Option<usize>,
    params: LogicalParam,
    kernel: BarrierKernel,
}

#[derive(Debug, Clone)]
struct TsUnivariateMomentsConsumer {
    output_slot: usize,
    output_field_slot: Option<usize>,
    view: TsUnivariateMomentsView,
}

#[derive(Debug, Clone)]
struct TsUnivariateMomentsState {
    cap: usize,
    len: usize,
    write: usize,
    nan_count: usize,
    last_ts_ns: i64,
    last_input_generation: u64,
    sum: f64,
    sum_sq: f64,
    sum_cu: f64,
    sum_qu: f64,
    ring: Vec<f64>,
}

impl TsUnivariateMomentsState {
    fn with_window(window: usize) -> Self {
        let cap = window.max(1);
        Self {
            cap,
            len: 0,
            write: 0,
            nan_count: 0,
            last_ts_ns: i64::MIN,
            last_input_generation: 0,
            sum: 0.0,
            sum_sq: 0.0,
            sum_cu: 0.0,
            sum_qu: 0.0,
            ring: vec![f64::NAN; cap],
        }
    }

    #[inline]
    fn add_sample(&mut self, value: f64, order: u8) {
        if !value.is_finite() {
            self.nan_count += 1;
            return;
        }
        self.sum += value;
        if order >= 2 {
            let sq = value * value;
            self.sum_sq += sq;
            if order >= 3 {
                self.sum_cu += sq * value;
                if order >= 4 {
                    self.sum_qu += sq * sq;
                }
            }
        }
    }

    #[inline]
    fn remove_sample(&mut self, value: f64, order: u8) {
        if !value.is_finite() {
            if self.nan_count > 0 {
                self.nan_count -= 1;
            }
            return;
        }
        self.sum -= value;
        if order >= 2 {
            let sq = value * value;
            self.sum_sq -= sq;
            if order >= 3 {
                self.sum_cu -= sq * value;
                if order >= 4 {
                    self.sum_qu -= sq * sq;
                }
            }
        }
    }

    fn push(&mut self, value: f64, order: u8) {
        if self.len == self.cap {
            let old = self.ring[self.write];
            self.remove_sample(old, order);
        } else {
            self.len += 1;
        }
        self.ring[self.write] = value;
        self.add_sample(value, order);
        self.write += 1;
        if self.write == self.cap {
            self.write = 0;
        }
    }

    fn push_at(&mut self, value: f64, order: u8, ts_ns: i64) {
        if self.last_ts_ns == ts_ns && self.len > 0 {
            let idx = if self.write == 0 {
                self.cap - 1
            } else {
                self.write - 1
            };
            let old = self.ring[idx];
            self.remove_sample(old, order);
            self.ring[idx] = value;
            self.add_sample(value, order);
            return;
        }
        self.push(value, order);
        self.last_ts_ns = ts_ns;
    }
}

#[derive(Debug, Clone)]
struct TsUnivariateMomentsProfile {
    field_slot: usize,
    window: usize,
    required_order: u8,
    consumers: Vec<TsUnivariateMomentsConsumer>,
    states: Vec<TsUnivariateMomentsState>,
}

#[derive(Debug, Clone, Copy)]
struct TsUnivariateMomentsSnapshot {
    sum: f64,
    mean: f64,
    var: f64,
    std: f64,
    skew: f64,
    kurt: f64,
}

#[derive(Debug, Clone)]
struct TsBivariateMomentsConsumer {
    output_slot: usize,
    output_field_slot: Option<usize>,
    view: TsBivariateMomentsView,
}

#[derive(Debug, Clone)]
struct TsBivariateMomentsState {
    cap: usize,
    len: usize,
    write: usize,
    nan_count: usize,
    last_ts_ns: i64,
    last_lhs_generation: u64,
    last_rhs_generation: u64,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
    ring_x: Vec<f64>,
    ring_y: Vec<f64>,
}

impl TsBivariateMomentsState {
    fn with_window(window: usize) -> Self {
        let cap = window.max(1);
        Self {
            cap,
            len: 0,
            write: 0,
            nan_count: 0,
            last_ts_ns: i64::MIN,
            last_lhs_generation: 0,
            last_rhs_generation: 0,
            sum_x: 0.0,
            sum_y: 0.0,
            sum_xx: 0.0,
            sum_yy: 0.0,
            sum_xy: 0.0,
            ring_x: vec![f64::NAN; cap],
            ring_y: vec![f64::NAN; cap],
        }
    }

    #[inline]
    fn add_sample(&mut self, x: f64, y: f64) {
        if !x.is_finite() || !y.is_finite() {
            self.nan_count += 1;
            return;
        }
        self.sum_x += x;
        self.sum_y += y;
        self.sum_xx += x * x;
        self.sum_yy += y * y;
        self.sum_xy += x * y;
    }

    #[inline]
    fn remove_sample(&mut self, x: f64, y: f64) {
        if !x.is_finite() || !y.is_finite() {
            if self.nan_count > 0 {
                self.nan_count -= 1;
            }
            return;
        }
        self.sum_x -= x;
        self.sum_y -= y;
        self.sum_xx -= x * x;
        self.sum_yy -= y * y;
        self.sum_xy -= x * y;
    }

    fn push(&mut self, x: f64, y: f64) {
        if self.len == self.cap {
            let old_x = self.ring_x[self.write];
            let old_y = self.ring_y[self.write];
            self.remove_sample(old_x, old_y);
        } else {
            self.len += 1;
        }
        self.ring_x[self.write] = x;
        self.ring_y[self.write] = y;
        self.add_sample(x, y);
        self.write = (self.write + 1) % self.cap;
    }

    fn push_at(&mut self, x: f64, y: f64, ts_ns: i64) {
        if self.last_ts_ns == ts_ns && self.len > 0 {
            let idx = if self.write == 0 {
                self.cap - 1
            } else {
                self.write - 1
            };
            let old_x = self.ring_x[idx];
            let old_y = self.ring_y[idx];
            self.remove_sample(old_x, old_y);
            self.ring_x[idx] = x;
            self.ring_y[idx] = y;
            self.add_sample(x, y);
            return;
        }
        self.push(x, y);
        self.last_ts_ns = ts_ns;
    }
}

#[derive(Debug, Clone)]
struct TsBivariateMomentsProfile {
    lhs_field_slot: usize,
    rhs_field_slot: usize,
    window: usize,
    consumers: Vec<TsBivariateMomentsConsumer>,
    states: Vec<TsBivariateMomentsState>,
}

#[derive(Debug, Clone, Copy)]
struct TsBivariateMomentsSnapshot {
    cov: f64,
    beta: f64,
    corr: f64,
    slope: f64,
}

pub trait Engine {
    fn load(&mut self, plan: PhysicalPlan) -> Result<(), EngineError>;
    fn on_event(&mut self, event: &EventEnvelope) -> Result<(), EngineError>;
    fn is_graph_ready(&self, ts_ns: i64) -> bool;
    fn advance(&mut self, ts_ns: i64) -> Result<FeatureFrame, EngineError>;
}

#[derive(Debug, Default)]
pub struct OnlineFactorEngine {
    plan: Option<PhysicalPlan>,
    state: Option<EngineState>,
    instrument_index: HashMap<u32, usize>,
    factor_names: Option<Arc<[String]>>,
    factor_index: Option<Arc<HashMap<String, usize>>>,
    field_routes_by_source: Vec<Vec<FieldRoute>>,
    data_routes_by_source: Vec<HashMap<String, Vec<usize>>>,
    single_steps_by_field: Vec<Vec<SingleStep>>,
    ts_univariate_profiles: Vec<TsUnivariateMomentsProfile>,
    ts_univariate_profiles_by_field: Vec<Vec<usize>>,
    ts_bivariate_profiles: Vec<TsBivariateMomentsProfile>,
    ts_bivariate_profiles_by_field: Vec<Vec<usize>>,
    field_queue_seen: Vec<u8>,
    barrier_steps: Vec<BarrierStep>,
    scratch_changed_fields: Vec<usize>,
    last_multi_ts_ns: Option<i64>,
    last_advanced_ts_ns: Option<i64>,
    multi_dirty: bool,
    #[cfg(test)]
    multi_kernel_exec_count: u64,
}

impl Engine for OnlineFactorEngine {
    fn load(&mut self, plan: PhysicalPlan) -> Result<(), EngineError> {
        self.instrument_index.clear();
        self.field_routes_by_source.clear();
        self.data_routes_by_source.clear();
        self.single_steps_by_field.clear();
        self.ts_univariate_profiles.clear();
        self.ts_univariate_profiles_by_field.clear();
        self.ts_bivariate_profiles.clear();
        self.ts_bivariate_profiles_by_field.clear();
        self.field_queue_seen.clear();
        self.barrier_steps.clear();
        self.scratch_changed_fields.clear();
        self.factor_names = None;
        self.factor_index = None;
        self.last_multi_ts_ns = None;
        self.last_advanced_ts_ns = None;
        self.multi_dirty = true;
        #[cfg(test)]
        {
            self.multi_kernel_exec_count = 0;
        }

        for (idx, slot) in plan.universe_slots.iter().enumerate() {
            self.instrument_index.insert(*slot, idx);
        }
        let factor_names: Arc<[String]> = Arc::from(plan.output_names.clone().into_boxed_slice());
        let mut factor_index_map = factor_names
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, name)| (name, idx))
            .collect::<HashMap<_, _>>();
        for (alias, output_slot) in &plan.output_aliases {
            debug_assert!(
                *output_slot < factor_names.len(),
                "output alias slot out of bounds: {} >= {}",
                output_slot,
                factor_names.len()
            );
            factor_index_map.insert(alias.clone(), *output_slot);
        }
        let factor_index = Arc::new(factor_index_map);
        self.factor_names = Some(factor_names);
        self.factor_index = Some(factor_index);

        self.build_field_routes(&plan)?;

        let mut scratch_config = RuntimeScratchConfig::default();
        self.build_single_steps_and_profiles(&plan, &mut scratch_config)?;
        self.build_barrier_steps(&plan, &mut scratch_config)?;

        let mut state = EngineState::from_plan(&plan, scratch_config);
        Self::initialize_constant_fields(&plan, &mut state);
        self.state = Some(state);
        self.plan = Some(plan);
        Ok(())
    }

    fn on_event(&mut self, event: &EventEnvelope) -> Result<(), EngineError> {
        let plan = self.plan.as_ref().ok_or(EngineError::NotLoaded)?;
        let source_idx = event.source_slot as usize;
        let expected_kind =
            plan.source_kinds
                .get(source_idx)
                .copied()
                .ok_or(EngineError::InvalidSourceSlot {
                    slot: event.source_slot,
                })?;
        validate_payload_kind(expected_kind, &event.payload).map_err(|_| {
            EngineError::SourcePayloadMismatch {
                slot: event.source_slot,
            }
        })?;
        let instrument_idx = *self.instrument_index.get(&event.instrument_slot).ok_or(
            EngineError::InvalidInstrumentSlot {
                slot: event.instrument_slot,
            },
        )?;

        self.apply_event_payload_updates(event, source_idx, instrument_idx)?;
        if self.scratch_changed_fields.is_empty() {
            return Ok(());
        }
        self.run_single_path_for_changed_fields(instrument_idx, event.ts_event_ns)?;
        Ok(())
    }

    fn is_graph_ready(&self, ts_ns: i64) -> bool {
        self.state
            .as_ref()
            .map(|s| s.ready_gate.is_ready(ts_ns))
            .unwrap_or(false)
    }

    fn advance(&mut self, ts_ns: i64) -> Result<FeatureFrame, EngineError> {
        let factor_names = self
            .factor_names
            .as_ref()
            .ok_or(EngineError::NotLoaded)?
            .clone();
        let factor_index = self
            .factor_index
            .as_ref()
            .ok_or(EngineError::NotLoaded)?
            .clone();
        self.with_advanced_frame(ts_ns, move |frame| {
            FeatureFrame::with_shared_schema(
                frame.ts_ns,
                frame.instrument_count,
                factor_names,
                factor_index,
                frame.values.to_vec(),
                frame.valid_mask.to_vec(),
                frame.quality_flags.to_vec(),
            )
        })
    }
}

impl OnlineFactorEngine {
    fn apply_event_payload_updates(
        &mut self,
        event: &EventEnvelope,
        source_idx: usize,
        instrument_idx: usize,
    ) -> Result<(), EngineError> {
        let state = self.state.as_mut().ok_or(EngineError::NotLoaded)?;
        self.multi_dirty = true;
        state.quality_flags[instrument_idx] |= event.quality_flags;

        self.scratch_changed_fields.clear();
        let field_routes = &self.field_routes_by_source[source_idx];
        let data_routes = &self.data_routes_by_source[source_idx];
        match &event.payload {
            Payload::Data(data) => {
                Self::collect_changed_fields_from_data_lookup(
                    data_routes,
                    data,
                    state,
                    instrument_idx,
                    event.ts_event_ns,
                    &mut self.scratch_changed_fields,
                );
            }
            payload => Self::collect_changed_fields_from_typed_payload(
                payload,
                field_routes,
                state,
                instrument_idx,
                event.ts_event_ns,
                &mut self.scratch_changed_fields,
            ),
        }
        Ok(())
    }

    fn run_single_path_for_changed_fields(
        &mut self,
        instrument_idx: usize,
        ts_event_ns: i64,
    ) -> Result<(), EngineError> {
        let ts_univariate_profiles_by_field = &self.ts_univariate_profiles_by_field;
        let ts_bivariate_profiles_by_field = &self.ts_bivariate_profiles_by_field;
        let single_steps_by_field = &self.single_steps_by_field;
        let ts_univariate_profiles = &mut self.ts_univariate_profiles;
        let ts_bivariate_profiles = &mut self.ts_bivariate_profiles;
        let field_queue_seen = &mut self.field_queue_seen;
        let scratch_changed_fields = &mut self.scratch_changed_fields;
        let state = self.state.as_mut().ok_or(EngineError::NotLoaded)?;
        field_queue_seen.fill(0);
        Self::dedup_changed_fields(scratch_changed_fields, field_queue_seen);
        let mut cursor = 0usize;
        while cursor < scratch_changed_fields.len() {
            let field_slot = scratch_changed_fields[cursor];
            cursor += 1;
            field_queue_seen[field_slot] = 2;

            for &profile_idx in &ts_univariate_profiles_by_field[field_slot] {
                Self::update_ts_univariate_profile(
                    &mut ts_univariate_profiles[profile_idx],
                    state,
                    instrument_idx,
                    ts_event_ns,
                    field_queue_seen,
                    scratch_changed_fields,
                );
            }
            for &profile_idx in &ts_bivariate_profiles_by_field[field_slot] {
                Self::update_ts_bivariate_profile(
                    &mut ts_bivariate_profiles[profile_idx],
                    state,
                    instrument_idx,
                    ts_event_ns,
                    field_queue_seen,
                    scratch_changed_fields,
                );
            }

            for step in &single_steps_by_field[field_slot] {
                (step.kernel)(state, instrument_idx, step.io, step.params);
                if let Some(derived_field_slot) = step.output_field_slot {
                    if Self::publish_output_field_from_node_output(
                        state,
                        instrument_idx,
                        step.io.output,
                        derived_field_slot,
                        ts_event_ns,
                    ) {
                        Self::enqueue_changed_field(
                            field_queue_seen,
                            scratch_changed_fields,
                            derived_field_slot,
                        );
                    }
                }
            }
        }
        Ok(())
    }

    fn dedup_changed_fields(scratch_changed_fields: &mut Vec<usize>, field_queue_seen: &mut [u8]) {
        let mut write = 0usize;
        for idx in 0..scratch_changed_fields.len() {
            let field_slot = scratch_changed_fields[idx];
            if field_queue_seen[field_slot] != 0 {
                continue;
            }
            field_queue_seen[field_slot] = 1;
            scratch_changed_fields[write] = field_slot;
            write += 1;
        }
        scratch_changed_fields.truncate(write);
    }

    #[inline]
    fn enqueue_changed_field(
        field_queue_seen: &mut [u8],
        scratch_changed_fields: &mut Vec<usize>,
        field_slot: usize,
    ) {
        let state = field_queue_seen[field_slot];
        if state == 1 {
            return;
        }
        field_queue_seen[field_slot] = 1;
        scratch_changed_fields.push(field_slot);
    }

    #[inline]
    fn publish_output_field_from_node_output(
        state: &mut EngineState,
        instrument_idx: usize,
        output_slot: usize,
        output_field_slot: usize,
        ts_ns: i64,
    ) -> bool {
        let out_idx = state.cell_idx(instrument_idx, output_slot);
        let value = if state.node_valid[out_idx] {
            state.node_outputs[out_idx]
        } else {
            f64::NAN
        };
        state
            .field_store
            .update_at(instrument_idx, output_field_slot, value, ts_ns)
    }

    fn build_field_routes(&mut self, plan: &PhysicalPlan) -> Result<(), EngineError> {
        self.field_routes_by_source = vec![Vec::new(); plan.source_kinds.len()];
        self.data_routes_by_source = vec![HashMap::new(); plan.source_kinds.len()];
        for field in &plan.fields {
            let source_slot = field.source_slot as usize;
            if source_slot >= self.field_routes_by_source.len() {
                return Err(EngineError::InvalidSourceSlot {
                    slot: field.source_slot,
                });
            }
            if is_derived_field_name(&field.key.field) || is_const_field_name(&field.key.field) {
                continue;
            }
            if matches!(field.key.source_kind, SourceKind::Data) {
                self.data_routes_by_source[source_slot]
                    .entry(field.key.field.clone())
                    .or_default()
                    .push(field.field_slot);
            } else {
                let accessor = FieldAccessor::from_binding(field.key.source_kind, &field.key.field)
                    .ok_or(EngineError::UnsupportedFieldAccessor {
                        source_kind: source_kind_name(field.key.source_kind),
                        field: field.key.field.clone(),
                    })?;
                self.field_routes_by_source[source_slot].push(FieldRoute {
                    field_slot: field.field_slot,
                    accessor,
                });
            }
        }
        Ok(())
    }

    fn initialize_constant_fields(plan: &PhysicalPlan, state: &mut EngineState) {
        for field in &plan.fields {
            let Some(value) = parse_const_field_value(&field.key.field) else {
                continue;
            };
            for instrument_idx in 0..state.instrument_count {
                state
                    .field_store
                    .update(instrument_idx, field.field_slot, value);
            }
        }
    }

    fn build_single_steps_and_profiles(
        &mut self,
        plan: &PhysicalPlan,
        scratch_config: &mut RuntimeScratchConfig,
    ) -> Result<(), EngineError> {
        self.single_steps_by_field = vec![Vec::new(); plan.fields.len()];
        self.ts_univariate_profiles_by_field = vec![Vec::new(); plan.fields.len()];
        self.ts_bivariate_profiles_by_field = vec![Vec::new(); plan.fields.len()];
        let mut ts_univariate_profile_idx_by_key: HashMap<(usize, usize), usize> = HashMap::new();
        let mut ts_bivariate_profile_idx_by_key: HashMap<(usize, usize, usize), usize> =
            HashMap::new();

        for &node_idx in &plan.single_nodes {
            let node = &plan.nodes[node_idx];
            debug_assert!(
                matches!(node.exec_mode, crate::plan::ExecMode::EventSingle),
                "single_nodes must only contain event-mode nodes"
            );
            debug_assert!(
                node.lineage.source_cardinality() <= 1,
                "event-mode node must be single-source lineage"
            );
            let meta = Self::resolve_op_meta(node, "single")?;
            if node.output_field_slot.is_none()
                && self.try_register_shared_single_node(
                    plan,
                    node,
                    meta,
                    &mut ts_univariate_profile_idx_by_key,
                    &mut ts_bivariate_profile_idx_by_key,
                )
            {
                continue;
            }

            let step = Self::build_single_step(node, meta, scratch_config)?;
            Self::route_single_step_by_inputs(&mut self.single_steps_by_field, node, step);
        }

        self.field_queue_seen = vec![0; plan.fields.len()];
        Ok(())
    }

    fn build_barrier_steps(
        &mut self,
        plan: &PhysicalPlan,
        scratch_config: &mut RuntimeScratchConfig,
    ) -> Result<(), EngineError> {
        for &node_idx in &plan.multi_nodes {
            let node = &plan.nodes[node_idx];
            debug_assert!(
                node.exec_mode.is_barrier(),
                "multi_nodes must only contain barrier-mode nodes"
            );
            let meta = Self::resolve_op_meta(node, "multi")?;
            let step = Self::build_barrier_step(node, meta, scratch_config)?;
            self.barrier_steps.push(step);
        }
        Ok(())
    }

    fn resolve_op_meta(
        node: &PhysicalNode,
        phase: &'static str,
    ) -> Result<&'static OpMeta, EngineError> {
        OperatorRegistry::get_by_op(node.op)
            .ok_or_else(|| Self::unsupported_kernel_error(node, phase))
    }

    fn try_register_shared_single_node(
        &mut self,
        plan: &PhysicalPlan,
        node: &PhysicalNode,
        meta: &OpMeta,
        ts_univariate_profile_idx_by_key: &mut HashMap<(usize, usize), usize>,
        ts_bivariate_profile_idx_by_key: &mut HashMap<(usize, usize, usize), usize>,
    ) -> bool {
        match meta.shared_family {
            Some(SharedFamily::TsUnivariateMoments { view }) => {
                let field_slot = node.input_slot(0);
                let window = node
                    .param
                    .window()
                    .expect("ts univariate shared profile requires window param");
                let key = (field_slot, window);
                let profile_idx = if let Some(existing) = ts_univariate_profile_idx_by_key.get(&key)
                {
                    *existing
                } else {
                    let idx = self.ts_univariate_profiles.len();
                    self.ts_univariate_profiles
                        .push(TsUnivariateMomentsProfile {
                            field_slot,
                            window,
                            required_order: view.required_order(),
                            consumers: Vec::new(),
                            states: (0..plan.universe_slots.len())
                                .map(|_| TsUnivariateMomentsState::with_window(window))
                                .collect(),
                        });
                    self.ts_univariate_profiles_by_field[field_slot].push(idx);
                    ts_univariate_profile_idx_by_key.insert(key, idx);
                    idx
                };
                let profile = &mut self.ts_univariate_profiles[profile_idx];
                profile.required_order = profile.required_order.max(view.required_order());
                profile.consumers.push(TsUnivariateMomentsConsumer {
                    output_slot: node.output_slot,
                    output_field_slot: node.output_field_slot,
                    view,
                });
                true
            }
            Some(SharedFamily::TsBivariateMoments { view }) => {
                let lhs_field_slot = node.input_slot(0);
                let rhs_field_slot = node.input_slot(1);
                let window = node
                    .param
                    .window()
                    .expect("ts bivariate shared profile requires window param");
                let key = (lhs_field_slot, rhs_field_slot, window);
                let profile_idx = if let Some(existing) = ts_bivariate_profile_idx_by_key.get(&key)
                {
                    *existing
                } else {
                    let idx = self.ts_bivariate_profiles.len();
                    self.ts_bivariate_profiles.push(TsBivariateMomentsProfile {
                        lhs_field_slot,
                        rhs_field_slot,
                        window,
                        consumers: Vec::new(),
                        states: (0..plan.universe_slots.len())
                            .map(|_| TsBivariateMomentsState::with_window(window))
                            .collect(),
                    });
                    self.ts_bivariate_profiles_by_field[lhs_field_slot].push(idx);
                    if rhs_field_slot != lhs_field_slot {
                        self.ts_bivariate_profiles_by_field[rhs_field_slot].push(idx);
                    }
                    ts_bivariate_profile_idx_by_key.insert(key, idx);
                    idx
                };
                self.ts_bivariate_profiles[profile_idx].consumers.push(
                    TsBivariateMomentsConsumer {
                        output_slot: node.output_slot,
                        output_field_slot: node.output_field_slot,
                        view,
                    },
                );
                true
            }
            None => false,
        }
    }

    fn build_single_step(
        node: &PhysicalNode,
        meta: &OpMeta,
        scratch_config: &mut RuntimeScratchConfig,
    ) -> Result<SingleStep, EngineError> {
        apply_scratch_profiles(scratch_config, meta.scratch_profiles);
        let kernel = meta
            .single_kernel
            .ok_or_else(|| Self::unsupported_kernel_error(node, "single"))?;
        let params = meta.single_param_spec.build(meta.name, node.param)?;
        Ok(SingleStep {
            io: KernelIo::from_slice(
                &node.input_field_slots[..node.input_count as usize],
                node.output_slot,
            ),
            output_field_slot: node.output_field_slot,
            params,
            kernel,
        })
    }

    fn route_single_step_by_inputs(
        single_steps_by_field: &mut [Vec<SingleStep>],
        node: &PhysicalNode,
        step: SingleStep,
    ) {
        let mut routed_inputs = [usize::MAX; MAX_NODE_INPUTS];
        let mut routed_count = 0usize;
        for idx in 0..node.input_count as usize {
            let field_slot = node.input_slot(idx);
            if routed_inputs[..routed_count].contains(&field_slot) {
                continue;
            }
            routed_inputs[routed_count] = field_slot;
            routed_count += 1;
            single_steps_by_field[field_slot].push(step);
        }
    }

    fn build_barrier_step(
        node: &PhysicalNode,
        meta: &OpMeta,
        scratch_config: &mut RuntimeScratchConfig,
    ) -> Result<BarrierStep, EngineError> {
        apply_scratch_profiles(scratch_config, meta.scratch_profiles);
        let io = KernelIo::from_slice(
            &node.input_field_slots[..node.input_count as usize],
            node.output_slot,
        );
        match node.exec_mode {
            crate::plan::ExecMode::BarrierMulti => {
                let kernel = meta
                    .multi_kernel
                    .ok_or_else(|| Self::unsupported_kernel_error(node, "multi"))?;
                let params = meta.multi_param_spec.build(meta.name, node.param)?;
                Ok(BarrierStep {
                    node_id: node.node_id,
                    io,
                    output_field_slot: node.output_field_slot,
                    params,
                    kernel: BarrierKernel::Batch(kernel),
                })
            }
            crate::plan::ExecMode::BarrierSingle => {
                debug_assert!(
                    node.lineage.barrier_tainted,
                    "barrier-single should only be used for barrier-tainted lineage"
                );
                // Non-CS operators can still be forced into barrier clock when
                // input sources are not single-stream aligned (including CS-tainted ancestry).
                let kernel = meta
                    .single_kernel
                    .ok_or_else(|| Self::unsupported_kernel_error(node, "multi"))?;
                let params = meta.single_param_spec.build(meta.name, node.param)?;
                Ok(BarrierStep {
                    node_id: node.node_id,
                    io,
                    output_field_slot: node.output_field_slot,
                    params,
                    kernel: BarrierKernel::PerInstrument(kernel),
                })
            }
            crate::plan::ExecMode::EventSingle => {
                Err(Self::unsupported_kernel_error(node, "multi"))
            }
        }
    }

    fn unsupported_kernel_error(node: &PhysicalNode, phase: &'static str) -> EngineError {
        EngineError::UnsupportedKernel {
            node_id: node.node_id,
            op: format!("{:?}", node.op),
            phase,
        }
    }

    /// Zero-allocation consumption path for latency-critical users.
    ///
    /// The callback receives borrowed frame slices valid only during this call.
    pub fn with_advanced_frame<R>(
        &mut self,
        ts_ns: i64,
        consume: impl FnOnce(BorrowedFeatureFrame<'_>) -> R,
    ) -> Result<R, EngineError> {
        self.run_multi_for_ts(ts_ns)?;
        let out = {
            let state = self.state.as_ref().ok_or(EngineError::NotLoaded)?;
            let factor_names = self.factor_names.as_ref().ok_or(EngineError::NotLoaded)?;
            let factor_index = self.factor_index.as_ref().ok_or(EngineError::NotLoaded)?;
            let frame = BorrowedFeatureFrame {
                ts_ns,
                instrument_count: state.instrument_count,
                factor_count: factor_names.len(),
                factor_names: factor_names.as_ref(),
                factor_index: factor_index.as_ref(),
                values: &state.node_outputs,
                valid_mask: &state.node_valid,
                quality_flags: &state.quality_flags,
            };
            consume(frame)
        };
        if let Some(state) = self.state.as_mut() {
            state.quality_flags.fill(0);
        }
        Ok(out)
    }

    /// Copying path with allocation reuse controlled by caller-owned buffers.
    pub fn advance_into_buffers(
        &mut self,
        ts_ns: i64,
        buffers: &mut FeatureFrameBuffers,
    ) -> Result<(usize, usize), EngineError> {
        self.with_advanced_frame(ts_ns, |frame| {
            buffers.ensure_shape(frame.instrument_count, frame.factor_count);
            buffers.values.copy_from_slice(frame.values);
            buffers.valid_mask.copy_from_slice(frame.valid_mask);
            buffers.quality_flags.copy_from_slice(frame.quality_flags);
            (frame.instrument_count, frame.factor_count)
        })
    }

    /// Owned-frame reuse path: caller reuses one `FeatureFrame` across ticks.
    pub fn advance_in_place(
        &mut self,
        ts_ns: i64,
        out: &mut FeatureFrame,
    ) -> Result<(), EngineError> {
        let factor_names = self
            .factor_names
            .as_ref()
            .ok_or(EngineError::NotLoaded)?
            .clone();
        let factor_index = self
            .factor_index
            .as_ref()
            .ok_or(EngineError::NotLoaded)?
            .clone();
        self.with_advanced_frame(ts_ns, |frame| {
            out.ensure_shared_schema(frame.instrument_count, factor_names, factor_index);
            out.overwrite_from_slices(
                frame.ts_ns,
                frame.values,
                frame.valid_mask,
                frame.quality_flags,
            );
        })?;
        Ok(())
    }

    fn run_multi_for_ts(&mut self, ts_ns: i64) -> Result<(), EngineError> {
        let plan = self.plan.as_ref().ok_or(EngineError::NotLoaded)?;
        if let Some(last_ts_ns) = self.last_advanced_ts_ns {
            if ts_ns < last_ts_ns {
                return Err(EngineError::NonMonotonicAdvance {
                    current_ts_ns: ts_ns,
                    last_ts_ns,
                });
            }
        }
        let state = self.state.as_mut().ok_or(EngineError::NotLoaded)?;
        let ready = state.ready_gate.is_ready(ts_ns);
        if matches!(plan.policy, AdvancePolicy::StrictAllReady) && !ready {
            return Err(EngineError::NotReady { ts_ns });
        }
        if !ready && matches!(plan.policy, AdvancePolicy::ForceWithLast) {
            for flag in &mut state.quality_flags {
                *flag |= QUALITY_FORCED_ADVANCE;
            }
        }
        if self.last_multi_ts_ns == Some(ts_ns) && !self.multi_dirty {
            self.last_advanced_ts_ns = Some(ts_ns);
            return Ok(());
        }
        for step in &self.barrier_steps {
            Self::execute_barrier_step(plan, state, step, ts_ns);
        }
        self.last_multi_ts_ns = Some(ts_ns);
        self.last_advanced_ts_ns = Some(ts_ns);
        self.multi_dirty = false;
        #[cfg(test)]
        {
            self.multi_kernel_exec_count += self.barrier_steps.len() as u64;
        }
        Ok(())
    }

    #[inline]
    fn execute_barrier_step(
        plan: &PhysicalPlan,
        state: &mut EngineState,
        step: &BarrierStep,
        ts_ns: i64,
    ) {
        match step.kernel {
            BarrierKernel::Batch(kernel) => {
                debug_assert!(
                    plan.nodes[step.node_id].exec_mode == crate::plan::ExecMode::BarrierMulti
                );
                kernel(plan, state, step.io, step.params);
                if let Some(output_field_slot) = step.output_field_slot {
                    Self::publish_output_field_for_all_instruments(
                        state,
                        step.io.output,
                        output_field_slot,
                        ts_ns,
                    );
                }
            }
            BarrierKernel::PerInstrument(kernel) => {
                debug_assert!(
                    plan.nodes[step.node_id].exec_mode == crate::plan::ExecMode::BarrierSingle
                );
                for instrument_idx in 0..state.instrument_count {
                    kernel(state, instrument_idx, step.io, step.params);
                    Self::publish_output_field_if_needed(
                        state,
                        instrument_idx,
                        step.io.output,
                        step.output_field_slot,
                        ts_ns,
                    );
                }
            }
        }
    }

    fn update_ts_univariate_profile(
        profile: &mut TsUnivariateMomentsProfile,
        state: &mut EngineState,
        instrument_idx: usize,
        ts_event_ns: i64,
        field_queue_seen: &mut [u8],
        scratch_changed_fields: &mut Vec<usize>,
    ) {
        let input = state.field_store.get(instrument_idx, profile.field_slot);
        if !input.has_latest {
            for consumer in &profile.consumers {
                state.set_node_output(instrument_idx, consumer.output_slot, f64::NAN, false);
                if let Some(output_field_slot) = consumer.output_field_slot {
                    if Self::publish_output_field_from_node_output(
                        state,
                        instrument_idx,
                        consumer.output_slot,
                        output_field_slot,
                        ts_event_ns,
                    ) {
                        Self::enqueue_changed_field(
                            field_queue_seen,
                            scratch_changed_fields,
                            output_field_slot,
                        );
                    }
                }
            }
            return;
        }
        let added = input.latest;
        let profile_state = &mut profile.states[instrument_idx];
        if profile_state.last_input_generation == input.generation {
            return;
        }
        profile_state.last_input_generation = input.generation;
        profile_state.push_at(added, profile.required_order, ts_event_ns);

        if let Some(snapshot) =
            ts_univariate_moments_snapshot(profile_state, profile.window, profile.required_order)
        {
            for consumer in &profile.consumers {
                let value = match consumer.view {
                    TsUnivariateMomentsView::Sum => snapshot.sum,
                    TsUnivariateMomentsView::Mean => snapshot.mean,
                    TsUnivariateMomentsView::Std => snapshot.std,
                    TsUnivariateMomentsView::Var => snapshot.var,
                    TsUnivariateMomentsView::Zscore => {
                        if snapshot.std.is_finite() && snapshot.std > 0.0 {
                            (added - snapshot.mean) / snapshot.std
                        } else {
                            0.0
                        }
                    }
                    TsUnivariateMomentsView::Skew => snapshot.skew,
                    TsUnivariateMomentsView::Kurt => snapshot.kurt,
                };
                if value.is_finite() {
                    state.set_node_output(instrument_idx, consumer.output_slot, value, true);
                } else {
                    state.set_node_output(instrument_idx, consumer.output_slot, f64::NAN, false);
                }
                if let Some(output_field_slot) = consumer.output_field_slot {
                    if Self::publish_output_field_from_node_output(
                        state,
                        instrument_idx,
                        consumer.output_slot,
                        output_field_slot,
                        ts_event_ns,
                    ) {
                        Self::enqueue_changed_field(
                            field_queue_seen,
                            scratch_changed_fields,
                            output_field_slot,
                        );
                    }
                }
            }
        } else {
            for consumer in &profile.consumers {
                state.set_node_output(instrument_idx, consumer.output_slot, f64::NAN, false);
                if let Some(output_field_slot) = consumer.output_field_slot {
                    if Self::publish_output_field_from_node_output(
                        state,
                        instrument_idx,
                        consumer.output_slot,
                        output_field_slot,
                        ts_event_ns,
                    ) {
                        Self::enqueue_changed_field(
                            field_queue_seen,
                            scratch_changed_fields,
                            output_field_slot,
                        );
                    }
                }
            }
        }
    }

    fn update_ts_bivariate_profile(
        profile: &mut TsBivariateMomentsProfile,
        state: &mut EngineState,
        instrument_idx: usize,
        ts_event_ns: i64,
        field_queue_seen: &mut [u8],
        scratch_changed_fields: &mut Vec<usize>,
    ) {
        let lhs_input = state
            .field_store
            .get(instrument_idx, profile.lhs_field_slot);
        let rhs_input = state
            .field_store
            .get(instrument_idx, profile.rhs_field_slot);
        if !lhs_input.has_latest || !rhs_input.has_latest {
            for consumer in &profile.consumers {
                state.set_node_output(instrument_idx, consumer.output_slot, f64::NAN, false);
                if let Some(output_field_slot) = consumer.output_field_slot {
                    if Self::publish_output_field_from_node_output(
                        state,
                        instrument_idx,
                        consumer.output_slot,
                        output_field_slot,
                        ts_event_ns,
                    ) {
                        Self::enqueue_changed_field(
                            field_queue_seen,
                            scratch_changed_fields,
                            output_field_slot,
                        );
                    }
                }
            }
            return;
        }
        let lhs = lhs_input.latest;
        let rhs = rhs_input.latest;

        let profile_state = &mut profile.states[instrument_idx];
        if profile_state.last_lhs_generation == lhs_input.generation
            && profile_state.last_rhs_generation == rhs_input.generation
        {
            return;
        }
        profile_state.last_lhs_generation = lhs_input.generation;
        profile_state.last_rhs_generation = rhs_input.generation;
        profile_state.push_at(lhs, rhs, ts_event_ns);
        if let Some(snapshot) = ts_bivariate_moments_snapshot(profile_state, profile.window) {
            for consumer in &profile.consumers {
                let value = match consumer.view {
                    TsBivariateMomentsView::Cov => snapshot.cov,
                    TsBivariateMomentsView::Beta => snapshot.beta,
                    TsBivariateMomentsView::Corr => snapshot.corr,
                    TsBivariateMomentsView::RegressionSlope => snapshot.slope,
                };
                if value.is_finite() {
                    state.set_node_output(instrument_idx, consumer.output_slot, value, true);
                } else {
                    state.set_node_output(instrument_idx, consumer.output_slot, f64::NAN, false);
                }
                if let Some(output_field_slot) = consumer.output_field_slot {
                    if Self::publish_output_field_from_node_output(
                        state,
                        instrument_idx,
                        consumer.output_slot,
                        output_field_slot,
                        ts_event_ns,
                    ) {
                        Self::enqueue_changed_field(
                            field_queue_seen,
                            scratch_changed_fields,
                            output_field_slot,
                        );
                    }
                }
            }
        } else {
            for consumer in &profile.consumers {
                state.set_node_output(instrument_idx, consumer.output_slot, f64::NAN, false);
                if let Some(output_field_slot) = consumer.output_field_slot {
                    if Self::publish_output_field_from_node_output(
                        state,
                        instrument_idx,
                        consumer.output_slot,
                        output_field_slot,
                        ts_event_ns,
                    ) {
                        Self::enqueue_changed_field(
                            field_queue_seen,
                            scratch_changed_fields,
                            output_field_slot,
                        );
                    }
                }
            }
        }
    }

    fn collect_changed_fields_from_routes(
        routes: &[FieldRoute],
        state: &mut EngineState,
        instrument_idx: usize,
        ts_event_ns: i64,
        scratch_changed_fields: &mut Vec<usize>,
        mut read_value: impl FnMut(&FieldAccessor) -> Option<f64>,
    ) {
        for route in routes {
            if let Some(value) = read_value(&route.accessor) {
                Self::mark_field_update(
                    state,
                    instrument_idx,
                    route.field_slot,
                    value,
                    ts_event_ns,
                    scratch_changed_fields,
                );
            }
        }
    }

    fn collect_changed_fields_from_data_lookup(
        routes: &HashMap<String, Vec<usize>>,
        data: &crate::types::DataLite,
        state: &mut EngineState,
        instrument_idx: usize,
        ts_event_ns: i64,
        scratch_changed_fields: &mut Vec<usize>,
    ) {
        for (name, value) in &data.values {
            if let Some(field_slots) = routes.get(name) {
                for &field_slot in field_slots {
                    Self::mark_field_update(
                        state,
                        instrument_idx,
                        field_slot,
                        *value,
                        ts_event_ns,
                        scratch_changed_fields,
                    );
                }
            }
        }
    }

    fn collect_changed_fields_from_typed_payload(
        payload: &Payload,
        field_routes: &[FieldRoute],
        state: &mut EngineState,
        instrument_idx: usize,
        ts_event_ns: i64,
        scratch_changed_fields: &mut Vec<usize>,
    ) {
        match payload {
            Payload::Bar(bar) => Self::collect_changed_fields_from_routes(
                field_routes,
                state,
                instrument_idx,
                ts_event_ns,
                scratch_changed_fields,
                |accessor| accessor.read_bar(bar),
            ),
            Payload::TradeTick(trade) => Self::collect_changed_fields_from_routes(
                field_routes,
                state,
                instrument_idx,
                ts_event_ns,
                scratch_changed_fields,
                |accessor| accessor.read_trade(trade),
            ),
            Payload::QuoteTick(quote) => Self::collect_changed_fields_from_routes(
                field_routes,
                state,
                instrument_idx,
                ts_event_ns,
                scratch_changed_fields,
                |accessor| accessor.read_quote(quote),
            ),
            Payload::OrderBookSnapshot(_) | Payload::Data(_) => {}
        }
    }

    #[inline]
    fn mark_field_update(
        state: &mut EngineState,
        instrument_idx: usize,
        field_slot: usize,
        value: f64,
        ts_event_ns: i64,
        scratch_changed_fields: &mut Vec<usize>,
    ) {
        if state
            .field_store
            .update_at(instrument_idx, field_slot, value, ts_event_ns)
        {
            state
                .ready_gate
                .mark(ts_event_ns, instrument_idx, field_slot);
            scratch_changed_fields.push(field_slot);
        }
    }

    #[inline]
    fn publish_output_field_if_needed(
        state: &mut EngineState,
        instrument_idx: usize,
        output_slot: usize,
        output_field_slot: Option<usize>,
        ts_ns: i64,
    ) {
        if let Some(output_field_slot) = output_field_slot {
            let _ = Self::publish_output_field_from_node_output(
                state,
                instrument_idx,
                output_slot,
                output_field_slot,
                ts_ns,
            );
        }
    }

    #[inline]
    fn publish_output_field_for_all_instruments(
        state: &mut EngineState,
        output_slot: usize,
        output_field_slot: usize,
        ts_ns: i64,
    ) {
        for instrument_idx in 0..state.instrument_count {
            let _ = Self::publish_output_field_from_node_output(
                state,
                instrument_idx,
                output_slot,
                output_field_slot,
                ts_ns,
            );
        }
    }

    #[cfg(test)]
    pub(crate) fn debug_cs_rank_scratch_capacity(&self) -> Option<usize> {
        self.state
            .as_ref()
            .and_then(|s| s.scratch.rank_pairs_capacity())
    }

    #[cfg(test)]
    pub(crate) fn debug_tmp_f64_scratch_capacity(&self) -> Option<usize> {
        self.state
            .as_ref()
            .and_then(|s| s.scratch.tmp_f64_capacity())
    }

    #[cfg(test)]
    pub(crate) fn debug_tmp_usize_scratch_capacity(&self) -> Option<usize> {
        self.state
            .as_ref()
            .and_then(|s| s.scratch.tmp_usize_capacity())
    }

    #[cfg(test)]
    pub(crate) fn debug_multi_kernel_exec_count(&self) -> u64 {
        self.multi_kernel_exec_count
    }

    #[cfg(test)]
    pub(crate) fn debug_ts_univariate_profile_count(&self) -> usize {
        self.ts_univariate_profiles.len()
    }

    #[cfg(test)]
    pub(crate) fn debug_ts_univariate_consumer_count(&self, profile_idx: usize) -> Option<usize> {
        self.ts_univariate_profiles
            .get(profile_idx)
            .map(|p| p.consumers.len())
    }

    #[cfg(test)]
    pub(crate) fn debug_ts_bivariate_profile_count(&self) -> usize {
        self.ts_bivariate_profiles.len()
    }

    #[cfg(test)]
    pub(crate) fn debug_ts_bivariate_consumer_count(&self, profile_idx: usize) -> Option<usize> {
        self.ts_bivariate_profiles
            .get(profile_idx)
            .map(|p| p.consumers.len())
    }
}

fn ts_univariate_moments_snapshot(
    state: &TsUnivariateMomentsState,
    window: usize,
    required_order: u8,
) -> Option<TsUnivariateMomentsSnapshot> {
    if window == 0 || state.len < window || state.nan_count > 0 {
        return None;
    }
    let n = window as f64;
    let mut sum = 0.0_f64;
    let mut sum_c = 0.0_f64;
    for lag in 0..window {
        let idx = (state.write + state.cap - 1 - lag) % state.cap;
        let v = state.ring[idx];
        if !v.is_finite() {
            return None;
        }
        let y = v - sum_c;
        let t = sum + y;
        sum_c = (t - sum) - y;
        sum = t;
    }
    let mean = sum / n;
    let mut var = f64::NAN;
    let mut std = f64::NAN;
    let mut skew = f64::NAN;
    let mut kurt = f64::NAN;

    if required_order >= 2 {
        let mut m2 = 0.0_f64;
        let mut m2_c = 0.0_f64;
        let mut m3 = 0.0_f64;
        let mut m3_c = 0.0_f64;
        let mut m4 = 0.0_f64;
        let mut m4_c = 0.0_f64;
        for lag in 0..window {
            let idx = (state.write + state.cap - 1 - lag) % state.cap;
            let v = state.ring[idx];
            let d = v - mean;
            let d2 = d * d;
            let y2 = d2 - m2_c;
            let t2 = m2 + y2;
            m2_c = (t2 - m2) - y2;
            m2 = t2;
            if required_order >= 3 {
                let d3 = d2 * d;
                let y3 = d3 - m3_c;
                let t3 = m3 + y3;
                m3_c = (t3 - m3) - y3;
                m3 = t3;
                if required_order >= 4 {
                    let d4 = d2 * d2;
                    let y4 = d4 - m4_c;
                    let t4 = m4 + y4;
                    m4_c = (t4 - m4) - y4;
                    m4 = t4;
                }
            }
        }
        if n > 1.0 {
            var = (m2 / (n - 1.0)).max(0.0);
            std = var.sqrt();
        }
        if required_order >= 3 && std.is_finite() && std > 0.0 && n > 2.0 {
            let denom = (n - 1.0) * (n - 2.0) * std.powi(3);
            if denom.abs() > f64::EPSILON {
                skew = (n * m3) / denom;
            }
        }
        if required_order >= 4 && std.is_finite() && std > 0.0 && n > 3.0 {
            let denom = (n - 1.0) * (n - 2.0) * (n - 3.0) * std.powi(4);
            if denom.abs() > f64::EPSILON {
                let term1 = (n * (n + 1.0) * m4) / denom;
                let term2 = 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0));
                kurt = term1 - term2;
            }
        }
    }

    Some(TsUnivariateMomentsSnapshot {
        sum,
        mean,
        var,
        std,
        skew,
        kurt,
    })
}

fn ts_bivariate_moments_snapshot(
    state: &TsBivariateMomentsState,
    window: usize,
) -> Option<TsBivariateMomentsSnapshot> {
    if window == 0 || state.len < window || state.nan_count > 0 {
        return None;
    }
    let n = window as f64;
    let mut sum_x = 0.0_f64;
    let mut sum_x_c = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_y_c = 0.0_f64;
    for lag in 0..window {
        let idx = (state.write + state.cap - 1 - lag) % state.cap;
        let x = state.ring_x[idx];
        let y = state.ring_y[idx];
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let yx = x - sum_x_c;
        let tx = sum_x + yx;
        sum_x_c = (tx - sum_x) - yx;
        sum_x = tx;

        let yy = y - sum_y_c;
        let ty = sum_y + yy;
        sum_y_c = (ty - sum_y) - yy;
        sum_y = ty;
    }
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    let mut cov_num = 0.0_f64;
    let mut cov_c = 0.0_f64;
    let mut var_x_num = 0.0_f64;
    let mut var_x_c = 0.0_f64;
    let mut var_y_num = 0.0_f64;
    let mut var_y_c = 0.0_f64;
    for lag in 0..window {
        let idx = (state.write + state.cap - 1 - lag) % state.cap;
        let x = state.ring_x[idx];
        let y = state.ring_y[idx];
        let dx = x - mean_x;
        let dy = y - mean_y;
        let cov_inc = dx * dy;
        let yc = cov_inc - cov_c;
        let tc = cov_num + yc;
        cov_c = (tc - cov_num) - yc;
        cov_num = tc;

        let vx_inc = dx * dx;
        let yx = vx_inc - var_x_c;
        let tx = var_x_num + yx;
        var_x_c = (tx - var_x_num) - yx;
        var_x_num = tx;

        let vy_inc = dy * dy;
        let yy = vy_inc - var_y_c;
        let ty = var_y_num + yy;
        var_y_c = (ty - var_y_num) - yy;
        var_y_num = ty;
    }
    let cov = if n > 1.0 {
        cov_num / (n - 1.0)
    } else {
        f64::NAN
    };
    let beta = if var_y_num > VAR_NUM_EPS {
        cov_num / var_y_num
    } else {
        f64::NAN
    };
    let corr = if var_x_num > VAR_NUM_EPS && var_y_num > VAR_NUM_EPS {
        cov_num / (var_x_num.sqrt() * var_y_num.sqrt())
    } else {
        f64::NAN
    };
    let slope = if var_x_num > VAR_NUM_EPS {
        cov_num / var_x_num
    } else {
        f64::NAN
    };
    Some(TsBivariateMomentsSnapshot {
        cov,
        beta,
        corr,
        slope,
    })
}

fn validate_payload_kind(expected: SourceKind, payload: &Payload) -> Result<(), EngineError> {
    match (payload, expected) {
        (Payload::Bar(_), SourceKind::Bar) => Ok(()),
        (Payload::TradeTick(_), SourceKind::TradeTick) => Ok(()),
        (Payload::QuoteTick(_), SourceKind::QuoteTick) => Ok(()),
        (Payload::OrderBookSnapshot(_), SourceKind::OrderBookSnapshot) => Ok(()),
        (Payload::Data(_), SourceKind::Data) => Ok(()),
        _ => Err(EngineError::PayloadMismatch),
    }
}

#[inline]
fn is_derived_field_name(field: &str) -> bool {
    field.starts_with("__derived__")
}

#[inline]
fn is_const_field_name(field: &str) -> bool {
    field.starts_with(CONST_FIELD_PREFIX)
}

fn parse_const_field_value(field: &str) -> Option<f64> {
    let bits = field.strip_prefix(CONST_FIELD_PREFIX)?;
    if bits.len() != 16 {
        return None;
    }
    let raw = u64::from_str_radix(bits, 16).ok()?;
    Some(f64::from_bits(raw))
}

const fn source_kind_name(kind: SourceKind) -> &'static str {
    match kind {
        SourceKind::Bar => "bar",
        SourceKind::TradeTick => "trade_tick",
        SourceKind::QuoteTick => "quote_tick",
        SourceKind::OrderBookSnapshot => "orderbook_snapshot",
        SourceKind::Data => "data",
    }
}

fn apply_scratch_profiles(config: &mut RuntimeScratchConfig, profiles: &[ScratchProfile]) {
    for profile in profiles {
        match profile {
            ScratchProfile::RankPairs => config.enable_rank_pairs = true,
            ScratchProfile::TmpF64 => config.enable_tmp_f64 = true,
            ScratchProfile::TmpUsize => config.enable_tmp_usize = true,
        }
    }
}
