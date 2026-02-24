use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SourceKind {
    Bar,
    TradeTick,
    QuoteTick,
    OrderBookSnapshot,
    Data,
}

/// Compact bar payload using OHLCV field names.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BarLite {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Compact trade tick payload.
#[derive(Debug, Clone, PartialEq)]
pub struct TradeTickLite {
    pub price: f64,
    pub size: f64,
    pub aggressor_side: i8,
    pub trade_id: String,
}

/// Compact quote tick payload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuoteTickLite {
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: f64,
    pub ask_size: f64,
}

/// OrderBookSnapshot payload (top-N levels).
#[derive(Debug, Clone, PartialEq)]
pub struct OrderBookSnapshotLite {
    pub bid_prices: Vec<f64>,
    pub ask_prices: Vec<f64>,
    pub bid_sizes: Vec<f64>,
    pub ask_sizes: Vec<f64>,
}

/// Generic `on_data` payload for custom numeric fields.
#[derive(Debug, Clone, PartialEq)]
pub struct DataLite {
    pub values: Vec<(String, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Payload {
    Bar(BarLite),
    TradeTick(TradeTickLite),
    QuoteTick(QuoteTickLite),
    OrderBookSnapshot(OrderBookSnapshotLite),
    Data(DataLite),
}

#[derive(Debug, Clone, PartialEq)]
pub struct EventEnvelope {
    /// Event time (ns) used for scheduling and graph readiness.
    pub ts_event_ns: i64,
    /// Source object creation time (ns).
    pub ts_init_ns: i64,
    /// Stable tie-breaker for same `ts_event_ns`.
    pub seq: u32,
    /// Bound source index from PhysicalPlan. Replaces runtime string matching.
    pub source_slot: u16,
    /// Dense runtime index for instrument state lookup.
    pub instrument_slot: u32,
    /// Data-quality / replay / force-advance markers.
    pub quality_flags: u32,
    pub payload: Payload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvancePolicy {
    StrictAllReady,
    ForceWithLast,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompileOptions {
    pub default_source_kind: SourceKind,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            default_source_kind: SourceKind::Bar,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FactorRequest {
    pub exprs: Vec<String>,
    pub outputs: Vec<String>,
    pub opts: CompileOptions,
}

impl FactorRequest {
    pub fn new(exprs: Vec<String>) -> Self {
        Self {
            exprs,
            outputs: Vec::new(),
            opts: CompileOptions::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Universe {
    pub instrument_slots: Vec<u32>,
}

impl Universe {
    pub fn new(instrument_slots: Vec<u32>) -> Self {
        Self { instrument_slots }
    }

    pub fn len(&self) -> usize {
        self.instrument_slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instrument_slots.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InputFieldCatalog {
    fields: BTreeSet<String>,
}

impl InputFieldCatalog {
    pub fn new(fields: impl IntoIterator<Item = String>) -> Self {
        Self {
            fields: fields.into_iter().collect(),
        }
    }

    pub fn contains(&self, field: &str) -> bool {
        self.fields.contains(field)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureFrame {
    pub ts_ns: i64,
    pub instrument_count: usize,
    pub factor_count: usize,
    pub factor_names: Arc<[String]>,
    factor_index: Arc<HashMap<String, usize>>,
    /// Row-major matrix: `values[instrument_idx * factor_count + factor_idx]`.
    pub values: Vec<f64>,
    pub valid_mask: Vec<bool>,
    /// Per-instrument quality flags.
    pub quality_flags: Vec<u32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FeatureFrameBuffers {
    pub values: Vec<f64>,
    pub valid_mask: Vec<bool>,
    pub quality_flags: Vec<u32>,
}

impl FeatureFrameBuffers {
    pub fn ensure_shape(&mut self, instrument_count: usize, factor_count: usize) {
        resize_frame_storage(
            &mut self.values,
            &mut self.valid_mask,
            &mut self.quality_flags,
            instrument_count,
            factor_count,
        );
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BorrowedFeatureFrame<'a> {
    pub ts_ns: i64,
    pub instrument_count: usize,
    pub factor_count: usize,
    pub factor_names: &'a [String],
    pub(crate) factor_index: &'a HashMap<String, usize>,
    pub values: &'a [f64],
    pub valid_mask: &'a [bool],
    pub quality_flags: &'a [u32],
}

impl<'a> BorrowedFeatureFrame<'a> {
    #[inline]
    pub fn factor_idx(&self, factor_name: &str) -> Option<usize> {
        self.factor_index.get(factor_name).copied()
    }

    #[inline]
    pub fn value_at(&self, instrument_idx: usize, factor_idx: usize) -> Option<f64> {
        let idx = self.cell_idx(instrument_idx, factor_idx)?;
        Some(self.values[idx])
    }

    #[inline]
    pub fn factor_value(&self, instrument_idx: usize, factor_name: &str) -> Option<f64> {
        let factor_idx = self.factor_idx(factor_name)?;
        self.value_at(instrument_idx, factor_idx)
    }

    #[inline]
    fn cell_idx(&self, instrument_idx: usize, factor_idx: usize) -> Option<usize> {
        checked_cell_idx(
            instrument_idx,
            factor_idx,
            self.instrument_count,
            self.factor_count,
        )
    }
}

impl FeatureFrame {
    pub fn new(
        ts_ns: i64,
        instrument_count: usize,
        factor_names: Vec<String>,
        values: Vec<f64>,
        valid_mask: Vec<bool>,
        quality_flags: Vec<u32>,
    ) -> Self {
        let factor_count = factor_names.len();
        assert_frame_shape(
            instrument_count,
            factor_count,
            values.len(),
            valid_mask.len(),
            quality_flags.len(),
        );
        let factor_names: Arc<[String]> = Arc::from(factor_names.into_boxed_slice());
        let factor_index = Arc::new(
            factor_names
                .iter()
                .cloned()
                .enumerate()
                .map(|(idx, name)| (name, idx))
                .collect::<HashMap<_, _>>(),
        );
        Self {
            ts_ns,
            instrument_count,
            factor_count,
            factor_names,
            factor_index,
            values,
            valid_mask,
            quality_flags,
        }
    }

    pub fn with_shared_schema(
        ts_ns: i64,
        instrument_count: usize,
        factor_names: Arc<[String]>,
        factor_index: Arc<HashMap<String, usize>>,
        values: Vec<f64>,
        valid_mask: Vec<bool>,
        quality_flags: Vec<u32>,
    ) -> Self {
        let factor_count = factor_names.len();
        assert_frame_shape(
            instrument_count,
            factor_count,
            values.len(),
            valid_mask.len(),
            quality_flags.len(),
        );
        Self {
            ts_ns,
            instrument_count,
            factor_count,
            factor_names,
            factor_index,
            values,
            valid_mask,
            quality_flags,
        }
    }

    pub fn ensure_shared_schema(
        &mut self,
        instrument_count: usize,
        factor_names: Arc<[String]>,
        factor_index: Arc<HashMap<String, usize>>,
    ) {
        let factor_count = factor_names.len();
        resize_frame_storage(
            &mut self.values,
            &mut self.valid_mask,
            &mut self.quality_flags,
            instrument_count,
            factor_count,
        );
        self.instrument_count = instrument_count;
        self.factor_count = factor_count;
        self.factor_names = factor_names;
        self.factor_index = factor_index;
    }

    pub fn overwrite_from_slices(
        &mut self,
        ts_ns: i64,
        values: &[f64],
        valid_mask: &[bool],
        quality_flags: &[u32],
    ) {
        debug_assert_eq!(values.len(), self.values.len());
        debug_assert_eq!(valid_mask.len(), self.valid_mask.len());
        debug_assert_eq!(quality_flags.len(), self.quality_flags.len());
        self.ts_ns = ts_ns;
        self.values.copy_from_slice(values);
        self.valid_mask.copy_from_slice(valid_mask);
        self.quality_flags.copy_from_slice(quality_flags);
    }

    pub fn as_row_major(&self) -> (&[f64], usize, usize) {
        (&self.values, self.instrument_count, self.factor_count)
    }

    #[inline]
    pub fn factor_idx(&self, factor_name: &str) -> Option<usize> {
        self.factor_index.get(factor_name).copied()
    }

    #[inline]
    pub fn value_at(&self, instrument_idx: usize, factor_idx: usize) -> Option<f64> {
        let idx = self.cell_idx(instrument_idx, factor_idx)?;
        Some(self.values[idx])
    }

    pub fn is_valid_at(&self, instrument_idx: usize, factor_idx: usize) -> bool {
        self.cell_idx(instrument_idx, factor_idx)
            .map(|idx| self.valid_mask[idx])
            .unwrap_or(false)
    }

    pub fn row_values(&self, instrument_idx: usize) -> Option<&[f64]> {
        if instrument_idx >= self.instrument_count {
            return None;
        }
        let start = instrument_idx * self.factor_count;
        let end = start + self.factor_count;
        Some(&self.values[start..end])
    }

    pub fn factor_value(&self, instrument_idx: usize, factor_name: &str) -> Option<f64> {
        let factor_idx = self.factor_idx(factor_name)?;
        self.value_at(instrument_idx, factor_idx)
    }

    pub fn factor_values(&self, factor_name: &str) -> Option<Vec<f64>> {
        let factor_idx = self.factor_idx(factor_name)?;
        let mut out = Vec::with_capacity(self.instrument_count);
        for idx in (factor_idx..self.values.len()).step_by(self.factor_count) {
            out.push(self.values[idx]);
        }
        Some(out)
    }

    #[inline]
    fn cell_idx(&self, instrument_idx: usize, factor_idx: usize) -> Option<usize> {
        checked_cell_idx(
            instrument_idx,
            factor_idx,
            self.instrument_count,
            self.factor_count,
        )
    }
}

#[inline]
fn expected_cells(instrument_count: usize, factor_count: usize) -> usize {
    instrument_count * factor_count
}

fn assert_frame_shape(
    instrument_count: usize,
    factor_count: usize,
    values_len: usize,
    valid_mask_len: usize,
    quality_flags_len: usize,
) {
    let cells = expected_cells(instrument_count, factor_count);
    assert_eq!(
        values_len, cells,
        "values length must equal instrument_count * factor_count",
    );
    assert_eq!(
        valid_mask_len, cells,
        "valid_mask length must equal instrument_count * factor_count",
    );
    assert_eq!(
        quality_flags_len, instrument_count,
        "quality_flags length must equal instrument_count",
    );
}

fn resize_frame_storage(
    values: &mut Vec<f64>,
    valid_mask: &mut Vec<bool>,
    quality_flags: &mut Vec<u32>,
    instrument_count: usize,
    factor_count: usize,
) {
    let cells = expected_cells(instrument_count, factor_count);
    if values.len() != cells {
        values.resize(cells, f64::NAN);
    }
    if valid_mask.len() != cells {
        valid_mask.resize(cells, false);
    }
    if quality_flags.len() != instrument_count {
        quality_flags.resize(instrument_count, 0);
    }
}

#[inline]
fn checked_cell_idx(
    instrument_idx: usize,
    factor_idx: usize,
    instrument_count: usize,
    factor_count: usize,
) -> Option<usize> {
    if instrument_idx >= instrument_count || factor_idx >= factor_count {
        return None;
    }
    Some(instrument_idx * factor_count + factor_idx)
}

/// Set when advance is forced under incomplete readiness.
pub const QUALITY_FORCED_ADVANCE: u32 = 1 << 0;
/// Set when upstream marks this event as a revision of same event-time data.
pub const QUALITY_REVISION: u32 = 1 << 1;
