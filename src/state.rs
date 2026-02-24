use crate::plan::{FieldBinding, PhysicalPlan};

#[derive(Debug, Clone)]
pub struct RingBuffer {
    data: Vec<f64>,
    cap: usize,
    len: usize,
    write: usize,
}

impl RingBuffer {
    pub fn new(cap: usize) -> Self {
        Self {
            data: vec![f64::NAN; cap.max(1)],
            cap: cap.max(1),
            len: 0,
            write: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, value: f64) {
        self.data[self.write] = value;
        self.write += 1;
        if self.write == self.cap {
            self.write = 0;
        }
        if self.len < self.cap {
            self.len += 1;
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn get_lag(&self, lag: usize) -> Option<f64> {
        if lag >= self.len {
            return None;
        }
        let last = if self.write == 0 {
            self.cap - 1
        } else {
            self.write - 1
        };
        let idx = if last >= lag {
            last - lag
        } else {
            self.cap + last - lag
        };
        Some(self.data[idx])
    }

    #[inline]
    pub fn overwrite_latest(&mut self, value: f64) {
        if self.len == 0 {
            self.push(value);
            return;
        }
        let idx = if self.write == 0 {
            self.cap - 1
        } else {
            self.write - 1
        };
        self.data[idx] = value;
    }
}

#[derive(Debug, Clone)]
pub struct FieldState {
    pub latest: f64,
    pub has_latest: bool,
    pub latest_ts_ns: i64,
    pub generation: u64,
    pub ring: RingBuffer,
}

impl FieldState {
    pub fn from_binding(binding: &FieldBinding) -> Self {
        Self {
            latest: f64::NAN,
            has_latest: false,
            latest_ts_ns: i64::MIN,
            generation: 0,
            ring: RingBuffer::new(binding.history_len),
        }
    }

    pub fn update(&mut self, value: f64) {
        self.latest = value;
        self.has_latest = true;
        self.generation = self.generation.wrapping_add(1);
        self.ring.push(value);
    }

    pub fn update_at(&mut self, value: f64, ts_ns: i64) -> bool {
        if self.has_latest && ts_ns < self.latest_ts_ns {
            return false;
        }
        if self.has_latest && ts_ns == self.latest_ts_ns && self.latest.to_bits() == value.to_bits()
        {
            return false;
        }
        self.latest = value;
        self.has_latest = true;
        self.generation = self.generation.wrapping_add(1);
        if self.latest_ts_ns == ts_ns {
            self.ring.overwrite_latest(value);
            return true;
        }
        self.latest_ts_ns = ts_ns;
        self.ring.push(value);
        true
    }
}

#[derive(Debug, Clone)]
pub struct FieldStore {
    // Flat layout: [instrument][field] for fewer allocations and better locality.
    fields: Vec<FieldState>,
    instrument_count: usize,
    field_count: usize,
}

impl FieldStore {
    pub fn from_plan(plan: &PhysicalPlan) -> Self {
        let instrument_count = plan.universe_slots.len();
        let field_count = plan.fields.len();
        let mut fields = Vec::with_capacity(instrument_count * field_count);
        for _ in 0..instrument_count {
            fields.extend(plan.fields.iter().map(FieldState::from_binding));
        }
        Self {
            fields,
            instrument_count,
            field_count,
        }
    }

    #[inline]
    fn cell_idx(&self, instrument_idx: usize, field_slot: usize) -> usize {
        debug_assert!(
            instrument_idx < self.instrument_count,
            "instrument_idx out of bounds: {instrument_idx} >= {}",
            self.instrument_count
        );
        debug_assert!(
            field_slot < self.field_count,
            "field_slot out of bounds: {field_slot} >= {}",
            self.field_count
        );
        instrument_idx * self.field_count + field_slot
    }

    pub fn update(&mut self, instrument_idx: usize, field_slot: usize, value: f64) {
        let idx = self.cell_idx(instrument_idx, field_slot);
        self.fields[idx].update(value);
    }

    pub fn update_at(
        &mut self,
        instrument_idx: usize,
        field_slot: usize,
        value: f64,
        ts_ns: i64,
    ) -> bool {
        let idx = self.cell_idx(instrument_idx, field_slot);
        self.fields[idx].update_at(value, ts_ns)
    }

    #[inline]
    pub fn get(&self, instrument_idx: usize, field_slot: usize) -> &FieldState {
        let idx = self.cell_idx(instrument_idx, field_slot);
        &self.fields[idx]
    }
}

#[derive(Debug, Clone)]
pub struct GraphReadyGate {
    required_field_mask: Vec<bool>,
    field_count: usize,
    required_indices: Vec<usize>,
    latest_ts_ns: Vec<i64>,
    // Exact global min latest-ts over all required cells.
    min_latest_ts_ns: i64,
    // Number of required cells currently equal to min_latest_ts_ns.
    min_latest_ts_count: usize,
}

impl GraphReadyGate {
    pub fn from_plan(plan: &PhysicalPlan) -> Self {
        let field_count = plan.fields.len();
        let instrument_count = plan.universe_slots.len();
        let mut required_field_mask = vec![false; field_count];
        for &field_slot in &plan.ready_required_fields {
            required_field_mask[field_slot] = true;
        }
        let required_field_slots: Vec<usize> = required_field_mask
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(field_slot, required)| required.then_some(field_slot))
            .collect();
        let mut required_indices =
            Vec::with_capacity(required_field_slots.len() * instrument_count);
        for instrument_idx in 0..instrument_count {
            let row_base = instrument_idx * field_count;
            for &field_slot in &required_field_slots {
                required_indices.push(row_base + field_slot);
            }
        }
        let (min_latest_ts_ns, min_latest_ts_count) = if required_indices.is_empty() {
            // No ready constraints -> always ready.
            (i64::MAX, 0)
        } else {
            (i64::MIN, required_indices.len())
        };
        Self {
            required_field_mask,
            field_count,
            required_indices,
            latest_ts_ns: vec![i64::MIN; field_count * instrument_count],
            min_latest_ts_ns,
            min_latest_ts_count,
        }
    }

    #[inline]
    pub fn mark(&mut self, ts_ns: i64, instrument_idx: usize, field_slot: usize) {
        debug_assert!(
            field_slot < self.field_count,
            "field_slot out of bounds: {field_slot} >= {}",
            self.field_count
        );
        debug_assert!(
            instrument_idx < self.latest_ts_ns.len() / self.field_count,
            "instrument_idx out of bounds: {instrument_idx} >= {}",
            self.latest_ts_ns.len() / self.field_count
        );
        if !self.required_field_mask[field_slot] {
            return;
        }
        let idx = instrument_idx * self.field_count + field_slot;
        debug_assert!(
            idx < self.latest_ts_ns.len(),
            "required index out of bounds: {idx} >= {}",
            self.latest_ts_ns.len()
        );
        let prev_ts_ns = self.latest_ts_ns[idx];
        if ts_ns <= prev_ts_ns {
            return;
        }
        self.latest_ts_ns[idx] = ts_ns;

        if self.required_indices.is_empty() {
            return;
        }
        if prev_ts_ns == self.min_latest_ts_ns {
            debug_assert!(
                self.min_latest_ts_count > 0,
                "min_latest_ts_count must be > 0 before decrement"
            );
            self.min_latest_ts_count -= 1;
            if self.min_latest_ts_count == 0 {
                self.recompute_min_latest_ts();
            }
        }
    }

    pub fn is_ready(&self, ts_ns: i64) -> bool {
        self.min_latest_ts_ns >= ts_ns
    }

    fn recompute_min_latest_ts(&mut self) {
        if self.required_indices.is_empty() {
            self.min_latest_ts_ns = i64::MAX;
            self.min_latest_ts_count = 0;
            return;
        }

        let mut min_ts = i64::MAX;
        let mut min_count = 0usize;
        for &idx in &self.required_indices {
            let ts = self.latest_ts_ns[idx];
            if ts < min_ts {
                min_ts = ts;
                min_count = 1;
            } else if ts == min_ts {
                min_count += 1;
            }
        }
        self.min_latest_ts_ns = min_ts;
        self.min_latest_ts_count = min_count;
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RuntimeScratchConfig {
    pub enable_rank_pairs: bool,
    pub enable_tmp_f64: bool,
    pub enable_tmp_usize: bool,
}

#[derive(Debug, Clone, Default)]
pub struct RuntimeScratch {
    rank_pairs: Option<Vec<(usize, f64)>>,
    tmp_f64: Option<Vec<f64>>,
    tmp_usize: Option<Vec<usize>>,
}

impl RuntimeScratch {
    pub fn from_config(config: RuntimeScratchConfig, instrument_count: usize) -> Self {
        let mut scratch = Self::default();
        if config.enable_rank_pairs {
            scratch.rank_pairs = Some(Vec::with_capacity(instrument_count));
        }
        if config.enable_tmp_f64 {
            scratch.tmp_f64 = Some(Vec::with_capacity(instrument_count));
        }
        if config.enable_tmp_usize {
            scratch.tmp_usize = Some(Vec::with_capacity(instrument_count));
        }
        scratch
    }

    pub fn take_rank_pairs(&mut self, min_capacity: usize) -> Vec<(usize, f64)> {
        take_buffer(&mut self.rank_pairs, min_capacity)
    }

    pub fn put_rank_pairs(&mut self, mut buf: Vec<(usize, f64)>) {
        put_buffer(&mut self.rank_pairs, &mut buf);
    }

    pub fn take_tmp_f64(&mut self, min_capacity: usize) -> Vec<f64> {
        take_buffer(&mut self.tmp_f64, min_capacity)
    }

    pub fn put_tmp_f64(&mut self, mut buf: Vec<f64>) {
        put_buffer(&mut self.tmp_f64, &mut buf);
    }

    pub fn take_tmp_usize(&mut self, min_capacity: usize) -> Vec<usize> {
        take_buffer(&mut self.tmp_usize, min_capacity)
    }

    pub fn put_tmp_usize(&mut self, mut buf: Vec<usize>) {
        put_buffer(&mut self.tmp_usize, &mut buf);
    }

    #[cfg(test)]
    pub fn rank_pairs_capacity(&self) -> Option<usize> {
        self.rank_pairs.as_ref().map(Vec::capacity)
    }

    #[cfg(test)]
    pub fn tmp_f64_capacity(&self) -> Option<usize> {
        self.tmp_f64.as_ref().map(Vec::capacity)
    }

    #[cfg(test)]
    pub fn tmp_usize_capacity(&self) -> Option<usize> {
        self.tmp_usize.as_ref().map(Vec::capacity)
    }
}

#[inline]
fn take_buffer<T>(slot: &mut Option<Vec<T>>, min_capacity: usize) -> Vec<T> {
    let mut buf = slot
        .take()
        .unwrap_or_else(|| Vec::with_capacity(min_capacity.max(1)));
    if buf.capacity() < min_capacity {
        buf.reserve(min_capacity - buf.capacity());
    }
    buf.clear();
    buf
}

#[inline]
fn put_buffer<T>(slot: &mut Option<Vec<T>>, buf: &mut Vec<T>) {
    buf.clear();
    *slot = Some(std::mem::take(buf));
}

#[derive(Debug, Clone)]
pub struct EngineState {
    pub instrument_count: usize,
    pub node_count: usize,
    pub field_store: FieldStore,
    pub node_outputs: Vec<f64>,
    pub node_valid: Vec<bool>,
    pub quality_flags: Vec<u32>,
    pub scratch: RuntimeScratch,
    pub ready_gate: GraphReadyGate,
}

impl EngineState {
    pub fn from_plan(plan: &PhysicalPlan, scratch_config: RuntimeScratchConfig) -> Self {
        let instrument_count = plan.universe_slots.len();
        let node_count = plan.nodes.len();
        Self {
            instrument_count,
            node_count,
            field_store: FieldStore::from_plan(plan),
            node_outputs: vec![f64::NAN; instrument_count * node_count],
            node_valid: vec![false; instrument_count * node_count],
            quality_flags: vec![0; instrument_count],
            scratch: RuntimeScratch::from_config(scratch_config, instrument_count),
            ready_gate: GraphReadyGate::from_plan(plan),
        }
    }

    #[inline]
    pub fn cell_idx(&self, instrument_idx: usize, output_slot: usize) -> usize {
        debug_assert!(
            instrument_idx < self.instrument_count,
            "instrument_idx out of bounds: {instrument_idx} >= {}",
            self.instrument_count
        );
        debug_assert!(
            output_slot < self.node_count,
            "output_slot out of bounds: {output_slot} >= {}",
            self.node_count
        );
        instrument_idx * self.node_count + output_slot
    }

    pub fn set_node_output(
        &mut self,
        instrument_idx: usize,
        output_slot: usize,
        value: f64,
        valid: bool,
    ) {
        let idx = self.cell_idx(instrument_idx, output_slot);
        self.node_outputs[idx] = value;
        self.node_valid[idx] = valid;
    }
}

#[cfg(test)]
mod tests {
    use super::{RingBuffer, RuntimeScratch, RuntimeScratchConfig};

    #[test]
    fn ring_buffer_get_lag_wraps_without_signed_math() {
        let mut ring = RingBuffer::new(3);
        ring.push(10.0);
        ring.push(20.0);
        ring.push(30.0);
        assert_eq!(ring.get_lag(0), Some(30.0));
        assert_eq!(ring.get_lag(1), Some(20.0));
        assert_eq!(ring.get_lag(2), Some(10.0));

        ring.push(40.0);
        assert_eq!(ring.get_lag(0), Some(40.0));
        assert_eq!(ring.get_lag(1), Some(30.0));
        assert_eq!(ring.get_lag(2), Some(20.0));
        assert_eq!(ring.get_lag(3), None);
    }

    #[test]
    fn runtime_scratch_preallocates_enabled_profiles() {
        let config = RuntimeScratchConfig {
            enable_rank_pairs: true,
            enable_tmp_f64: true,
            enable_tmp_usize: true,
        };
        let scratch = RuntimeScratch::from_config(config, 8);
        assert_eq!(scratch.rank_pairs_capacity(), Some(8));
        assert_eq!(scratch.tmp_f64_capacity(), Some(8));
        assert_eq!(scratch.tmp_usize_capacity(), Some(8));
    }

    #[test]
    fn runtime_scratch_can_lazy_allocate_disabled_profile() {
        let mut scratch = RuntimeScratch::from_config(RuntimeScratchConfig::default(), 0);
        assert_eq!(scratch.tmp_f64_capacity(), None);

        let mut tmp = scratch.take_tmp_f64(16);
        tmp.push(1.0);
        scratch.put_tmp_f64(tmp);

        assert!(
            scratch
                .tmp_f64_capacity()
                .expect("tmp_f64 should be initialized")
                >= 16
        );
    }
}
