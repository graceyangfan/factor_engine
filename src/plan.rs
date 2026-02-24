use crate::ops::OpCode;
use crate::types::{AdvancePolicy, SourceKind};
use std::collections::BTreeMap;

pub const MAX_NODE_INPUTS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecMode {
    /// Single-source node updated directly on event ingestion.
    EventSingle,
    /// Single-source lineage, but scheduled on barrier clock.
    ///
    /// Typical case: non-CS operator whose inputs are bound across source streams in
    /// current simplified planner, or future lineage carrying barrier ancestors.
    BarrierSingle,
    /// Native barrier batch operator (e.g. CS operators).
    BarrierMulti,
}

impl ExecMode {
    #[inline]
    pub const fn is_barrier(self) -> bool {
        matches!(self, Self::BarrierSingle | Self::BarrierMulti)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeLineage {
    /// Bitset of bound source streams reachable from this node.
    pub source_mask: u64,
    /// Monotonic flag: once any ancestor/subtree is multi-source, it stays true upward.
    pub has_multi_ancestor: bool,
    /// Monotonic flag: once barrier semantics are required, they stay true upward.
    pub barrier_tainted: bool,
}

impl NodeLineage {
    #[inline]
    pub const fn source_cardinality(self) -> u8 {
        self.source_mask.count_ones() as u8
    }

    #[inline]
    pub fn from_source_slot(source_slot: u16) -> Self {
        let mask = 1_u64 << (source_slot as u64);
        Self {
            source_mask: mask,
            has_multi_ancestor: false,
            barrier_tainted: false,
        }
    }

    #[inline]
    pub fn merge(children: &[Self], op_barrier_semantic: bool) -> Self {
        let mut source_mask = 0_u64;
        let mut inherited_multi = false;
        let mut inherited_barrier = false;
        for child in children {
            source_mask |= child.source_mask;
            inherited_multi |= child.has_multi_ancestor;
            inherited_barrier |= child.barrier_tainted;
        }
        let multi_here = source_mask.count_ones() > 1;
        let has_multi_ancestor = inherited_multi || multi_here;
        let barrier_tainted = inherited_barrier || op_barrier_semantic || has_multi_ancestor;
        Self {
            source_mask,
            has_multi_ancestor,
            barrier_tainted,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FieldKey {
    pub source_kind: SourceKind,
    pub field: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicalParam {
    None,
    Window(usize),
    Lag(usize),
    WindowQuantile {
        window: usize,
        q_bits: u64,
    },
    CsNeutralize {
        regressor_count: u8,
        has_group: bool,
        has_weights: bool,
        standardize: bool,
    },
}

impl LogicalParam {
    #[inline]
    pub const fn window(self) -> Option<usize> {
        match self {
            Self::Window(window) | Self::WindowQuantile { window, .. } => Some(window),
            _ => None,
        }
    }

    #[inline]
    pub const fn lag(self) -> Option<usize> {
        match self {
            Self::Lag(lag) => Some(lag),
            _ => None,
        }
    }

    #[inline]
    pub fn quantile(self) -> Option<f64> {
        match self {
            Self::WindowQuantile { q_bits, .. } => Some(f64::from_bits(q_bits)),
            _ => None,
        }
    }

    #[inline]
    pub const fn cs_neutralize(self) -> Option<(u8, bool, bool, bool)> {
        match self {
            Self::CsNeutralize {
                regressor_count,
                has_group,
                has_weights,
                standardize,
            } => Some((regressor_count, has_group, has_weights, standardize)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalNode {
    pub node_id: usize,
    pub op: OpCode,
    pub lineage: NodeLineage,
    pub input_fields: Vec<FieldKey>,
    /// Optional derived field key emitted when this node is referenced by parent expressions.
    pub output_field: Option<FieldKey>,
    /// Structured operator parameter for planner/bind stages.
    pub param: LogicalParam,
    pub output_name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalPlan {
    pub nodes: Vec<LogicalNode>,
    /// Output slot for each requested expression index.
    pub outputs: Vec<usize>,
    /// Extra output names pointing to existing output slots after CSE de-dup.
    pub output_aliases: Vec<(String, usize)>,
    pub required_fields: BTreeMap<FieldKey, usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldBinding {
    pub field_slot: usize,
    /// Which source stream this field belongs to.
    pub source_slot: u16,
    pub key: FieldKey,
    pub history_len: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalNode {
    pub node_id: usize,
    pub op: OpCode,
    pub lineage: NodeLineage,
    pub input_field_slots: [usize; MAX_NODE_INPUTS],
    pub input_count: u8,
    pub output_field_slot: Option<usize>,
    /// Final execution mode decided at bind time after source-slot resolution.
    pub exec_mode: ExecMode,
    pub param: LogicalParam,
    pub output_slot: usize,
}

impl PhysicalNode {
    #[inline]
    pub const fn input_slot(&self, idx: usize) -> usize {
        self.input_field_slots[idx]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalPlan {
    /// Bound source table: `source_slot -> SourceKind`.
    pub source_kinds: Vec<SourceKind>,
    pub fields: Vec<FieldBinding>,
    pub nodes: Vec<PhysicalNode>,
    pub single_nodes: Vec<usize>,
    pub multi_nodes: Vec<usize>,
    pub output_names: Vec<String>,
    /// Alias output names -> output slot index.
    pub output_aliases: Vec<(String, usize)>,
    pub policy: AdvancePolicy,
    pub universe_slots: Vec<u32>,
    pub ready_required_fields: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompileManifest {
    pub node_count: usize,
    pub field_count: usize,
    /// Number of requested top-level expressions.
    pub expr_count: usize,
    /// Number of successful operator-lowering attempts.
    pub lowered_op_count: usize,
    /// Common-subexpression hits from node signature lookup.
    pub cse_hit_count: usize,
    /// Algebraic identity folds reusing an existing node (e.g. `x + 0`).
    pub identity_fold_count: usize,
    /// Number of output aliases produced by expression de-dup.
    pub alias_count: usize,
    /// End-to-end compile latency in microseconds.
    pub compile_time_us: u64,
}

impl CompileManifest {
    #[inline]
    pub fn summary_line(&self) -> String {
        format!(
            "exprs={} nodes={} fields={} lowered={} cse_hits={} identity_folds={} aliases={} compile_us={}",
            self.expr_count,
            self.node_count,
            self.field_count,
            self.lowered_op_count,
            self.cse_hit_count,
            self.identity_fold_count,
            self.alias_count,
            self.compile_time_us
        )
    }
}

#[cfg(test)]
mod tests {
    use super::NodeLineage;

    #[test]
    fn lineage_merge_is_monotonic_for_multi_source_ancestor() {
        let child_a = NodeLineage::from_source_slot(0);
        let child_b = NodeLineage::from_source_slot(1);
        let mixed = NodeLineage::merge(&[child_a, child_b], false);
        assert_eq!(mixed.source_cardinality(), 2);
        assert!(mixed.has_multi_ancestor);
        assert!(mixed.barrier_tainted);

        let parent = NodeLineage::merge(&[mixed], false);
        assert!(parent.has_multi_ancestor);
        assert!(parent.barrier_tainted);
    }
}
