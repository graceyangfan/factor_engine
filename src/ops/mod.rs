//! Operator layer entry.
//!
//! Extension path (minimal touch points):
//! 1) implement kernel in `elem.rs` / `ts.rs` / `cs.rs`,
//! 2) add opcode in `spec.rs` and register meta in `catalog.rs` (`OP_METAS`),
//! 3) add compile/runtime parity tests.

use crate::plan::{LogicalParam, PhysicalPlan, MAX_NODE_INPUTS};
use crate::state::EngineState;

#[derive(Debug, Clone, Copy)]
pub struct KernelIo {
    pub inputs: [usize; MAX_NODE_INPUTS],
    pub input_count: u8,
    pub output: usize,
}

impl KernelIo {
    #[inline]
    pub const fn unary(input: usize, output: usize) -> Self {
        let mut inputs = [0; MAX_NODE_INPUTS];
        inputs[0] = input;
        Self {
            inputs,
            input_count: 1,
            output,
        }
    }

    #[inline]
    pub const fn binary(lhs: usize, rhs: usize, output: usize) -> Self {
        let mut inputs = [0; MAX_NODE_INPUTS];
        inputs[0] = lhs;
        inputs[1] = rhs;
        Self {
            inputs,
            input_count: 2,
            output,
        }
    }

    #[inline]
    pub fn from_slice(inputs: &[usize], output: usize) -> Self {
        debug_assert!(
            inputs.len() <= MAX_NODE_INPUTS,
            "KernelIo supports at most {MAX_NODE_INPUTS} inputs"
        );
        let mut io = Self {
            inputs: [0; MAX_NODE_INPUTS],
            input_count: inputs.len() as u8,
            output,
        };
        for (idx, input) in inputs.iter().enumerate() {
            io.inputs[idx] = *input;
        }
        io
    }

    #[inline]
    pub const fn input(self, idx: usize) -> usize {
        self.inputs[idx]
    }
}

pub type SingleKernel =
    fn(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam);
pub type MultiKernel =
    fn(plan: &PhysicalPlan, state: &mut EngineState, io: KernelIo, p: LogicalParam);

pub mod arg_spec;
pub mod catalog;
pub mod spec;

mod cs;
mod elem;
mod stats;
mod ts;

pub use arg_spec::{CompileArgSpec, ParsedCompileArgs};
pub use catalog::{
    HistorySpec, KernelParamSpec, OpMeta, OperatorRegistry, ScratchProfile, SharedFamily,
    TsBivariateMomentsView, TsUnivariateMomentsView,
};
pub use cs::{
    cs_center, cs_fillna, cs_neutralize, cs_neutralize_ols, cs_neutralize_ols_multi, cs_norm,
    cs_percentiles, cs_rank, cs_scale, cs_winsorize, cs_zscore,
};
pub use elem::{
    elem_abs, elem_add, elem_and, elem_clip, elem_div, elem_eq, elem_exp, elem_fillna, elem_ge,
    elem_gt, elem_le, elem_log, elem_lt, elem_max, elem_min, elem_mul, elem_ne, elem_not, elem_or,
    elem_pow, elem_sign, elem_signed_power, elem_sqrt, elem_sub, elem_to_int, elem_where,
};
pub use spec::{Domain, ExecCapability, OpCode};
pub use ts::{
    delta, ts_argmax, ts_argmin, ts_beta, ts_corr, ts_cov, ts_decay_linear, ts_ewm_cov,
    ts_ewm_mean, ts_ewm_var, ts_kurt, ts_lag, ts_linear_regression, ts_mad, ts_max, ts_mean,
    ts_min, ts_product, ts_quantile, ts_rank, ts_skew, ts_std, ts_sum, ts_var, ts_zscore,
};
