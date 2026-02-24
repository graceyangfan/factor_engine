use crate::error::EngineError;
use crate::ops::{self, CompileArgSpec, Domain, ExecCapability, MultiKernel, OpCode, SingleKernel};
use crate::plan::LogicalParam;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelParamSpec {
    /// Operator does not consume runtime numeric params.
    None,
    /// Operator expects one `window` param.
    Window,
    /// Operator expects one `lag` param.
    Lag,
    /// Operator expects `(window, q)` for time-series quantile.
    WindowQuantile,
    /// Operator expects CS neutralization config in `LogicalParam::CsNeutralize`.
    CsNeutralize,
}

impl KernelParamSpec {
    #[inline]
    pub fn build(
        self,
        op_name: &'static str,
        param: LogicalParam,
    ) -> Result<LogicalParam, EngineError> {
        match self {
            Self::None => Ok(LogicalParam::None),
            Self::Window => {
                let raw = param.window().ok_or(EngineError::KernelParamMismatch {
                    op: op_name,
                    expected: "window",
                })?;
                if raw == 0 {
                    Err(EngineError::InvalidKernelParam {
                        op: op_name,
                        param: "window",
                        value: raw,
                    })
                } else {
                    Ok(LogicalParam::Window(raw))
                }
            }
            Self::Lag => {
                let raw = param.lag().ok_or(EngineError::KernelParamMismatch {
                    op: op_name,
                    expected: "lag",
                })?;
                if raw == 0 {
                    Err(EngineError::InvalidKernelParam {
                        op: op_name,
                        param: "lag",
                        value: raw,
                    })
                } else {
                    Ok(LogicalParam::Lag(raw))
                }
            }
            Self::WindowQuantile => {
                let window = param.window().ok_or(EngineError::KernelParamMismatch {
                    op: op_name,
                    expected: "window_quantile",
                })?;
                if window == 0 {
                    return Err(EngineError::InvalidKernelParam {
                        op: op_name,
                        param: "window",
                        value: window,
                    });
                }
                let q = param.quantile().ok_or(EngineError::KernelParamMismatch {
                    op: op_name,
                    expected: "window_quantile",
                })?;
                if !q.is_finite() || !(0.0..=1.0).contains(&q) {
                    return Err(EngineError::KernelParamMismatch {
                        op: op_name,
                        expected: "q in [0,1]",
                    });
                }
                Ok(param)
            }
            Self::CsNeutralize => {
                if param.cs_neutralize().is_none() {
                    Err(EngineError::KernelParamMismatch {
                        op: op_name,
                        expected: "cs_neutralize",
                    })
                } else {
                    Ok(param)
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HistorySpec {
    /// Keep 1 latest sample only.
    One,
    /// Keep `param` samples.
    Param,
    /// Keep `param + 1` samples (e.g. delta with lag).
    ParamPlusOne,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScratchProfile {
    /// Shared `(instrument_idx, value)` buffer for CS rank.
    RankPairs,
    /// Shared temporary `Vec<f64>` buffer.
    TmpF64,
    /// Shared temporary `Vec<usize>` buffer.
    TmpUsize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TsUnivariateMomentsView {
    Sum,
    Mean,
    Std,
    Var,
    Zscore,
    Skew,
    Kurt,
}

impl TsUnivariateMomentsView {
    #[inline]
    pub const fn required_order(self) -> u8 {
        match self {
            Self::Sum => 1,
            Self::Mean => 1,
            Self::Std => 2,
            Self::Var => 2,
            Self::Zscore => 2,
            Self::Skew => 3,
            Self::Kurt => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TsBivariateMomentsView {
    Cov,
    Beta,
    Corr,
    RegressionSlope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedFamily {
    /// Shared univariate rolling moments profile: sum/mean/std/var/zscore/skew/kurt.
    TsUnivariateMoments { view: TsUnivariateMomentsView },
    /// Shared bivariate rolling moments profile: cov/beta/corr/linear-regression slope.
    TsBivariateMoments { view: TsBivariateMomentsView },
}

#[derive(Debug, Clone, Copy)]
pub struct OpMeta {
    /// DSL operator name, used at compile parse/validation time.
    pub name: &'static str,
    /// Internal opcode persisted into logical/physical plans.
    pub op: OpCode,
    /// Domain tag for planner/scheduler semantics.
    pub domain: Domain,
    /// Execution capability label (O(1), barrier, etc.).
    pub exec: ExecCapability,
    /// Compile-time arg shape parser converting raw args into structured fields/params.
    pub arg_spec: CompileArgSpec,
    /// History policy used to size ring buffers at bind time.
    pub history_spec: HistorySpec,
    /// Per-instrument kernel entry for single-source execution.
    pub single_kernel: Option<SingleKernel>,
    /// Barrier/multi-source kernel entry for batch execution.
    pub multi_kernel: Option<MultiKernel>,
    /// Runtime param type for single kernel.
    pub single_param_spec: KernelParamSpec,
    /// Runtime param type for multi kernel.
    pub multi_param_spec: KernelParamSpec,
    /// Whether the first two inputs are mathematically commutative and can be canonicalized.
    pub commutative_first_two: bool,
    /// Whether numeric scalar literals are allowed in field argument positions.
    pub allow_scalar_literals: bool,
    /// Scratch buffers required by this operator.
    pub scratch_profiles: &'static [ScratchProfile],
    /// Optional shared-profile family; when present runtime routes node into shared state pool.
    pub shared_family: Option<SharedFamily>,
}

pub struct OperatorRegistry;

impl OperatorRegistry {
    pub fn get(name: &str) -> Option<&'static OpMeta> {
        let index = REGISTRY_INDEX.get_or_init(build_registry_index);
        index.by_name.get(name).map(|idx| &OP_METAS[*idx])
    }

    pub fn get_by_op(op: OpCode) -> Option<&'static OpMeta> {
        let index = REGISTRY_INDEX.get_or_init(build_registry_index);
        let idx = index.by_op[op.as_usize()];
        if idx == MISSING_IDX {
            None
        } else {
            Some(&OP_METAS[idx])
        }
    }
}

struct RegistryIndex {
    by_name: HashMap<&'static str, usize>,
    by_op: [usize; OP_CODE_COUNT],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ComputeSignature {
    single_kernel_ptr: usize,
    multi_kernel_ptr: usize,
    arg_spec: CompileArgSpec,
    single_param_spec: KernelParamSpec,
    multi_param_spec: KernelParamSpec,
    history_spec: HistorySpec,
}

static REGISTRY_INDEX: OnceLock<RegistryIndex> = OnceLock::new();
const MISSING_IDX: usize = usize::MAX;
const OP_CODE_COUNT: usize = OpCode::COUNT;

fn build_registry_index() -> RegistryIndex {
    let mut by_name = HashMap::with_capacity(OP_METAS.len());
    let mut by_op = [MISSING_IDX; OP_CODE_COUNT];
    let mut by_compute_sig: HashMap<ComputeSignature, &'static str> =
        HashMap::with_capacity(OP_METAS.len());

    for (idx, meta) in OP_METAS.iter().enumerate() {
        validate_meta(meta);
        if by_name.insert(meta.name, idx).is_some() {
            panic!("duplicate operator name in registry: {}", meta.name);
        }
        let sig = compute_signature(meta);
        if let Some(prev_name) = by_compute_sig.insert(sig, meta.name) {
            panic!(
                "duplicate compute process detected: `{}` and `{}` share the same implementation signature",
                prev_name, meta.name
            );
        }

        let op_slot = meta.op.as_usize();
        if by_op[op_slot] != MISSING_IDX {
            panic!("duplicate opcode in registry: {:?}", meta.op);
        }
        by_op[op_slot] = idx;
    }

    for (op_slot, idx) in by_op.iter().enumerate() {
        if *idx == MISSING_IDX {
            panic!("opcode slot not registered: {}", op_slot);
        }
    }

    RegistryIndex { by_name, by_op }
}

fn validate_meta(meta: &OpMeta) {
    if meta.single_kernel.is_none() && meta.multi_kernel.is_none() {
        panic!(
            "operator `{}` must provide single_kernel or multi_kernel",
            meta.name
        );
    }
    if meta.single_kernel.is_none() && !matches!(meta.single_param_spec, KernelParamSpec::None) {
        panic!(
            "operator `{}` has no single_kernel but single_param_spec is not None",
            meta.name
        );
    }
    if meta.multi_kernel.is_none() && !matches!(meta.multi_param_spec, KernelParamSpec::None) {
        panic!(
            "operator `{}` has no multi_kernel but multi_param_spec is not None",
            meta.name
        );
    }
    if meta.shared_family.is_some() && meta.single_kernel.is_none() {
        panic!(
            "operator `{}` declares shared_family but has no single_kernel",
            meta.name
        );
    }
    validate_arg_and_param_specs(meta);
    validate_arg_and_history_specs(meta);
}

fn validate_arg_and_param_specs(meta: &OpMeta) {
    let expects_window = matches!(
        meta.arg_spec,
        CompileArgSpec::FieldWindow | CompileArgSpec::TwoFieldsWindow
    );
    let expects_window_quantile = matches!(meta.arg_spec, CompileArgSpec::FieldWindowQuantile);
    let expects_lag = matches!(meta.arg_spec, CompileArgSpec::FieldLag);
    let expects_none = matches!(
        meta.arg_spec,
        CompileArgSpec::FieldOnly
            | CompileArgSpec::TwoFields
            | CompileArgSpec::ThreeFields
            | CompileArgSpec::Fields2To4
    );

    if expects_window {
        if meta.single_kernel.is_some()
            && !matches!(meta.single_param_spec, KernelParamSpec::Window)
        {
            panic!(
                "operator `{}` arg_spec requires Window param for single kernel",
                meta.name
            );
        }
        if meta.multi_kernel.is_some() && !matches!(meta.multi_param_spec, KernelParamSpec::Window)
        {
            panic!(
                "operator `{}` arg_spec requires Window param for multi kernel",
                meta.name
            );
        }
    } else if expects_window_quantile {
        if meta.single_kernel.is_some()
            && !matches!(meta.single_param_spec, KernelParamSpec::WindowQuantile)
        {
            panic!(
                "operator `{}` arg_spec requires WindowQuantile param for single kernel",
                meta.name
            );
        }
        if meta.multi_kernel.is_some()
            && !matches!(meta.multi_param_spec, KernelParamSpec::WindowQuantile)
        {
            panic!(
                "operator `{}` arg_spec requires WindowQuantile param for multi kernel",
                meta.name
            );
        }
    } else if expects_lag {
        if meta.single_kernel.is_some() && !matches!(meta.single_param_spec, KernelParamSpec::Lag) {
            panic!(
                "operator `{}` arg_spec requires Lag param for single kernel",
                meta.name
            );
        }
        if meta.multi_kernel.is_some() && !matches!(meta.multi_param_spec, KernelParamSpec::Lag) {
            panic!(
                "operator `{}` arg_spec requires Lag param for multi kernel",
                meta.name
            );
        }
    } else if expects_none {
        if meta.single_kernel.is_some()
            && !matches!(
                meta.single_param_spec,
                KernelParamSpec::None | KernelParamSpec::CsNeutralize
            )
        {
            panic!(
                "operator `{}` arg_spec requires None param for single kernel",
                meta.name
            );
        }
        if meta.multi_kernel.is_some()
            && !matches!(
                meta.multi_param_spec,
                KernelParamSpec::None | KernelParamSpec::CsNeutralize
            )
        {
            panic!(
                "operator `{}` arg_spec requires None param for multi kernel",
                meta.name
            );
        }
    }
}

fn validate_arg_and_history_specs(meta: &OpMeta) {
    let history_matches = match meta.arg_spec {
        CompileArgSpec::FieldOnly
        | CompileArgSpec::TwoFields
        | CompileArgSpec::ThreeFields
        | CompileArgSpec::Fields2To4 => matches!(meta.history_spec, HistorySpec::One),
        CompileArgSpec::FieldLag => matches!(meta.history_spec, HistorySpec::ParamPlusOne),
        CompileArgSpec::FieldWindow
        | CompileArgSpec::TwoFieldsWindow
        | CompileArgSpec::FieldWindowQuantile => match meta.history_spec {
            HistorySpec::Param => true,
            HistorySpec::One => meta.shared_family.is_some(),
            HistorySpec::ParamPlusOne => false,
        },
    };
    if !history_matches {
        panic!(
            "operator `{}` has incompatible arg_spec {:?} and history_spec {:?}",
            meta.name, meta.arg_spec, meta.history_spec
        );
    }
}

#[inline]
fn compute_signature(meta: &OpMeta) -> ComputeSignature {
    ComputeSignature {
        single_kernel_ptr: meta.single_kernel.map(|f| f as usize).unwrap_or(0),
        multi_kernel_ptr: meta.multi_kernel.map(|f| f as usize).unwrap_or(0),
        arg_spec: meta.arg_spec,
        single_param_spec: meta.single_param_spec,
        multi_param_spec: meta.multi_param_spec,
        history_spec: meta.history_spec,
    }
}

const NO_SCRATCH: &[ScratchProfile] = &[];
const TS_RANK_SCRATCH: &[ScratchProfile] = &[ScratchProfile::TmpF64];
const CS_RANK_SCRATCH: &[ScratchProfile] = &[ScratchProfile::RankPairs];
const CS_ZSCORE_SCRATCH: &[ScratchProfile] = &[ScratchProfile::TmpF64, ScratchProfile::TmpUsize];
const CS_QUANTILE_SCRATCH: &[ScratchProfile] = &[ScratchProfile::TmpF64];

const fn ts_unary_window_single(
    name: &'static str,
    op: OpCode,
    exec: ExecCapability,
    kernel: SingleKernel,
    scratch_profiles: &'static [ScratchProfile],
    shared_family: Option<SharedFamily>,
) -> OpMeta {
    let history_spec = match shared_family {
        Some(_) => HistorySpec::One,
        None => HistorySpec::Param,
    };
    OpMeta {
        name,
        op,
        domain: Domain::Ts,
        exec,
        arg_spec: CompileArgSpec::FieldWindow,
        history_spec,
        single_kernel: Some(kernel),
        multi_kernel: None,
        single_param_spec: KernelParamSpec::Window,
        multi_param_spec: KernelParamSpec::None,
        commutative_first_two: false,
        allow_scalar_literals: false,
        scratch_profiles,
        shared_family,
    }
}

const fn ts_binary_window_single(
    name: &'static str,
    op: OpCode,
    exec: ExecCapability,
    kernel: SingleKernel,
    commutative_first_two: bool,
    scratch_profiles: &'static [ScratchProfile],
    shared_family: Option<SharedFamily>,
) -> OpMeta {
    let history_spec = match shared_family {
        Some(_) => HistorySpec::One,
        None => HistorySpec::Param,
    };
    OpMeta {
        name,
        op,
        domain: Domain::Ts,
        exec,
        arg_spec: CompileArgSpec::TwoFieldsWindow,
        history_spec,
        single_kernel: Some(kernel),
        multi_kernel: None,
        single_param_spec: KernelParamSpec::Window,
        multi_param_spec: KernelParamSpec::None,
        commutative_first_two,
        allow_scalar_literals: false,
        scratch_profiles,
        shared_family,
    }
}

const fn ts_lag_single(
    name: &'static str,
    op: OpCode,
    exec: ExecCapability,
    kernel: SingleKernel,
) -> OpMeta {
    OpMeta {
        name,
        op,
        domain: Domain::Ts,
        exec,
        arg_spec: CompileArgSpec::FieldLag,
        history_spec: HistorySpec::ParamPlusOne,
        single_kernel: Some(kernel),
        multi_kernel: None,
        single_param_spec: KernelParamSpec::Lag,
        multi_param_spec: KernelParamSpec::None,
        commutative_first_two: false,
        allow_scalar_literals: false,
        scratch_profiles: NO_SCRATCH,
        shared_family: None,
    }
}

const fn elem_binary_single(
    name: &'static str,
    op: OpCode,
    kernel: SingleKernel,
    commutative_first_two: bool,
) -> OpMeta {
    OpMeta {
        name,
        op,
        domain: Domain::Elem,
        exec: ExecCapability::ExactIncrementalO1,
        arg_spec: CompileArgSpec::TwoFields,
        history_spec: HistorySpec::One,
        single_kernel: Some(kernel),
        multi_kernel: None,
        single_param_spec: KernelParamSpec::None,
        multi_param_spec: KernelParamSpec::None,
        commutative_first_two,
        allow_scalar_literals: true,
        scratch_profiles: NO_SCRATCH,
        shared_family: None,
    }
}

const fn elem_unary_single(name: &'static str, op: OpCode, kernel: SingleKernel) -> OpMeta {
    OpMeta {
        name,
        op,
        domain: Domain::Elem,
        exec: ExecCapability::ExactIncrementalO1,
        arg_spec: CompileArgSpec::FieldOnly,
        history_spec: HistorySpec::One,
        single_kernel: Some(kernel),
        multi_kernel: None,
        single_param_spec: KernelParamSpec::None,
        multi_param_spec: KernelParamSpec::None,
        commutative_first_two: false,
        allow_scalar_literals: true,
        scratch_profiles: NO_SCRATCH,
        shared_family: None,
    }
}

const fn elem_ternary_single(name: &'static str, op: OpCode, kernel: SingleKernel) -> OpMeta {
    OpMeta {
        name,
        op,
        domain: Domain::Elem,
        exec: ExecCapability::ExactIncrementalO1,
        arg_spec: CompileArgSpec::ThreeFields,
        history_spec: HistorySpec::One,
        single_kernel: Some(kernel),
        multi_kernel: None,
        single_param_spec: KernelParamSpec::None,
        multi_param_spec: KernelParamSpec::None,
        commutative_first_two: false,
        allow_scalar_literals: true,
        scratch_profiles: NO_SCRATCH,
        shared_family: None,
    }
}

const fn cs_barrier_single_field(
    name: &'static str,
    op: OpCode,
    kernel: MultiKernel,
    scratch_profiles: &'static [ScratchProfile],
) -> OpMeta {
    OpMeta {
        name,
        op,
        domain: Domain::Cs,
        exec: ExecCapability::BarrierBatchExact,
        arg_spec: CompileArgSpec::FieldOnly,
        history_spec: HistorySpec::One,
        single_kernel: None,
        multi_kernel: Some(kernel),
        single_param_spec: KernelParamSpec::None,
        multi_param_spec: KernelParamSpec::None,
        commutative_first_two: false,
        allow_scalar_literals: false,
        scratch_profiles,
        shared_family: None,
    }
}

const OP_METAS: [OpMeta; 62] = [
    elem_unary_single("elem_abs", OpCode::ElemAbs, ops::elem_abs),
    elem_unary_single("elem_exp", OpCode::ElemExp, ops::elem_exp),
    elem_unary_single("elem_log", OpCode::ElemLog, ops::elem_log),
    elem_unary_single("elem_sign", OpCode::ElemSign, ops::elem_sign),
    elem_unary_single("elem_sqrt", OpCode::ElemSqrt, ops::elem_sqrt),
    elem_ternary_single("elem_clip", OpCode::ElemClip, ops::elem_clip),
    elem_ternary_single("elem_where", OpCode::ElemWhere, ops::elem_where),
    elem_binary_single("elem_fillna", OpCode::ElemFillNa, ops::elem_fillna, false),
    elem_binary_single("elem_add", OpCode::ElemAdd, ops::elem_add, true),
    elem_binary_single("elem_sub", OpCode::ElemSub, ops::elem_sub, false),
    elem_binary_single("elem_mul", OpCode::ElemMul, ops::elem_mul, true),
    elem_binary_single("elem_div", OpCode::ElemDiv, ops::elem_div, false),
    elem_binary_single("elem_pow", OpCode::ElemPow, ops::elem_pow, false),
    elem_binary_single("elem_min", OpCode::ElemMin, ops::elem_min, true),
    elem_binary_single("elem_max", OpCode::ElemMax, ops::elem_max, true),
    elem_binary_single(
        "elem_signed_power",
        OpCode::ElemSignedPower,
        ops::elem_signed_power,
        false,
    ),
    elem_unary_single("elem_to_int", OpCode::ElemToInt, ops::elem_to_int),
    elem_unary_single("elem_not", OpCode::ElemNot, ops::elem_not),
    elem_binary_single("elem_lt", OpCode::ElemLt, ops::elem_lt, false),
    elem_binary_single("elem_le", OpCode::ElemLe, ops::elem_le, false),
    elem_binary_single("elem_gt", OpCode::ElemGt, ops::elem_gt, false),
    elem_binary_single("elem_ge", OpCode::ElemGe, ops::elem_ge, false),
    elem_binary_single("elem_eq", OpCode::ElemEq, ops::elem_eq, true),
    elem_binary_single("elem_ne", OpCode::ElemNe, ops::elem_ne, true),
    elem_binary_single("elem_and", OpCode::ElemAnd, ops::elem_and, true),
    elem_binary_single("elem_or", OpCode::ElemOr, ops::elem_or, true),
    ts_unary_window_single(
        "ts_mean",
        OpCode::TsMean,
        ExecCapability::ExactIncrementalO1,
        ops::ts_mean,
        NO_SCRATCH,
        Some(SharedFamily::TsUnivariateMoments {
            view: TsUnivariateMomentsView::Mean,
        }),
    ),
    ts_unary_window_single(
        "ts_sum",
        OpCode::TsSum,
        ExecCapability::ExactIncrementalO1,
        ops::ts_sum,
        NO_SCRATCH,
        Some(SharedFamily::TsUnivariateMoments {
            view: TsUnivariateMomentsView::Sum,
        }),
    ),
    ts_unary_window_single(
        "ts_product",
        OpCode::TsProduct,
        ExecCapability::ExactIncrementalLogW,
        ops::ts_product,
        NO_SCRATCH,
        None,
    ),
    ts_unary_window_single(
        "ts_min",
        OpCode::TsMin,
        ExecCapability::ExactIncrementalLogW,
        ops::ts_min,
        NO_SCRATCH,
        None,
    ),
    ts_unary_window_single(
        "ts_max",
        OpCode::TsMax,
        ExecCapability::ExactIncrementalLogW,
        ops::ts_max,
        NO_SCRATCH,
        None,
    ),
    ts_unary_window_single(
        "ts_mad",
        OpCode::TsMad,
        ExecCapability::ExactIncrementalLogW,
        ops::ts_mad,
        NO_SCRATCH,
        None,
    ),
    ts_unary_window_single(
        "ts_std",
        OpCode::TsStd,
        ExecCapability::ExactIncrementalO1,
        ops::ts_std,
        NO_SCRATCH,
        Some(SharedFamily::TsUnivariateMoments {
            view: TsUnivariateMomentsView::Std,
        }),
    ),
    ts_unary_window_single(
        "ts_var",
        OpCode::TsVar,
        ExecCapability::ExactIncrementalO1,
        ops::ts_var,
        NO_SCRATCH,
        Some(SharedFamily::TsUnivariateMoments {
            view: TsUnivariateMomentsView::Var,
        }),
    ),
    ts_unary_window_single(
        "ts_skew",
        OpCode::TsSkew,
        ExecCapability::ExactIncrementalO1,
        ops::ts_skew,
        NO_SCRATCH,
        Some(SharedFamily::TsUnivariateMoments {
            view: TsUnivariateMomentsView::Skew,
        }),
    ),
    ts_unary_window_single(
        "ts_kurt",
        OpCode::TsKurt,
        ExecCapability::ExactIncrementalO1,
        ops::ts_kurt,
        NO_SCRATCH,
        Some(SharedFamily::TsUnivariateMoments {
            view: TsUnivariateMomentsView::Kurt,
        }),
    ),
    ts_lag_single(
        "ts_delta",
        OpCode::Delta,
        ExecCapability::ExactIncrementalO1,
        ops::delta,
    ),
    ts_lag_single(
        "ts_lag",
        OpCode::TsLag,
        ExecCapability::ExactIncrementalO1,
        ops::ts_lag,
    ),
    ts_unary_window_single(
        "ts_zscore",
        OpCode::TsZscore,
        ExecCapability::ExactIncrementalO1,
        ops::ts_zscore,
        NO_SCRATCH,
        Some(SharedFamily::TsUnivariateMoments {
            view: TsUnivariateMomentsView::Zscore,
        }),
    ),
    ts_binary_window_single(
        "ts_cov",
        OpCode::TsCov,
        ExecCapability::ExactIncrementalO1,
        ops::ts_cov,
        true,
        NO_SCRATCH,
        Some(SharedFamily::TsBivariateMoments {
            view: TsBivariateMomentsView::Cov,
        }),
    ),
    ts_binary_window_single(
        "ts_beta",
        OpCode::TsBeta,
        ExecCapability::ExactIncrementalO1,
        ops::ts_beta,
        false,
        NO_SCRATCH,
        Some(SharedFamily::TsBivariateMoments {
            view: TsBivariateMomentsView::Beta,
        }),
    ),
    ts_unary_window_single(
        "ts_ewm_mean",
        OpCode::TsEwmMean,
        ExecCapability::ExactIncrementalLogW,
        ops::ts_ewm_mean,
        NO_SCRATCH,
        None,
    ),
    ts_unary_window_single(
        "ts_ewm_var",
        OpCode::TsEwmVar,
        ExecCapability::ExactIncrementalLogW,
        ops::ts_ewm_var,
        NO_SCRATCH,
        None,
    ),
    ts_binary_window_single(
        "ts_ewm_cov",
        OpCode::TsEwmCov,
        ExecCapability::ExactIncrementalLogW,
        ops::ts_ewm_cov,
        true,
        NO_SCRATCH,
        None,
    ),
    ts_unary_window_single(
        "ts_decay_linear",
        OpCode::TsDecayLinear,
        ExecCapability::ExactIncrementalLogW,
        ops::ts_decay_linear,
        NO_SCRATCH,
        None,
    ),
    ts_unary_window_single(
        "ts_argmax",
        OpCode::TsArgMax,
        ExecCapability::ExactIncrementalLogW,
        ops::ts_argmax,
        NO_SCRATCH,
        None,
    ),
    ts_unary_window_single(
        "ts_argmin",
        OpCode::TsArgMin,
        ExecCapability::ExactIncrementalLogW,
        ops::ts_argmin,
        NO_SCRATCH,
        None,
    ),
    OpMeta {
        name: "ts_quantile",
        op: OpCode::TsQuantile,
        domain: Domain::Ts,
        exec: ExecCapability::ExactIncrementalLogW,
        arg_spec: CompileArgSpec::FieldWindowQuantile,
        history_spec: HistorySpec::Param,
        single_kernel: Some(ops::ts_quantile),
        multi_kernel: None,
        single_param_spec: KernelParamSpec::WindowQuantile,
        multi_param_spec: KernelParamSpec::None,
        commutative_first_two: false,
        allow_scalar_literals: false,
        scratch_profiles: TS_RANK_SCRATCH,
        shared_family: None,
    },
    ts_unary_window_single(
        "ts_rank",
        OpCode::TsRank,
        ExecCapability::ExactIncrementalLogW,
        ops::ts_rank,
        TS_RANK_SCRATCH,
        None,
    ),
    ts_binary_window_single(
        "ts_corr",
        OpCode::TsCorr,
        ExecCapability::ExactIncrementalO1,
        ops::ts_corr,
        true,
        NO_SCRATCH,
        Some(SharedFamily::TsBivariateMoments {
            view: TsBivariateMomentsView::Corr,
        }),
    ),
    ts_binary_window_single(
        "ts_linear_regression",
        OpCode::TsLinearRegression,
        ExecCapability::ExactIncrementalO1,
        ops::ts_linear_regression,
        false,
        NO_SCRATCH,
        Some(SharedFamily::TsBivariateMoments {
            view: TsBivariateMomentsView::RegressionSlope,
        }),
    ),
    OpMeta {
        name: "cs_neutralize",
        op: OpCode::CsNeutralize,
        domain: Domain::Cs,
        exec: ExecCapability::BarrierBatchExact,
        arg_spec: CompileArgSpec::FieldOnly,
        history_spec: HistorySpec::One,
        single_kernel: None,
        multi_kernel: Some(ops::cs_neutralize),
        single_param_spec: KernelParamSpec::None,
        multi_param_spec: KernelParamSpec::CsNeutralize,
        commutative_first_two: false,
        allow_scalar_literals: false,
        scratch_profiles: NO_SCRATCH,
        shared_family: None,
    },
    cs_barrier_single_field("cs_rank", OpCode::CsRank, ops::cs_rank, CS_RANK_SCRATCH),
    cs_barrier_single_field(
        "cs_zscore",
        OpCode::CsZscore,
        ops::cs_zscore,
        CS_ZSCORE_SCRATCH,
    ),
    cs_barrier_single_field("cs_center", OpCode::CsCenter, ops::cs_center, NO_SCRATCH),
    cs_barrier_single_field("cs_scale", OpCode::CsScale, ops::cs_scale, NO_SCRATCH),
    cs_barrier_single_field("cs_norm", OpCode::CsNorm, ops::cs_norm, NO_SCRATCH),
    cs_barrier_single_field("cs_fillna", OpCode::CsFillNa, ops::cs_fillna, NO_SCRATCH),
    OpMeta {
        name: "cs_winsorize",
        op: OpCode::CsWinsorize,
        domain: Domain::Cs,
        exec: ExecCapability::BarrierBatchExact,
        arg_spec: CompileArgSpec::TwoFields,
        history_spec: HistorySpec::One,
        single_kernel: None,
        multi_kernel: Some(ops::cs_winsorize),
        single_param_spec: KernelParamSpec::None,
        multi_param_spec: KernelParamSpec::None,
        commutative_first_two: false,
        allow_scalar_literals: true,
        scratch_profiles: CS_QUANTILE_SCRATCH,
        shared_family: None,
    },
    OpMeta {
        name: "cs_percentiles",
        op: OpCode::CsPercentiles,
        domain: Domain::Cs,
        exec: ExecCapability::BarrierBatchExact,
        arg_spec: CompileArgSpec::ThreeFields,
        history_spec: HistorySpec::One,
        single_kernel: None,
        multi_kernel: Some(ops::cs_percentiles),
        single_param_spec: KernelParamSpec::None,
        multi_param_spec: KernelParamSpec::None,
        commutative_first_two: false,
        allow_scalar_literals: true,
        scratch_profiles: CS_QUANTILE_SCRATCH,
        shared_family: None,
    },
    OpMeta {
        name: "cs_neutralize_ols",
        op: OpCode::CsNeutralizeOls,
        domain: Domain::Cs,
        exec: ExecCapability::BarrierBatchExact,
        arg_spec: CompileArgSpec::TwoFields,
        history_spec: HistorySpec::One,
        single_kernel: None,
        multi_kernel: Some(ops::cs_neutralize_ols),
        single_param_spec: KernelParamSpec::None,
        multi_param_spec: KernelParamSpec::CsNeutralize,
        commutative_first_two: false,
        allow_scalar_literals: false,
        scratch_profiles: NO_SCRATCH,
        shared_family: None,
    },
    OpMeta {
        name: "cs_neutralize_ols_multi",
        op: OpCode::CsNeutralizeOlsMulti,
        domain: Domain::Cs,
        exec: ExecCapability::BarrierBatchExact,
        arg_spec: CompileArgSpec::Fields2To4,
        history_spec: HistorySpec::One,
        single_kernel: None,
        multi_kernel: Some(ops::cs_neutralize_ols_multi),
        single_param_spec: KernelParamSpec::None,
        multi_param_spec: KernelParamSpec::CsNeutralize,
        commutative_first_two: false,
        allow_scalar_literals: false,
        scratch_profiles: NO_SCRATCH,
        shared_family: None,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_panics<F>(f: F)
    where
        F: FnOnce() + std::panic::UnwindSafe,
    {
        assert!(std::panic::catch_unwind(f).is_err());
    }

    fn base_single_meta() -> OpMeta {
        ts_unary_window_single(
            "test_single",
            OpCode::TsMean,
            ExecCapability::ExactIncrementalO1,
            ops::ts_mean,
            NO_SCRATCH,
            None,
        )
    }

    #[test]
    fn validate_meta_rejects_operator_without_kernels() {
        let mut meta = base_single_meta();
        meta.single_kernel = None;
        meta.single_param_spec = KernelParamSpec::None;
        assert_panics(|| validate_meta(&meta));
    }

    #[test]
    fn validate_meta_rejects_single_param_without_single_kernel() {
        let mut meta = base_single_meta();
        meta.single_kernel = None;
        assert_panics(|| validate_meta(&meta));
    }

    #[test]
    fn validate_meta_rejects_multi_param_without_multi_kernel() {
        let mut meta = base_single_meta();
        meta.multi_param_spec = KernelParamSpec::Window;
        assert_panics(|| validate_meta(&meta));
    }

    #[test]
    fn validate_meta_rejects_shared_family_without_single_kernel() {
        let mut meta = base_single_meta();
        meta.shared_family = Some(SharedFamily::TsUnivariateMoments {
            view: TsUnivariateMomentsView::Mean,
        });
        meta.single_kernel = None;
        meta.single_param_spec = KernelParamSpec::None;
        assert_panics(|| validate_meta(&meta));
    }

    #[test]
    fn validate_meta_rejects_arg_spec_and_param_spec_mismatch() {
        let mut meta = base_single_meta();
        meta.arg_spec = CompileArgSpec::FieldLag;
        assert_panics(|| validate_meta(&meta));
    }

    #[test]
    fn validate_meta_rejects_arg_spec_and_history_spec_mismatch() {
        let mut meta = base_single_meta();
        meta.arg_spec = CompileArgSpec::FieldOnly;
        assert_panics(|| validate_meta(&meta));
    }

    #[test]
    fn kernel_param_spec_builds_expected_params() {
        let p = KernelParamSpec::Window
            .build("ts_mean", LogicalParam::Window(8))
            .expect("window should be valid");
        assert!(matches!(p, LogicalParam::Window(8)));

        let p = KernelParamSpec::Lag
            .build("ts_delta", LogicalParam::Lag(3))
            .expect("lag should be valid");
        assert!(matches!(p, LogicalParam::Lag(3)));

        let p = KernelParamSpec::WindowQuantile
            .build(
                "ts_quantile",
                LogicalParam::WindowQuantile {
                    window: 8,
                    q_bits: 0.25_f64.to_bits(),
                },
            )
            .expect("window quantile should be valid");
        assert!(matches!(p, LogicalParam::WindowQuantile { window: 8, .. }));
    }

    #[test]
    fn kernel_param_spec_rejects_zero_window_or_lag() {
        assert!(KernelParamSpec::Window
            .build("ts_mean", LogicalParam::Window(0))
            .is_err());
        assert!(KernelParamSpec::Lag
            .build("ts_delta", LogicalParam::Lag(0))
            .is_err());
    }

    #[test]
    fn kernel_param_spec_rejects_param_kind_mismatch() {
        assert!(KernelParamSpec::Window
            .build("ts_mean", LogicalParam::Lag(5))
            .is_err());
        assert!(KernelParamSpec::Lag
            .build("ts_delta", LogicalParam::None)
            .is_err());
        assert!(KernelParamSpec::WindowQuantile
            .build("ts_quantile", LogicalParam::Window(5))
            .is_err());
    }

    #[test]
    fn opcode_count_matches_registry_entries() {
        assert_eq!(
            OP_METAS.len(),
            OpCode::COUNT,
            "new opcode should be wired into OP_METAS"
        );
        let mut seen = [false; OP_CODE_COUNT];
        for meta in OP_METAS {
            let slot = meta.op.as_usize();
            assert!(!seen[slot], "duplicate opcode slot {}", slot);
            seen[slot] = true;
        }
        assert!(
            seen.into_iter().all(|v| v),
            "all opcode slots must be covered"
        );
    }
}
