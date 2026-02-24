use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompileError {
    #[error("empty factor request")]
    EmptyRequest,
    #[error("invalid expression `{expr}`: {reason}")]
    InvalidExpression { expr: String, reason: String },
    #[error("unknown operator `{name}`")]
    UnknownOperator { name: String },
    #[error("operator `{name}` requires {expected} args, got {actual}")]
    InvalidArity {
        name: String,
        expected: usize,
        actual: usize,
    },
}

#[derive(Debug, Error)]
pub enum BindError {
    #[error("required field `{field}` not found in data catalog")]
    MissingField { field: String },
    #[error("universe is empty")]
    EmptyUniverse,
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("engine is not loaded")]
    NotLoaded,
    #[error("event payload does not match source kind")]
    PayloadMismatch,
    #[error("invalid instrument slot {slot}")]
    InvalidInstrumentSlot { slot: u32 },
    #[error("invalid source slot {slot}")]
    InvalidSourceSlot { slot: u16 },
    #[error("payload kind does not match source slot {slot}")]
    SourcePayloadMismatch { slot: u16 },
    #[error("field `{field}` is not supported for source kind `{source_kind}`")]
    UnsupportedFieldAccessor {
        source_kind: &'static str,
        field: String,
    },
    #[error("invalid kernel parameter `{param}` for op `{op}`: {value}")]
    InvalidKernelParam {
        op: &'static str,
        param: &'static str,
        value: usize,
    },
    #[error("kernel parameter kind mismatch for op `{op}`: expected {expected}")]
    KernelParamMismatch {
        op: &'static str,
        expected: &'static str,
    },
    #[error("unsupported kernel for node_id={node_id}, op={op}, phase={phase}")]
    UnsupportedKernel {
        node_id: usize,
        op: String,
        phase: &'static str,
    },
    #[error("advance ts must be monotonic: current={current_ts_ns}, last={last_ts_ns}")]
    NonMonotonicAdvance { current_ts_ns: i64, last_ts_ns: i64 },
    #[error("advance not allowed at ts={ts_ns} under strict policy")]
    NotReady { ts_ns: i64 },
}
