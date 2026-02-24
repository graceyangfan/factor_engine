#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    Ts,
    Cs,
    Elem,
    L2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecCapability {
    ExactIncrementalO1,
    ExactIncrementalLogW,
    BarrierBatchExact,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCode {
    ElemAbs,
    ElemExp,
    ElemLog,
    ElemSign,
    ElemSqrt,
    ElemClip,
    ElemWhere,
    ElemFillNa,
    ElemAdd,
    ElemSub,
    ElemMul,
    ElemDiv,
    ElemPow,
    ElemMin,
    ElemMax,
    ElemSignedPower,
    ElemToInt,
    ElemNot,
    ElemLt,
    ElemLe,
    ElemGt,
    ElemGe,
    ElemEq,
    ElemNe,
    ElemAnd,
    ElemOr,
    TsMean,
    TsSum,
    TsProduct,
    TsMin,
    TsMax,
    TsMad,
    TsStd,
    TsVar,
    TsSkew,
    TsKurt,
    Delta,
    TsLag,
    TsZscore,
    TsCov,
    TsBeta,
    TsEwmMean,
    TsEwmVar,
    TsEwmCov,
    TsDecayLinear,
    TsArgMax,
    TsArgMin,
    TsQuantile,
    TsRank,
    TsCorr,
    TsLinearRegression,
    CsNeutralize,
    CsRank,
    CsZscore,
    CsCenter,
    CsScale,
    CsNorm,
    CsFillNa,
    CsWinsorize,
    CsPercentiles,
    CsNeutralizeOls,
    CsNeutralizeOlsMulti,
}

impl OpCode {
    pub const COUNT: usize = Self::CsNeutralizeOlsMulti as usize + 1;

    #[inline]
    pub const fn as_usize(self) -> usize {
        self as usize
    }
}
