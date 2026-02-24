use crate::compile::Planner;
use crate::ops::OpCode;
use crate::plan::ExecMode;
use crate::runtime::Engine;
use crate::types::{
    AdvancePolicy, BarLite, CompileOptions, DataLite, EventEnvelope, FactorRequest,
    FeatureFrameBuffers, InputFieldCatalog, Payload, SourceKind, Universe, QUALITY_FORCED_ADVANCE,
    QUALITY_REVISION,
};
use crate::{EngineError, OnlineFactorEngine, SimplePlanner};
use std::collections::HashMap;
use std::time::Instant;

mod compile;
mod runtime;

fn three_factor_request() -> FactorRequest {
    FactorRequest::new(vec![
        "ts_mean(close, 20)".to_string(),
        "ts_delta(volume, 1)".to_string(),
        "cs_rank(close)".to_string(),
    ])
}

fn ts_only_request() -> FactorRequest {
    FactorRequest::new(vec![
        "ts_mean(close, 20)".to_string(),
        "ts_delta(volume, 1)".to_string(),
    ])
}

fn cs_zscore_request() -> FactorRequest {
    FactorRequest::new(vec!["cs_zscore(close)".to_string()])
}

fn ts_rank_request() -> FactorRequest {
    FactorRequest::new(vec!["ts_rank(close, 3)".to_string()])
}

fn ts_moments_request() -> FactorRequest {
    FactorRequest::new(vec![
        "ts_mean(close, 4)".to_string(),
        "ts_std(close, 4)".to_string(),
        "ts_skew(close, 4)".to_string(),
        "ts_kurt(close, 4)".to_string(),
    ])
}

fn cross_source_policy_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "ts_corr(bar.close, quote_tick.bid_price, 2)".to_string(),
            "ts_corr(bar.close, quote_tick.bid_price, 2) + 0.0".to_string(),
            "ts_corr(bar.close, quote_tick.bid_price, 2) * ts_corr(bar.close, quote_tick.bid_price, 2)"
                .to_string(),
            "ts_linear_regression(bar.close, quote_tick.bid_price, 2)".to_string(),
        ],
        outputs: vec![
            "corr".to_string(),
            "corr_pass".to_string(),
            "corr_sq".to_string(),
            "slope".to_string(),
        ],
        opts: CompileOptions::default(),
    }
}

fn alpha101_canonical_subset_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            // Alpha003: -1 * correlation(rank(open), rank(volume), 10)
            "-1 * ts_corr(cs_rank(open), cs_rank(volume), 10)".to_string(),
            // Alpha004: -1 * ts_rank(rank(low), 9)
            "-1 * ts_rank(cs_rank(low), 9)".to_string(),
            // Alpha006: -1 * correlation(open, volume, 10)
            "-1 * ts_corr(open, volume, 10)".to_string(),
            // Alpha012: sign(delta(volume,1)) * (-1 * delta(close,1))
            "elem_sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1))".to_string(),
            // Alpha020
            "-1 * cs_rank(open - ts_lag(high, 1)) * cs_rank(open - ts_lag(close, 1)) * cs_rank(open - ts_lag(low, 1))".to_string(),
            // Alpha040
            "-1 * cs_rank(ts_std(high, 10)) * ts_corr(high, volume, 10)".to_string(),
            // Alpha044
            "-1 * ts_corr(high, cs_rank(volume), 5)".to_string(),
        ],
        outputs: vec![
            "alpha101_003".to_string(),
            "alpha101_004".to_string(),
            "alpha101_006".to_string(),
            "alpha101_012".to_string(),
            "alpha101_020".to_string(),
            "alpha101_040".to_string(),
            "alpha101_044".to_string(),
        ],
        opts: CompileOptions::default(),
    }
}

fn phase_b_boolean_and_scale_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "elem_where(close < open, volume, close)".to_string(),
            "elem_to_int((close > open) | (open == close))".to_string(),
            "elem_signed_power(close - open, 2)".to_string(),
            "cs_scale(close)".to_string(),
        ],
        outputs: vec![
            "where_cmp".to_string(),
            "to_int_cmp".to_string(),
            "signed_pow".to_string(),
            "cs_scale_close".to_string(),
        ],
        opts: CompileOptions::default(),
    }
}

fn wave1_new_ops_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "ts_sum(close, 3)".to_string(),
            "ts_min(close, 3)".to_string(),
            "ts_max(close, 3)".to_string(),
            "elem_pow(close, 2)".to_string(),
            "elem_min(close, volume)".to_string(),
            "elem_max(close, volume)".to_string(),
            "ts_sum(elem_max(close, volume), 3)".to_string(),
        ],
        outputs: vec![
            "sum3".to_string(),
            "min3".to_string(),
            "max3".to_string(),
            "pow_close2".to_string(),
            "min_cv".to_string(),
            "max_cv".to_string(),
            "sum3_maxcv".to_string(),
        ],
        opts: CompileOptions::default(),
    }
}

fn wave2_ts_stats_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "ts_lag(close, 1)".to_string(),
            "ts_zscore(close, 3)".to_string(),
            "ts_cov(close, volume, 3)".to_string(),
            "ts_zscore(ts_lag(close, 1), 3)".to_string(),
        ],
        outputs: vec![
            "lag1".to_string(),
            "z3".to_string(),
            "cov3".to_string(),
            "z_lag1_3".to_string(),
        ],
        opts: CompileOptions::default(),
    }
}

fn wave3_var_beta_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "ts_var(close, 4)".to_string(),
            "ts_beta(close, volume, 4)".to_string(),
            "ts_beta(volume, close, 4)".to_string(),
            "ts_var(ts_mean(close, 3), 3)".to_string(),
        ],
        outputs: vec![
            "var4".to_string(),
            "beta_cv_4".to_string(),
            "beta_vc_4".to_string(),
            "var_mean3_3".to_string(),
        ],
        opts: CompileOptions::default(),
    }
}

fn wave4_ewm_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "ts_ewm_mean(close, 3)".to_string(),
            "ts_ewm_var(close, 3)".to_string(),
            "ts_ewm_cov(close, volume, 3)".to_string(),
            "ts_ewm_mean(ts_lag(close, 1), 3)".to_string(),
            "ts_ewm_var(ts_lag(close, 1), 3)".to_string(),
        ],
        outputs: vec![
            "ewm_mean3".to_string(),
            "ewm_var3".to_string(),
            "ewm_cov3".to_string(),
            "ewm_mean_lag1_3".to_string(),
            "ewm_var_lag1_3".to_string(),
        ],
        opts: CompileOptions::default(),
    }
}

fn wave5_quantile_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "ts_quantile(close, 4, 0.25)".to_string(),
            "ts_quantile(ts_lag(close, 1), 4, 0.5)".to_string(),
            "ts_quantile(ts_lag(close, 1), window=4, q=0.75)".to_string(),
        ],
        outputs: vec![
            "q25_4".to_string(),
            "q50_lag1_4".to_string(),
            "q75_lag1_4".to_string(),
        ],
        opts: CompileOptions::default(),
    }
}

fn wave6_argext_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "ts_argmax(close, 4)".to_string(),
            "ts_argmin(close, 4)".to_string(),
            "ts_argmax(ts_lag(close, 1), 4)".to_string(),
        ],
        outputs: vec![
            "argmax4".to_string(),
            "argmin4".to_string(),
            "argmax_lag1_4".to_string(),
        ],
        opts: CompileOptions::default(),
    }
}

fn wave7_elem_unary_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "elem_abs(ts_delta(close, 1))".to_string(),
            "elem_log(close)".to_string(),
            "elem_sign(close - ts_mean(close, 3))".to_string(),
        ],
        outputs: vec![
            "abs_delta1".to_string(),
            "log_close".to_string(),
            "sign_dev3".to_string(),
        ],
        opts: CompileOptions::default(),
    }
}

fn wave8_elem_extended_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "elem_exp(ts_lag(close, 1))".to_string(),
            "elem_sqrt(elem_abs(ts_delta(close, 1)))".to_string(),
            "elem_clip(close, 2.5, 4.5)".to_string(),
        ],
        outputs: vec![
            "exp_lag1".to_string(),
            "sqrt_abs_delta1".to_string(),
            "clip_close".to_string(),
        ],
        opts: CompileOptions::default(),
    }
}

fn wave9_elem_conditional_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "elem_where(1.0, close, volume)".to_string(),
            "elem_fillna(elem_div(close, close - close), 0.0)".to_string(),
            "elem_where(0.0, close, volume)".to_string(),
        ],
        outputs: vec![
            "where_one".to_string(),
            "fillna_div0".to_string(),
            "where_zero".to_string(),
        ],
        opts: CompileOptions::default(),
    }
}

fn wave10_decay_linear_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "ts_decay_linear(close, 4)".to_string(),
            "ts_decay_linear(ts_lag(close, 1), 4)".to_string(),
        ],
        outputs: vec!["decay4".to_string(), "decay4_lag1".to_string()],
        opts: CompileOptions::default(),
    }
}

fn wave11_product_mad_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "ts_product(close, 4)".to_string(),
            "ts_mad(close, 4)".to_string(),
            "ts_mad(ts_lag(close, 1), 4)".to_string(),
        ],
        outputs: vec![
            "product4".to_string(),
            "mad4".to_string(),
            "mad4_lag1".to_string(),
        ],
        opts: CompileOptions::default(),
    }
}

fn wave12_cs_preprocess_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "cs_center(close)".to_string(),
            "cs_norm(close)".to_string(),
            "cs_fillna(close)".to_string(),
        ],
        outputs: vec![
            "center_close".to_string(),
            "norm_close".to_string(),
            "fillna_close".to_string(),
        ],
        opts: CompileOptions::default(),
    }
}

fn wave13_cs_clip_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "cs_winsorize(close, 0.1)".to_string(),
            "cs_percentiles(close, 0.25, 0.75)".to_string(),
        ],
        outputs: vec!["wins_close".to_string(), "pct_close".to_string()],
        opts: CompileOptions::default(),
    }
}

fn wave14_cs_neutralize_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "cs_neutralize_ols(close, volume)".to_string(),
            "cs_neutralize_ols_multi(close, volume, elem_mul(volume, volume))".to_string(),
        ],
        outputs: vec!["neutral_ols".to_string(), "neutral_ols_multi".to_string()],
        opts: CompileOptions::default(),
    }
}

fn wave15_cs_neutralize_multi3_request() -> FactorRequest {
    FactorRequest {
        exprs: vec![
            "cs_neutralize(close)".to_string(),
            "cs_neutralize_ols_multi(close, volume, elem_mul(volume, volume), elem_mul(elem_mul(volume, volume), volume))".to_string(),
        ],
        outputs: vec![
            "neutral_alias".to_string(),
            "neutral_ols_multi3".to_string(),
        ],
        opts: CompileOptions::default(),
    }
}

fn bar_event(ts: i64, instrument_slot: u32, close: f64, volume: f64) -> EventEnvelope {
    bar_event_with_flags(ts, instrument_slot, close, volume, 0)
}

fn bar_event_ohlcv(
    ts: i64,
    instrument_slot: u32,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
) -> EventEnvelope {
    EventEnvelope {
        ts_event_ns: ts,
        ts_init_ns: ts,
        seq: 1,
        source_slot: 0,
        instrument_slot,
        quality_flags: 0,
        payload: Payload::Bar(BarLite {
            open,
            high,
            low,
            close,
            volume,
        }),
    }
}

fn bar_event_with_flags(
    ts: i64,
    instrument_slot: u32,
    close: f64,
    volume: f64,
    quality_flags: u32,
) -> EventEnvelope {
    EventEnvelope {
        ts_event_ns: ts,
        ts_init_ns: ts,
        seq: 1,
        source_slot: 0,
        instrument_slot,
        quality_flags,
        payload: Payload::Bar(BarLite {
            open: close,
            high: close,
            low: close,
            close,
            volume,
        }),
    }
}

fn quote_event(ts: i64, instrument_slot: u32, bid_price: f64, ask_price: f64) -> EventEnvelope {
    EventEnvelope {
        ts_event_ns: ts,
        ts_init_ns: ts,
        seq: 1,
        source_slot: 1,
        instrument_slot,
        quality_flags: 0,
        payload: Payload::QuoteTick(crate::types::QuoteTickLite {
            bid_price,
            ask_price,
            bid_size: 1.0,
            ask_size: 1.0,
        }),
    }
}

fn data_event(ts: i64, instrument_slot: u32, entries: Vec<(&str, f64)>) -> EventEnvelope {
    EventEnvelope {
        ts_event_ns: ts,
        ts_init_ns: ts,
        seq: 1,
        source_slot: 0,
        instrument_slot,
        quality_flags: 0,
        payload: Payload::Data(DataLite {
            values: entries
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        }),
    }
}

fn approx_eq(lhs: f64, rhs: f64) -> bool {
    (lhs.is_nan() && rhs.is_nan()) || (lhs - rhs).abs() < 1e-9
}

fn naive_moments(values: &[f64]) -> (f64, f64, f64, f64) {
    let n = values.len() as f64;
    let sum = values.iter().sum::<f64>();
    let mean = sum / n;
    let sum_sq = values.iter().map(|v| v * v).sum::<f64>();
    let sum_cu = values.iter().map(|v| v * v * v).sum::<f64>();
    let sum_qu = values
        .iter()
        .map(|v| {
            let sq = v * v;
            sq * sq
        })
        .sum::<f64>();
    let m2 = sum_sq - (sum * sum) / n;
    let std = (m2 / (n - 1.0)).sqrt();
    let mean_sq = mean * mean;
    let m3 = sum_cu - 3.0 * mean * sum_sq + 3.0 * mean_sq * sum - n * mean_sq * mean;
    let skew = (n * m3) / ((n - 1.0) * (n - 2.0) * std.powi(3));
    let mean_cu = mean_sq * mean;
    let mean_qu = mean_sq * mean_sq;
    let m4 =
        sum_qu - 4.0 * mean * sum_cu + 6.0 * mean_sq * sum_sq - 4.0 * mean_cu * sum + n * mean_qu;
    let kurt = (n * (n + 1.0) * m4) / ((n - 1.0) * (n - 2.0) * (n - 3.0) * std.powi(4))
        - 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0));
    (mean, std, skew, kurt)
}

fn synthetic_series(len: usize) -> Vec<f64> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let x = ((s >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
        out.push(x * 100.0);
    }
    out
}

fn ts_rank_series_with_reuse(series: &[f64], window: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; series.len()];
    let mut scratch = Vec::with_capacity(window.max(1));
    compute_ts_rank(series, window, &mut scratch, &mut out);
    out
}

fn ts_rank_series_with_alloc(series: &[f64], window: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; series.len()];
    for i in 0..series.len() {
        if i + 1 < window {
            continue;
        }
        let latest = series[i];
        if !latest.is_finite() {
            continue;
        }
        let mut scratch = Vec::with_capacity(window.max(1));
        let mut all_finite = true;
        for value in series.iter().take(i + 1).skip(i + 1 - window) {
            if !value.is_finite() {
                all_finite = false;
                break;
            }
            scratch.push(*value);
        }
        if !all_finite {
            continue;
        }
        scratch.sort_by(|a, b| a.total_cmp(b));
        let lower = scratch.partition_point(|v| *v < latest);
        let upper = scratch.partition_point(|v| *v <= latest);
        let avg_rank = ((lower + 1 + upper) as f64) * 0.5;
        out[i] = if window > 1 {
            (avg_rank - 1.0) / (window as f64 - 1.0)
        } else {
            0.0
        };
    }
    out
}

fn compute_ts_rank(series: &[f64], window: usize, scratch: &mut Vec<f64>, out: &mut [f64]) {
    for i in 0..series.len() {
        if i + 1 < window {
            continue;
        }
        let latest = series[i];
        if !latest.is_finite() {
            continue;
        }
        scratch.clear();
        let mut all_finite = true;
        for value in series.iter().take(i + 1).skip(i + 1 - window) {
            if !value.is_finite() {
                all_finite = false;
                break;
            }
            scratch.push(*value);
        }
        if !all_finite {
            continue;
        }
        scratch.sort_by(|a, b| a.total_cmp(b));
        let lower = scratch.partition_point(|v| *v < latest);
        let upper = scratch.partition_point(|v| *v <= latest);
        let avg_rank = ((lower + 1 + upper) as f64) * 0.5;
        out[i] = if window > 1 {
            (avg_rank - 1.0) / (window as f64 - 1.0)
        } else {
            0.0
        };
    }
}
