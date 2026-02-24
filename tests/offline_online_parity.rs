use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, BarLite, Engine, EventEnvelope, FactorRequest, InputFieldCatalog,
    OnlineFactorEngine, Payload, Planner, QuoteTickLite, SimplePlanner,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::process::{Command, Stdio};

const TOL: f64 = 1e-9;
const TOL_HIGHER_MOMENTS: f64 = 6e-8;
const TOL_CS_ZSCORE: f64 = 1e-8;

#[derive(Debug, Clone, Serialize)]
struct OfflineInputRow {
    ts: i64,
    instrument_slot: u32,
    fields: BTreeMap<String, f64>,
}

/// Legacy flat expression payload used by most non-neutralize parity tests.
#[derive(Debug, Clone, Serialize)]
struct LegacyOfflineExpr {
    output: String,
    op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    field: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lhs: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rhs: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    window: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lag: Option<usize>,
}

/// Legacy payload: rows + flat expressions.
#[derive(Debug, Serialize)]
struct LegacyOfflinePayload {
    rows: Vec<OfflineInputRow>,
    expressions: Vec<LegacyOfflineExpr>,
}

/// Structured payload: rows + JSON expressions (used by neutralize* tests).
#[derive(Debug, Serialize)]
struct OfflineJsonPayload {
    rows: Vec<OfflineInputRow>,
    expressions: Vec<JsonValue>,
}

#[derive(Debug, Clone, Serialize, Default)]
struct OfflineNeutralizeSpec {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    regressors: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    group: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    weights: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    standardize: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct OfflineRow {
    output: String,
    instrument_slot: u32,
    ts: i64,
    value: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct OfflineResponse {
    rows: Vec<OfflineRow>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CompareKey {
    output: String,
    instrument_slot: u32,
    ts: i64,
}

fn bar_event(ts: i64, instrument_slot: u32, close: f64, volume: f64) -> EventEnvelope {
    EventEnvelope {
        ts_event_ns: ts,
        ts_init_ns: ts,
        seq: 1,
        source_slot: 0,
        instrument_slot,
        quality_flags: 0,
        payload: Payload::Bar(BarLite {
            open: close,
            high: close,
            low: close,
            close,
            volume,
        }),
    }
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

fn quote_event(ts: i64, instrument_slot: u32, bid_price: f64) -> EventEnvelope {
    EventEnvelope {
        ts_event_ns: ts,
        ts_init_ns: ts,
        seq: 1,
        source_slot: 1,
        instrument_slot,
        quality_flags: 0,
        payload: Payload::QuoteTick(QuoteTickLite {
            bid_price,
            ask_price: bid_price + 0.05,
            bid_size: 1.0,
            ask_size: 1.0,
        }),
    }
}

fn python_polars_available() -> bool {
    Command::new("python3")
        .arg("-c")
        .arg("import polars")
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn run_polars_offline_with_mode<T: Serialize>(
    payload: &T,
    ts_rank_mode: Option<&str>,
) -> HashMap<CompareKey, Option<f64>> {
    let script = format!(
        "{}/tests/data/polars_offline_baseline.py",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut cmd = Command::new("python3");
    cmd.arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(mode) = ts_rank_mode {
        cmd.env("FACTOR_ENGINE_TS_RANK_MODE", mode);
    }
    let mut child = cmd.spawn().expect("failed to spawn python3");
    {
        let stdin = child.stdin.as_mut().expect("stdin unavailable");
        let bytes = serde_json::to_vec(payload).expect("serialize payload");
        stdin.write_all(&bytes).expect("write payload");
    }
    let output = child.wait_with_output().expect("python execution failed");
    if !output.status.success() {
        panic!(
            "python baseline failed: status={:?} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let parsed: OfflineResponse = serde_json::from_slice(&output.stdout).unwrap_or_else(|err| {
        panic!(
            "parse python output failed: {err}: {}",
            String::from_utf8_lossy(&output.stdout)
        )
    });
    parsed
        .rows
        .into_iter()
        .map(|row| {
            (
                CompareKey {
                    output: row.output,
                    instrument_slot: row.instrument_slot,
                    ts: row.ts,
                },
                row.value,
            )
        })
        .collect()
}

fn run_polars_offline<T: Serialize>(payload: &T) -> HashMap<CompareKey, Option<f64>> {
    run_polars_offline_with_mode(payload, None)
}

fn normalize(v: f64) -> Option<f64> {
    if v.is_finite() {
        Some(v)
    } else {
        None
    }
}

fn assert_close(lhs: Option<f64>, rhs: Option<f64>, key: &CompareKey) {
    let tol_abs = if key.output.contains("skew") || key.output.contains("kurt") {
        TOL_HIGHER_MOMENTS
    } else if key.output.contains("zscore") || key.output.ends_with("_z") {
        TOL_CS_ZSCORE
    } else {
        TOL
    };
    match (lhs, rhs) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            let abs = (a - b).abs();
            let scale = a.abs().max(b.abs()).max(1.0);
            let tol_rel = tol_abs * 10.0;
            assert!(
                abs <= tol_abs || abs <= tol_rel * scale,
                "mismatch at {:?}: lhs={} rhs={} abs={} tol_abs={} tol_rel={}",
                key,
                a,
                b,
                abs,
                tol_abs,
                tol_rel
            );
        }
        _ => panic!(
            "null/finite mismatch at {:?}: lhs={:?} rhs={:?}",
            key, lhs, rhs
        ),
    }
}

fn offline_neutralize_expr(
    output: &str,
    op: &str,
    field: &str,
    regressors: &[&str],
    group: Option<&str>,
    weights: Option<&str>,
    standardize: bool,
) -> JsonValue {
    let valid_layout = match op {
        "cs_neutralize" => regressors.is_empty(),
        "cs_neutralize_ols" => regressors.len() == 1,
        "cs_neutralize_ols_multi" => (1..=3).contains(&regressors.len()),
        _ => false,
    };
    assert!(
        valid_layout,
        "invalid neutralize layout: op={op}, regressors={}",
        regressors.len()
    );

    let spec = OfflineNeutralizeSpec {
        regressors: regressors.iter().map(|v| (*v).to_string()).collect(),
        group: group.map(ToString::to_string),
        weights: weights.map(ToString::to_string),
        standardize: standardize.then_some(true),
    };
    serde_json::json!({
        "output": output,
        "op": op,
        "field": field,
        "neutralize": spec,
    })
}

#[inline]
fn offline_cs_neutralize(
    output: &str,
    field: &str,
    group: Option<&str>,
    weights: Option<&str>,
    standardize: bool,
) -> JsonValue {
    offline_neutralize_expr(
        output,
        "cs_neutralize",
        field,
        &[],
        group,
        weights,
        standardize,
    )
}

#[inline]
fn offline_cs_neutralize_ols(
    output: &str,
    field: &str,
    regressor: &str,
    group: Option<&str>,
    weights: Option<&str>,
    standardize: bool,
) -> JsonValue {
    offline_neutralize_expr(
        output,
        "cs_neutralize_ols",
        field,
        &[regressor],
        group,
        weights,
        standardize,
    )
}

#[inline]
fn offline_cs_neutralize_ols_multi(
    output: &str,
    field: &str,
    regressors: &[&str],
    group: Option<&str>,
    weights: Option<&str>,
    standardize: bool,
) -> JsonValue {
    offline_neutralize_expr(
        output,
        "cs_neutralize_ols_multi",
        field,
        regressors,
        group,
        weights,
        standardize,
    )
}

fn max_abs_diff(
    lhs: &HashMap<CompareKey, Option<f64>>,
    rhs: &HashMap<CompareKey, Option<f64>>,
) -> f64 {
    let mut max_diff = 0.0_f64;
    for (key, a) in lhs {
        let b = rhs.get(key).expect("rhs key missing");
        match (a, b) {
            (Some(x), Some(y)) => {
                let d = (x - y).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
            (None, None) => {}
            _ => panic!("null/finite mismatch at {:?}", key),
        }
    }
    max_diff
}

fn retain_outputs(
    rows: &HashMap<CompareKey, Option<f64>>,
    outputs: &[String],
) -> HashMap<CompareKey, Option<f64>> {
    let wanted: BTreeSet<&str> = outputs.iter().map(|s| s.as_str()).collect();
    rows.iter()
        .filter(|(k, _)| wanted.contains(k.output.as_str()))
        .map(|(k, v)| (k.clone(), *v))
        .collect()
}

fn record_frame_results(
    out: &mut HashMap<CompareKey, Option<f64>>,
    frame: &factor_engine::FeatureFrame,
    outputs: &[String],
    universe: &[u32],
    ts: i64,
) {
    for output in outputs {
        let factor_idx = frame.factor_idx(output).expect("missing output");
        for (instrument_idx, &instrument_slot) in universe.iter().enumerate() {
            let value = frame
                .value_at(instrument_idx, factor_idx)
                .and_then(normalize);
            out.insert(
                CompareKey {
                    output: output.clone(),
                    instrument_slot,
                    ts,
                },
                value,
            );
        }
    }
}

#[test]
fn polars_offline_matches_online_bar_only_pipeline() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![101_u32, 202, 303];
    let ts_points: Vec<i64> = (1..=8).collect();
    let outputs = vec![
        "mean3".to_string(),
        "delta1".to_string(),
        "corr3".to_string(),
        "cs_rank_mean3".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "ts_mean(close, 3)".to_string(),
            "ts_delta(volume, 1)".to_string(),
            "ts_corr(close, volume, 3)".to_string(),
            "cs_rank(ts_mean(close, 3))".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = inst_idx as f64 + 1.0;
            let close = 10.0 * base + ts as f64 * 0.7 + (ts % 3) as f64 * 0.11;
            let volume = 100.0 + 8.0 * base + ts as f64 * 1.5 + (ts % 2) as f64 * 0.2;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }
    let offline = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "mean3".to_string(),
                op: "ts_mean".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "delta1".to_string(),
                op: "ts_delta".to_string(),
                field: Some("volume".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: Some(1),
            },
            LegacyOfflineExpr {
                output: "corr3".to_string(),
                op: "ts_corr".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("volume".to_string()),
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "cs_rank_mean3".to_string(),
                op: "cs_rank".to_string(),
                field: Some("mean3".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: None,
            },
        ],
    });
    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_cross_source_barrier_pipeline() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![1001_u32, 1002, 1003];
    let ts_points: Vec<i64> = (1..=8).collect();
    let outputs = vec![
        "corr_cross".to_string(),
        "mean_quote".to_string(),
        "cs_rank_quote".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "ts_corr(bar.close, quote_tick.bid_price, 3)".to_string(),
            "ts_mean(quote_tick.bid_price, 3)".to_string(),
            "cs_rank(quote_tick.bid_price)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec![
        "bar.close".to_string(),
        "quote_tick.bid_price".to_string(),
    ]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close = 20.0 + 2.0 * base + ts as f64 * 0.8 + (ts % 2) as f64 * 0.05;
            let bid = 10.0 + 1.5 * base + ts as f64 * 0.3 + (ts % 3) as f64 * 0.07;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, 0.0))
                .expect("bar event should succeed");
            engine
                .on_event(&quote_event(ts, instrument_slot, bid))
                .expect("quote event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("bid_price".to_string(), bid),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }
    let offline = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "corr_cross".to_string(),
                op: "ts_corr".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("bid_price".to_string()),
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "mean_quote".to_string(),
                op: "ts_mean".to_string(),
                field: Some("bid_price".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "cs_rank_quote".to_string(),
                op: "cs_rank".to_string(),
                field: Some("bid_price".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: None,
            },
        ],
    });
    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_extended_ts_and_cs_ops() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![3001_u32, 3002, 3003];
    let ts_points: Vec<i64> = (1..=10).collect();
    let outputs = vec![
        "std3".to_string(),
        "rank3".to_string(),
        "linreg3".to_string(),
        "zscore_close".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "ts_std(close, 3)".to_string(),
            "ts_rank(close, 3)".to_string(),
            "ts_linear_regression(close, volume, 3)".to_string(),
            "cs_zscore(close)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close = 30.0 + 3.2 * base + ts as f64 * 0.9 + (ts % 3) as f64 * 0.13;
            let volume = 90.0 + 5.0 * base + ts as f64 * 1.7 + (inst_idx % 2) as f64 * 0.31;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "std3".to_string(),
                op: "ts_std".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "rank3".to_string(),
                op: "ts_rank".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "linreg3".to_string(),
                op: "ts_linear_regression".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("volume".to_string()),
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "zscore_close".to_string(),
                op: "cs_zscore".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: None,
            },
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_elem_and_higher_moments_ops() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![4101_u32, 4102, 4103];
    let ts_points: Vec<i64> = (1..=14).collect();
    let outputs = vec![
        "elem_add_cv".to_string(),
        "elem_sub_cv".to_string(),
        "elem_mul_cv".to_string(),
        "elem_div_cv".to_string(),
        "skew4".to_string(),
        "kurt5".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "close + volume".to_string(),
            "close - volume".to_string(),
            "close * volume".to_string(),
            "close / volume".to_string(),
            "ts_skew(close, 4)".to_string(),
            "ts_kurt(close, 5)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close = 50.0 + 2.7 * base + ts as f64 * 0.6 + (ts % 4) as f64 * 0.21;
            let volume = 120.0 + 3.1 * base + ts as f64 * 0.8 + (inst_idx % 2) as f64 * 0.17;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "elem_add_cv".to_string(),
                op: "elem_add".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("volume".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "elem_sub_cv".to_string(),
                op: "elem_sub".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("volume".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "elem_mul_cv".to_string(),
                op: "elem_mul".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("volume".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "elem_div_cv".to_string(),
                op: "elem_div".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("volume".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "skew4".to_string(),
                op: "ts_skew".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(4),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "kurt5".to_string(),
                op: "ts_kurt".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(5),
                lag: None,
            },
        ],
    });

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_wave1_new_ops() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![4201_u32, 4202, 4203];
    let ts_points: Vec<i64> = (1..=14).collect();
    let outputs = vec![
        "sum3".to_string(),
        "min3".to_string(),
        "max3".to_string(),
        "pow_close2".to_string(),
        "min_cv".to_string(),
        "max_cv".to_string(),
        "sum3_maxcv".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "ts_sum(close, 3)".to_string(),
            "ts_min(close, 3)".to_string(),
            "ts_max(close, 3)".to_string(),
            "elem_pow(close, 2)".to_string(),
            "elem_min(close, volume)".to_string(),
            "elem_max(close, volume)".to_string(),
            "ts_sum(elem_max(close, volume), 3)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close = 40.0 + 1.9 * base + ts as f64 * 0.73 + (ts % 3) as f64 * 0.17;
            let volume = 65.0 + 4.1 * base + ts as f64 * 1.27 + (inst_idx % 2) as f64 * 0.23;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "sum3".to_string(),
                op: "ts_sum".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "min3".to_string(),
                op: "ts_min".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "max3".to_string(),
                op: "ts_max".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "pow_close2".to_string(),
                op: "elem_pow".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("2".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "min_cv".to_string(),
                op: "elem_min".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("volume".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "max_cv".to_string(),
                op: "elem_max".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("volume".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "sum3_maxcv".to_string(),
                op: "ts_sum".to_string(),
                field: Some("max_cv".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
        ],
    });

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_wave2_ts_stats_ops() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![4301_u32, 4302, 4303];
    let ts_points: Vec<i64> = (1..=16).collect();
    let outputs = vec![
        "lag1".to_string(),
        "z3".to_string(),
        "cov3".to_string(),
        "z_lag1_3".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "ts_lag(close, 1)".to_string(),
            "ts_zscore(close, 3)".to_string(),
            "ts_cov(close, volume, 3)".to_string(),
            "ts_zscore(ts_lag(close, 1), 3)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert!(
        manifest.cse_hit_count >= 1,
        "expected CSE in wave2 graph, got manifest={manifest:?}"
    );
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close = 35.0 + 1.8 * base + ts as f64 * 0.83 + (ts % 3) as f64 * 0.12;
            let volume = 88.0 + 2.9 * base + ts as f64 * 1.21 + (inst_idx % 2) as f64 * 0.19;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "lag1".to_string(),
                op: "ts_lag".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: Some(1),
            },
            LegacyOfflineExpr {
                output: "z3".to_string(),
                op: "ts_zscore".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "cov3".to_string(),
                op: "ts_cov".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("volume".to_string()),
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__lag1".to_string(),
                op: "ts_lag".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: Some(1),
            },
            LegacyOfflineExpr {
                output: "z_lag1_3".to_string(),
                op: "ts_zscore".to_string(),
                field: Some("__lag1".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_wave3_var_beta_ops() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![4401_u32, 4402, 4403];
    let ts_points: Vec<i64> = (1..=18).collect();
    let outputs = vec![
        "var4".to_string(),
        "beta_cv_4".to_string(),
        "beta_vc_4".to_string(),
        "var_mean3_3".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "ts_var(close, 4)".to_string(),
            "ts_beta(close, volume, 4)".to_string(),
            "ts_beta(volume, close, 4)".to_string(),
            "ts_var(ts_mean(close, 3), 3)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close = 45.0 + 2.2 * base + ts as f64 * 0.67 + (ts % 4) as f64 * 0.15;
            let volume = 92.0 + 3.7 * base + ts as f64 * 1.09 + (inst_idx % 2) as f64 * 0.21;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "var4".to_string(),
                op: "ts_var".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(4),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "beta_cv_4".to_string(),
                op: "ts_beta".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("volume".to_string()),
                window: Some(4),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "beta_vc_4".to_string(),
                op: "ts_beta".to_string(),
                field: None,
                lhs: Some("volume".to_string()),
                rhs: Some("close".to_string()),
                window: Some(4),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__mean_close3".to_string(),
                op: "ts_mean".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "var_mean3_3".to_string(),
                op: "ts_var".to_string(),
                field: Some("__mean_close3".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_wave4_ewm_ops() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![4501_u32, 4502, 4503];
    let ts_points: Vec<i64> = (1..=18).collect();
    let outputs = vec![
        "ewm_mean3".to_string(),
        "ewm_var3".to_string(),
        "ewm_cov3".to_string(),
        "ewm_mean_lag1_3".to_string(),
        "ewm_var_lag1_3".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "ts_ewm_mean(close, 3)".to_string(),
            "ts_ewm_var(close, 3)".to_string(),
            "ts_ewm_cov(close, volume, 3)".to_string(),
            "ts_ewm_mean(ts_lag(close, 1), 3)".to_string(),
            "ts_ewm_var(ts_lag(close, 1), 3)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert!(
        manifest.cse_hit_count >= 1,
        "expected CSE in wave4 graph, got manifest={manifest:?}"
    );
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close = 48.0 + 1.7 * base + ts as f64 * 0.74 + (ts % 3) as f64 * 0.14;
            let volume = 104.0 + 2.8 * base + ts as f64 * 1.06 + (inst_idx % 2) as f64 * 0.22;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "ewm_mean3".to_string(),
                op: "ts_ewm_mean".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "ewm_var3".to_string(),
                op: "ts_ewm_var".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "ewm_cov3".to_string(),
                op: "ts_ewm_cov".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("volume".to_string()),
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__lag1".to_string(),
                op: "ts_lag".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: Some(1),
            },
            LegacyOfflineExpr {
                output: "ewm_mean_lag1_3".to_string(),
                op: "ts_ewm_mean".to_string(),
                field: Some("__lag1".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "ewm_var_lag1_3".to_string(),
                op: "ts_ewm_var".to_string(),
                field: Some("__lag1".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_wave5_quantile_ops() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![4601_u32, 4602, 4603];
    let ts_points: Vec<i64> = (1..=18).collect();
    let outputs = vec![
        "q25_4".to_string(),
        "q50_lag1_4".to_string(),
        "q75_lag1_4".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "ts_quantile(close, 4, 0.25)".to_string(),
            "ts_quantile(ts_lag(close, 1), 4, 0.5)".to_string(),
            "ts_quantile(ts_lag(close, 1), window=4, q=0.75)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert!(
        manifest.cse_hit_count >= 1,
        "expected CSE in wave5 graph, got manifest={manifest:?}"
    );
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close = 52.0 + 1.9 * base + ts as f64 * 0.69 + (ts % 4) as f64 * 0.12;
            let volume = 96.0 + 3.5 * base + ts as f64 * 1.11 + (inst_idx % 2) as f64 * 0.31;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "q25_4".to_string(),
                op: "ts_quantile".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: Some("0.25".to_string()),
                window: Some(4),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__lag1".to_string(),
                op: "ts_lag".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: Some(1),
            },
            LegacyOfflineExpr {
                output: "q50_lag1_4".to_string(),
                op: "ts_quantile".to_string(),
                field: Some("__lag1".to_string()),
                lhs: None,
                rhs: Some("0.5".to_string()),
                window: Some(4),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "q75_lag1_4".to_string(),
                op: "ts_quantile".to_string(),
                field: Some("__lag1".to_string()),
                lhs: None,
                rhs: Some("0.75".to_string()),
                window: Some(4),
                lag: None,
            },
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_wave6_argext_ops() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![4701_u32, 4702, 4703];
    let ts_points: Vec<i64> = (1..=18).collect();
    let outputs = vec![
        "argmax4".to_string(),
        "argmin4".to_string(),
        "argmax_lag1_4".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "ts_argmax(close, 4)".to_string(),
            "ts_argmin(close, 4)".to_string(),
            "ts_argmax(ts_lag(close, 1), 4)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close = 51.0 + 2.1 * base + ts as f64 * 0.71 + (ts % 5) as f64 * 0.17;
            let volume = 98.0 + 2.9 * base + ts as f64 * 1.08 + (inst_idx % 2) as f64 * 0.21;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "argmax4".to_string(),
                op: "ts_argmax".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(4),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "argmin4".to_string(),
                op: "ts_argmin".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(4),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__lag1".to_string(),
                op: "ts_lag".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: Some(1),
            },
            LegacyOfflineExpr {
                output: "argmax_lag1_4".to_string(),
                op: "ts_argmax".to_string(),
                field: Some("__lag1".to_string()),
                lhs: None,
                rhs: None,
                window: Some(4),
                lag: None,
            },
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_wave7_elem_unary_ops() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![4801_u32, 4802, 4803];
    let ts_points: Vec<i64> = (1..=18).collect();
    let outputs = vec![
        "abs_delta1".to_string(),
        "log_close".to_string(),
        "sign_dev3".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "elem_abs(ts_delta(close, 1))".to_string(),
            "elem_log(close)".to_string(),
            "elem_sign(close - ts_mean(close, 3))".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close = 49.0 + 1.5 * base + ts as f64 * 0.83 + (ts % 3) as f64 * 0.16;
            let volume = 101.0 + 2.4 * base + ts as f64 * 1.03 + (inst_idx % 2) as f64 * 0.18;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "__delta1".to_string(),
                op: "ts_delta".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: Some(1),
            },
            LegacyOfflineExpr {
                output: "abs_delta1".to_string(),
                op: "elem_abs".to_string(),
                field: Some("__delta1".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "log_close".to_string(),
                op: "elem_log".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__mean3".to_string(),
                op: "ts_mean".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__dev3".to_string(),
                op: "elem_sub".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("__mean3".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "sign_dev3".to_string(),
                op: "elem_sign".to_string(),
                field: Some("__dev3".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: None,
            },
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_wave8_elem_extended_ops() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![4901_u32, 4902, 4903];
    let ts_points: Vec<i64> = (1..=18).collect();
    let outputs = vec![
        "exp_lag1".to_string(),
        "sqrt_abs_delta1".to_string(),
        "clip_close".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "elem_exp(ts_lag(close, 1))".to_string(),
            "elem_sqrt(elem_abs(ts_delta(close, 1)))".to_string(),
            "elem_clip(close, 2.5, 4.5)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close = 1.9 + 0.6 * base + ts as f64 * 0.19 + (ts % 3) as f64 * 0.04;
            let volume = 98.0 + 2.4 * base + ts as f64 * 1.05 + (inst_idx % 2) as f64 * 0.18;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "__lag1".to_string(),
                op: "ts_lag".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: Some(1),
            },
            LegacyOfflineExpr {
                output: "exp_lag1".to_string(),
                op: "elem_exp".to_string(),
                field: Some("__lag1".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__delta1".to_string(),
                op: "ts_delta".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: Some(1),
            },
            LegacyOfflineExpr {
                output: "__abs_delta1".to_string(),
                op: "elem_abs".to_string(),
                field: Some("__delta1".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "sqrt_abs_delta1".to_string(),
                op: "elem_sqrt".to_string(),
                field: Some("__abs_delta1".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "clip_close".to_string(),
                op: "elem_clip".to_string(),
                field: Some("close".to_string()),
                lhs: Some("2.5".to_string()),
                rhs: Some("4.5".to_string()),
                window: None,
                lag: None,
            },
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_wave9_elem_conditional_ops() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![5001_u32, 5002, 5003];
    let ts_points: Vec<i64> = (1..=18).collect();
    let outputs = vec![
        "where_one".to_string(),
        "fillna_div0".to_string(),
        "where_zero".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "elem_where(1.0, close, volume)".to_string(),
            "elem_fillna(elem_div(close, close - close), 0.0)".to_string(),
            "elem_where(0.0, close, volume)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close = 2.0 + 0.5 * base + ts as f64 * 0.23 + (ts % 3) as f64 * 0.05;
            let volume = 10.0 + 1.2 * base + ts as f64 * 0.41 + (inst_idx % 2) as f64 * 0.13;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "where_one".to_string(),
                op: "elem_where".to_string(),
                field: Some("1.0".to_string()),
                lhs: Some("close".to_string()),
                rhs: Some("volume".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__zero".to_string(),
                op: "elem_sub".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("close".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__div0".to_string(),
                op: "elem_div".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("__zero".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "fillna_div0".to_string(),
                op: "elem_fillna".to_string(),
                field: Some("__div0".to_string()),
                lhs: Some("0.0".to_string()),
                rhs: None,
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "where_zero".to_string(),
                op: "elem_where".to_string(),
                field: Some("0.0".to_string()),
                lhs: Some("close".to_string()),
                rhs: Some("volume".to_string()),
                window: None,
                lag: None,
            },
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_wave10_ts_decay_linear() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![5101_u32, 5102, 5103];
    let ts_points: Vec<i64> = (1..=18).collect();
    let outputs = vec!["decay4".to_string(), "decay4_lag1".to_string()];
    let request = FactorRequest {
        exprs: vec![
            "ts_decay_linear(close, 4)".to_string(),
            "ts_decay_linear(ts_lag(close, 1), 4)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close = 4.0 + 0.85 * base + ts as f64 * 0.37 + (ts % 3) as f64 * 0.07;
            let volume = 60.0 + 1.7 * base + ts as f64 * 1.11 + (inst_idx % 2) as f64 * 0.2;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "decay4".to_string(),
                op: "ts_decay_linear".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(4),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__lag1".to_string(),
                op: "ts_lag".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: Some(1),
            },
            LegacyOfflineExpr {
                output: "decay4_lag1".to_string(),
                op: "ts_decay_linear".to_string(),
                field: Some("__lag1".to_string()),
                lhs: None,
                rhs: None,
                window: Some(4),
                lag: None,
            },
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_wave11_ts_product_mad() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![5201_u32, 5202, 5203];
    let ts_points: Vec<i64> = (1..=18).collect();
    let outputs = vec![
        "product4".to_string(),
        "mad4".to_string(),
        "mad4_lag1".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "ts_product(close, 4)".to_string(),
            "ts_mad(close, 4)".to_string(),
            "ts_mad(ts_lag(close, 1), 4)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close = 2.0 + 0.55 * base + ts as f64 * 0.21 + (ts % 3) as f64 * 0.04;
            let volume = 42.0 + 1.6 * base + ts as f64 * 0.77 + (inst_idx % 2) as f64 * 0.12;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "product4".to_string(),
                op: "ts_product".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(4),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "mad4".to_string(),
                op: "ts_mad".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(4),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__lag1".to_string(),
                op: "ts_lag".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: Some(1),
            },
            LegacyOfflineExpr {
                output: "mad4_lag1".to_string(),
                op: "ts_mad".to_string(),
                field: Some("__lag1".to_string()),
                lhs: None,
                rhs: None,
                window: Some(4),
                lag: None,
            },
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_wave12_cs_preprocess_ops() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![5301_u32, 5302, 5303];
    let ts_points: Vec<i64> = (1..=18).collect();
    let outputs = vec![
        "center_close".to_string(),
        "norm_close".to_string(),
        "fillna_close".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "cs_center(close)".to_string(),
            "cs_norm(close)".to_string(),
            "cs_fillna(close)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close_raw = 2.5 + 0.8 * base + ts as f64 * 0.29 + (ts % 3) as f64 * 0.06;
            let close = if inst_idx == 2 && ts % 5 == 0 {
                f64::NAN
            } else {
                close_raw
            };
            let volume = 50.0 + 1.4 * base + ts as f64 * 0.66;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "center_close".to_string(),
                op: "cs_center".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "norm_close".to_string(),
                op: "cs_norm".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "fillna_close".to_string(),
                op: "cs_fillna".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: None,
            },
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_wave13_cs_clip_ops() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![5401_u32, 5402, 5403];
    let ts_points: Vec<i64> = (1..=18).collect();
    let outputs = vec!["wins_close".to_string(), "pct_close".to_string()];
    let request = FactorRequest {
        exprs: vec![
            "cs_winsorize(close, 0.1)".to_string(),
            "cs_percentiles(close, 0.25, 0.75)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close_raw = 3.0 + 0.7 * base + ts as f64 * 0.33 + (ts % 3) as f64 * 0.05;
            let close = if inst_idx == 2 && ts % 6 == 0 {
                f64::NAN
            } else {
                close_raw
            };
            let volume = 70.0 + 1.9 * base + ts as f64 * 0.9;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "wins_close".to_string(),
                op: "cs_winsorize".to_string(),
                field: Some("close".to_string()),
                lhs: Some("0.1".to_string()),
                rhs: None,
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "pct_close".to_string(),
                op: "cs_percentiles".to_string(),
                field: Some("close".to_string()),
                lhs: Some("0.25".to_string()),
                rhs: Some("0.75".to_string()),
                window: None,
                lag: None,
            },
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_wave14_cs_neutralize_ops() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![6401_u32, 6402, 6403, 6404];
    let ts_points: Vec<i64> = (1..=14).collect();
    let outputs = vec!["neutral_ols".to_string(), "neutral_ols_multi".to_string()];
    let request = FactorRequest {
        exprs: vec![
            "cs_neutralize_ols(close, volume)".to_string(),
            "cs_neutralize_ols_multi(close, volume, elem_mul(volume, volume))".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = 1.0 + inst_idx as f64 * 0.7;
            let volume = base + ts as f64 * 0.35;
            let close = 1.2 + 0.8 * volume + 0.2 * volume * volume + (inst_idx as f64 - 1.5) * 0.11;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&OfflineJsonPayload {
        rows: offline_rows,
        expressions: vec![
            serde_json::json!({
                "output": "vol_sq",
                "op": "elem_mul",
                "lhs": "volume",
                "rhs": "volume",
            }),
            offline_cs_neutralize_ols("neutral_ols", "close", "volume", None, None, false),
            offline_cs_neutralize_ols_multi(
                "neutral_ols_multi",
                "close",
                &["volume", "vol_sq"],
                None,
                None,
                false,
            ),
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_cs_neutralize_kwargs_layouts() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![6501_u32, 6502, 6503, 6504];
    let ts_points: Vec<i64> = (1..=20).collect();
    let outputs = vec![
        "neutral_gws".to_string(),
        "neutral_ols_gws".to_string(),
        "neutral_ols_multi_gs".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "cs_neutralize(close, group=volume, weights=open, standardize=1)".to_string(),
            "cs_neutralize_ols(close, open, group=volume, weights=high, standardize=1)".to_string(),
            "cs_neutralize_ols_multi(close, open, high, group=volume, standardize=1)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec![
        "bar.open".to_string(),
        "bar.high".to_string(),
        "bar.close".to_string(),
        "bar.volume".to_string(),
    ]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let inst = inst_idx as f64;
            let group = if inst_idx < 2 { 0.0 } else { 1.0 };
            let open = 1.5 + inst * 0.35 + ts as f64 * 0.07;
            let high = open + 0.4 + inst * 0.05;
            let low = open - 0.3;
            let close =
                2.0 + 1.2 * open - 0.45 * high + 0.3 * group + ts as f64 * 0.02 + inst * 0.03;
            let volume = group;

            engine
                .on_event(&bar_event_ohlcv(
                    ts,
                    instrument_slot,
                    open,
                    high,
                    low,
                    close,
                    volume,
                ))
                .expect("bar event should succeed");

            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("open".to_string(), open),
                    ("high".to_string(), high),
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&OfflineJsonPayload {
        rows: offline_rows,
        expressions: vec![
            offline_cs_neutralize("neutral_gws", "close", Some("volume"), Some("open"), true),
            offline_cs_neutralize_ols(
                "neutral_ols_gws",
                "close",
                "open",
                Some("volume"),
                Some("high"),
                true,
            ),
            offline_cs_neutralize_ols_multi(
                "neutral_ols_multi_gs",
                "close",
                &["open", "high"],
                Some("volume"),
                None,
                true,
            ),
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_cs_neutralize_multi3_standardize() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![6601_u32, 6602, 6603, 6604];
    let ts_points: Vec<i64> = (1..=20).collect();
    let outputs = vec![
        "neutral_center".to_string(),
        "neutral_multi3_gs".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "cs_neutralize(close)".to_string(),
            "cs_neutralize_ols_multi(close, open, high, low, standardize=1)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec![
        "bar.open".to_string(),
        "bar.high".to_string(),
        "bar.low".to_string(),
        "bar.close".to_string(),
        "bar.volume".to_string(),
    ]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let inst = inst_idx as f64;
            let group = if inst_idx < 2 { 0.0 } else { 1.0 };
            let open = 1.2 + inst * 0.25 + ts as f64 * 0.05;
            let high = open + 0.45 + inst * 0.03;
            let low = open - 0.35 - inst * 0.02;
            let close = 3.0 + 0.9 * open - 0.3 * high + 0.25 * low + 0.2 * group + 0.01 * ts as f64;
            let volume = group;

            engine
                .on_event(&bar_event_ohlcv(
                    ts,
                    instrument_slot,
                    open,
                    high,
                    low,
                    close,
                    volume,
                ))
                .expect("bar event should succeed");

            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("open".to_string(), open),
                    ("high".to_string(), high),
                    ("low".to_string(), low),
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&OfflineJsonPayload {
        rows: offline_rows,
        expressions: vec![
            offline_cs_neutralize("neutral_center", "close", None, None, false),
            offline_cs_neutralize_ols_multi(
                "neutral_multi3_gs",
                "close",
                &["open", "high", "low"],
                None,
                None,
                true,
            ),
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_cs_neutralize_weight_filtering() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![6701_u32, 6702, 6703, 6704];
    let ts_points: Vec<i64> = (1..=16).collect();
    let outputs = vec![
        "neutral_weighted".to_string(),
        "neutral_ols_weighted".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "cs_neutralize(close, weights=open, standardize=1)".to_string(),
            "cs_neutralize_ols(close, high, weights=open, standardize=1)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec![
        "bar.open".to_string(),
        "bar.high".to_string(),
        "bar.close".to_string(),
    ]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let inst = inst_idx as f64;
            let weight = match inst_idx {
                0 => 1.0 + ts as f64 * 0.01,
                1 => 2.0 + ts as f64 * 0.02,
                2 => -1.0, // invalid weight: filtered out
                _ => 0.0,  // invalid weight: filtered out
            };
            let high = 3.0 + inst * 0.4 + ts as f64 * 0.03;
            let close = 5.0 + 0.8 * high + inst * 0.2 + ts as f64 * 0.01;

            engine
                .on_event(&bar_event_ohlcv(
                    ts,
                    instrument_slot,
                    weight,
                    high,
                    high,
                    close,
                    1.0,
                ))
                .expect("bar event should succeed");

            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("open".to_string(), weight),
                    ("high".to_string(), high),
                    ("close".to_string(), close),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&OfflineJsonPayload {
        rows: offline_rows,
        expressions: vec![
            offline_cs_neutralize("neutral_weighted", "close", None, Some("open"), true),
            offline_cs_neutralize_ols(
                "neutral_ols_weighted",
                "close",
                "high",
                None,
                Some("open"),
                true,
            ),
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_cs_neutralize_group_nan_filtering() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![6751_u32, 6752, 6753, 6754];
    let ts_points: Vec<i64> = (1..=18).collect();
    let outputs = vec!["neutral_group".to_string(), "neutral_ols_group".to_string()];
    let request = FactorRequest {
        exprs: vec![
            "cs_neutralize(close, group=volume)".to_string(),
            "cs_neutralize_ols(close, open, group=volume)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec![
        "bar.open".to_string(),
        "bar.close".to_string(),
        "bar.volume".to_string(),
    ]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let inst = inst_idx as f64;
            let open = 1.0 + inst * 0.4 + ts as f64 * 0.05;
            let close = 2.0 + 1.3 * open + inst * 0.2 + ts as f64 * 0.02;
            let group = match inst_idx {
                0 | 1 => 0.0,
                2 => 1.0,
                _ => {
                    if ts % 4 == 0 {
                        f64::NAN
                    } else {
                        1.0
                    }
                }
            };

            engine
                .on_event(&bar_event_ohlcv(
                    ts,
                    instrument_slot,
                    open,
                    open + 0.2,
                    open - 0.2,
                    close,
                    group,
                ))
                .expect("bar event should succeed");

            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("open".to_string(), open),
                    ("close".to_string(), close),
                    ("volume".to_string(), group),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&OfflineJsonPayload {
        rows: offline_rows,
        expressions: vec![
            offline_cs_neutralize("neutral_group", "close", Some("volume"), None, false),
            offline_cs_neutralize_ols(
                "neutral_ols_group",
                "close",
                "open",
                Some("volume"),
                None,
                false,
            ),
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_cs_neutralize_weighted_standardize_single_sample() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![6771_u32, 6772, 6773, 6774];
    let ts_points: Vec<i64> = (1..=12).collect();
    let outputs = vec![
        "neutral_wstd_single".to_string(),
        "neutral_ols_wstd_single".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "cs_neutralize(close, weights=open, standardize=1)".to_string(),
            "cs_neutralize_ols(close, high, weights=open, standardize=1)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec![
        "bar.open".to_string(),
        "bar.high".to_string(),
        "bar.close".to_string(),
    ]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let inst = inst_idx as f64;
            let weight = match inst_idx {
                0 => 1.0 + ts as f64 * 0.01,
                1 => 0.0,
                2 => -1.0,
                _ => f64::NAN,
            };
            let high = 4.0 + inst * 0.3 + ts as f64 * 0.02;
            let close = 6.0 + 0.7 * high + inst * 0.15 + ts as f64 * 0.01;

            engine
                .on_event(&bar_event_ohlcv(
                    ts,
                    instrument_slot,
                    weight,
                    high,
                    high - 0.2,
                    close,
                    1.0,
                ))
                .expect("bar event should succeed");

            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("open".to_string(), weight),
                    ("high".to_string(), high),
                    ("close".to_string(), close),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&OfflineJsonPayload {
        rows: offline_rows,
        expressions: vec![
            offline_cs_neutralize("neutral_wstd_single", "close", None, Some("open"), true),
            offline_cs_neutralize_ols(
                "neutral_ols_wstd_single",
                "close",
                "high",
                None,
                Some("open"),
                true,
            ),
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
    assert!(
        streaming.values().all(|v| v.is_none()),
        "expected all outputs invalid when weighted standardize has only one valid sample"
    );
}

#[test]
fn polars_offline_matches_online_cs_neutralize_ols_degenerate_varx() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![6801_u32, 6802, 6803, 6804];
    let ts_points: Vec<i64> = (1..=18).collect();
    let outputs = vec!["neutral_ols_flatx".to_string(), "center_close".to_string()];
    let request = FactorRequest {
        exprs: vec![
            "cs_neutralize_ols(close, high)".to_string(),
            "cs_center(close)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec!["bar.high".to_string(), "bar.close".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        let x_const = 10.0 + ts as f64 * 0.02;
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let inst = inst_idx as f64;
            let close = 4.0 + inst * 0.6 + ts as f64 * 0.03;
            engine
                .on_event(&bar_event_ohlcv(
                    ts,
                    instrument_slot,
                    close,
                    x_const,
                    close,
                    close,
                    1.0,
                ))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("high".to_string(), x_const),
                    ("close".to_string(), close),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&OfflineJsonPayload {
        rows: offline_rows,
        expressions: vec![
            offline_cs_neutralize_ols("neutral_ols_flatx", "close", "high", None, None, false),
            serde_json::json!({
                "output": "center_close",
                "op": "cs_center",
                "field": "close",
            }),
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
#[ignore = "manual performance comparison for ts_rank offline implementations"]
fn compare_ts_rank_offline_modes_against_online() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe: Vec<u32> = (0..64).map(|i| 10_000 + i).collect();
    let ts_points: Vec<i64> = (1..=1200).collect();
    let output = "rank20".to_string();
    let request = FactorRequest {
        exprs: vec!["ts_rank(close, 20)".to_string()],
        outputs: vec![output.clone()],
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut online = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = inst_idx as f64 * 0.03125;
            let close = 100.0 + base + (ts as f64 * 0.07).sin() + (ts as f64 * 0.013).cos();
            engine
                .on_event(&bar_event(ts, instrument_slot, close, 0.0))
                .expect("event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([("close".to_string(), close)]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(
            &mut online,
            &frame,
            std::slice::from_ref(&output),
            &universe,
            ts,
        );
    }

    let payload = LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![LegacyOfflineExpr {
            output: output.clone(),
            op: "ts_rank".to_string(),
            field: Some("close".to_string()),
            lhs: None,
            rhs: None,
            window: Some(20),
            lag: None,
        }],
    };

    let t0 = std::time::Instant::now();
    let rolling_map = run_polars_offline_with_mode(&payload, Some("rolling_map"));
    let dt_rolling = t0.elapsed();

    let t1 = std::time::Instant::now();
    let loop_mode = run_polars_offline_with_mode(&payload, Some("loop"));
    let dt_loop = t1.elapsed();

    assert_eq!(online.len(), rolling_map.len());
    assert_eq!(online.len(), loop_mode.len());

    let err_rolling = max_abs_diff(&online, &rolling_map);
    let err_loop = max_abs_diff(&online, &loop_mode);
    let err_between = max_abs_diff(&rolling_map, &loop_mode);

    eprintln!(
        "ts_rank offline compare: rolling_map={:?}, loop={:?}, err(online,rolling_map)={}, err(online,loop)={}, err(rolling_map,loop)={}",
        dt_rolling, dt_loop, err_rolling, err_loop, err_between
    );

    assert!(
        err_rolling <= TOL,
        "rolling_map error too large: {}",
        err_rolling
    );
    assert!(err_loop <= TOL, "loop error too large: {}", err_loop);
}

#[test]
fn polars_offline_matches_online_complex_multi_expr_graph() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![5101_u32, 5102, 5103, 5104];
    let ts_points: Vec<i64> = (1..=16).collect();
    let outputs = vec![
        "a_mean_close3".to_string(),
        "b_cs_rank_a".to_string(),
        "c_corr_means".to_string(),
        "d_reg_means".to_string(),
        "f_elem_mix".to_string(),
    ];

    let request = FactorRequest {
        exprs: vec![
            "ts_mean(close + 0.0, 3)".to_string(),
            "cs_rank(ts_mean(close + 0.0, 3))".to_string(),
            "ts_corr(ts_mean(close, 3), ts_mean(volume, 3), 4)".to_string(),
            "ts_linear_regression(ts_mean(close, 3), ts_mean(volume, 3), 4)".to_string(),
            "(ts_mean(close, 3) + ts_mean(close, 3)) / (1 + ts_std(close, 3))".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert!(
        manifest.cse_hit_count >= 3,
        "expected CSE in complex graph, got manifest={manifest:?}"
    );
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close = 80.0 + 2.3 * base + ts as f64 * 0.47 + (ts % 3) as f64 * 0.19;
            let volume = 200.0 + 5.7 * base + ts as f64 * 1.13 + (inst_idx % 2) as f64 * 0.27;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, volume))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "__close_plus_zero".to_string(),
                op: "elem_add".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("0.0".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "a_mean_close3".to_string(),
                op: "ts_mean".to_string(),
                field: Some("__close_plus_zero".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "b_cs_rank_a".to_string(),
                op: "cs_rank".to_string(),
                field: Some("a_mean_close3".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__mean_close3".to_string(),
                op: "ts_mean".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__mean_volume3".to_string(),
                op: "ts_mean".to_string(),
                field: Some("volume".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "c_corr_means".to_string(),
                op: "ts_corr".to_string(),
                field: None,
                lhs: Some("__mean_close3".to_string()),
                rhs: Some("__mean_volume3".to_string()),
                window: Some(4),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "d_reg_means".to_string(),
                op: "ts_linear_regression".to_string(),
                field: None,
                lhs: Some("__mean_close3".to_string()),
                rhs: Some("__mean_volume3".to_string()),
                window: Some(4),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__std_close3".to_string(),
                op: "ts_std".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__sum_mean_close".to_string(),
                op: "elem_add".to_string(),
                field: None,
                lhs: Some("__mean_close3".to_string()),
                rhs: Some("__mean_close3".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__den".to_string(),
                op: "elem_add".to_string(),
                field: None,
                lhs: Some("1".to_string()),
                rhs: Some("__std_close3".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "f_elem_mix".to_string(),
                op: "elem_div".to_string(),
                field: None,
                lhs: Some("__sum_mean_close".to_string()),
                rhs: Some("__den".to_string()),
                window: None,
                lag: None,
            },
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_cross_source_nested_cs_ts_graph() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![6101_u32, 6102, 6103, 6104];
    let ts_points: Vec<i64> = (1..=18).collect();
    let outputs = vec![
        "cross_corr".to_string(),
        "cross_corr_z".to_string(),
        "cross_corr_rank".to_string(),
        "mean_close3".to_string(),
        "mean_bid3".to_string(),
    ];

    let request = FactorRequest {
        exprs: vec![
            "ts_corr(bar.close, quote_tick.bid_price, 3)".to_string(),
            "cs_zscore(ts_corr(bar.close, quote_tick.bid_price, 3))".to_string(),
            "cs_rank(ts_corr(bar.close, quote_tick.bid_price, 3))".to_string(),
            "ts_mean(bar.close, 3)".to_string(),
            "ts_mean(quote_tick.bid_price, 3)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert!(
        manifest.cse_hit_count >= 2,
        "expected CSE in cross-source nested graph, got manifest={manifest:?}"
    );
    let catalog = InputFieldCatalog::new(vec![
        "bar.close".to_string(),
        "quote_tick.bid_price".to_string(),
    ]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close =
                50.0 + 1.7 * base + (ts as f64 * 0.31 + base).sin() * 2.0 + (ts % 5) as f64 * 0.03;
            let bid = 20.0
                + 0.9 * base
                + (ts as f64 * 0.23 + base * 0.7).cos() * 1.5
                + (ts % 7) as f64 * 0.02;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, 0.0))
                .expect("bar event should succeed");
            engine
                .on_event(&quote_event(ts, instrument_slot, bid))
                .expect("quote event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("bid_price".to_string(), bid),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "__cross_corr3".to_string(),
                op: "ts_corr".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("bid_price".to_string()),
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "cross_corr".to_string(),
                op: "elem_add".to_string(),
                field: None,
                lhs: Some("__cross_corr3".to_string()),
                rhs: Some("0.0".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "cross_corr_z".to_string(),
                op: "cs_zscore".to_string(),
                field: Some("__cross_corr3".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "cross_corr_rank".to_string(),
                op: "cs_rank".to_string(),
                field: Some("__cross_corr3".to_string()),
                lhs: None,
                rhs: None,
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "mean_close3".to_string(),
                op: "ts_mean".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "mean_bid3".to_string(),
                op: "ts_mean".to_string(),
                field: Some("bid_price".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_cross_source_shared_graph_with_force_asof() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![7101_u32, 7102, 7103, 7104];
    let ts_points: Vec<i64> = (1..=20).collect();
    let outputs = vec![
        "corr_cross".to_string(),
        "corr_pass".to_string(),
        "corr_sq".to_string(),
        "reg_cross".to_string(),
        "mix_cross".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "ts_corr(bar.close, quote_tick.bid_price, 3)".to_string(),
            "ts_corr(bar.close, quote_tick.bid_price, 3) + 0.0".to_string(),
            "ts_corr(bar.close, quote_tick.bid_price, 3) * ts_corr(bar.close, quote_tick.bid_price, 3)"
                .to_string(),
            "ts_linear_regression(bar.close, quote_tick.bid_price, 3)".to_string(),
            "(ts_corr(bar.close, quote_tick.bid_price, 3) + ts_mean(bar.close, 3)) / (1 + ts_std(bar.close, 3))".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert!(
        manifest.cse_hit_count >= 3,
        "expected shared-node CSE in cross-source graph, got manifest={manifest:?}"
    );
    let catalog = InputFieldCatalog::new(vec![
        "bar.close".to_string(),
        "quote_tick.bid_price".to_string(),
    ]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::ForceWithLast,
        )
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let close = 60.0 + 1.6 * base + ts as f64 * 0.29 + (inst_idx % 2) as f64 * 0.03;
            let bid = 30.0 + 1.1 * base + ts as f64 * 0.21 + (inst_idx % 3) as f64 * 0.02;
            engine
                .on_event(&bar_event(ts, instrument_slot, close, 0.0))
                .expect("bar event should succeed");
            engine
                .on_event(&quote_event(ts, instrument_slot, bid))
                .expect("quote event should succeed");

            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("close".to_string(), close),
                    ("bid_price".to_string(), bid),
                ]),
            });
        }
        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            LegacyOfflineExpr {
                output: "__corr3".to_string(),
                op: "ts_corr".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("bid_price".to_string()),
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__mean_close3".to_string(),
                op: "ts_mean".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__std_close3".to_string(),
                op: "ts_std".to_string(),
                field: Some("close".to_string()),
                lhs: None,
                rhs: None,
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "corr_cross".to_string(),
                op: "elem_add".to_string(),
                field: None,
                lhs: Some("__corr3".to_string()),
                rhs: Some("0.0".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "corr_pass".to_string(),
                op: "elem_add".to_string(),
                field: None,
                lhs: Some("__corr3".to_string()),
                rhs: Some("0.0".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "corr_sq".to_string(),
                op: "elem_mul".to_string(),
                field: None,
                lhs: Some("__corr3".to_string()),
                rhs: Some("__corr3".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "reg_cross".to_string(),
                op: "ts_linear_regression".to_string(),
                field: None,
                lhs: Some("close".to_string()),
                rhs: Some("bid_price".to_string()),
                window: Some(3),
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__num".to_string(),
                op: "elem_add".to_string(),
                field: None,
                lhs: Some("__corr3".to_string()),
                rhs: Some("__mean_close3".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "__den".to_string(),
                op: "elem_add".to_string(),
                field: None,
                lhs: Some("1".to_string()),
                rhs: Some("__std_close3".to_string()),
                window: None,
                lag: None,
            },
            LegacyOfflineExpr {
                output: "mix_cross".to_string(),
                op: "elem_div".to_string(),
                field: None,
                lhs: Some("__num".to_string()),
                rhs: Some("__den".to_string()),
                window: None,
                lag: None,
            },
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}
