use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, Engine, FactorRequest, InputFieldCatalog, OnlineFactorEngine, Planner,
    SimplePlanner,
};
use std::collections::{BTreeMap, HashMap};

mod common;
use common::alpha101::*;

#[test]
fn polars_offline_matches_online_alpha101_alpha018_with_intermediate_nodes() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9401_u32, 9402, 9403, 9404, 9405];
    let ts_points: Vec<i64> = (1..=280).collect();
    let outputs = vec![
        "a018_diff".to_string(),
        "a018_abs_diff".to_string(),
        "a018_std5".to_string(),
        "a018_corr10".to_string(),
        "a018_sum".to_string(),
        "a018_rank".to_string(),
        "alpha018".to_string(),
    ];
    let sum_expr = "ts_std(elem_abs(close - open), 5) + (close - open) + ts_corr(close, open, 10)";
    let request = FactorRequest {
        exprs: vec![
            "close - open".to_string(),
            "elem_abs(close - open)".to_string(),
            "ts_std(elem_abs(close - open), 5)".to_string(),
            "ts_corr(close, open, 10)".to_string(),
            sum_expr.to_string(),
            format!("cs_rank({sum_expr})"),
            format!("-1 * cs_rank({sum_expr})"),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.expr_count, outputs.len(), "{manifest:?}");
    assert!(manifest.cse_hit_count >= 6, "{manifest:?}");

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
            let base = (inst_idx + 1) as f64;
            let open = 120.0
                + 3.0 * base
                + ts as f64 * (0.10 + 0.012 * base)
                + (ts as f64 * (0.017 + 0.002 * base)).sin() * (0.4 + 0.07 * base);
            let close = open
                + base * 0.06
                + (ts as f64 * (0.028 + 0.003 * base)).cos() * (0.25 + 0.09 * base);
            let high = open.max(close) + 0.35 + base * 0.02;
            let low = open.min(close) - 0.33 - base * 0.02;
            let volume = 1800.0 + 50.0 * base + ts as f64 * (2.0 + 0.2 * base);

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

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            expr(
                "a018_diff",
                "elem_sub",
                None,
                Some("close"),
                Some("open"),
                None,
                None,
            ),
            expr(
                "a018_abs_diff",
                "elem_abs",
                Some("a018_diff"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "a018_std5",
                "ts_std",
                Some("a018_abs_diff"),
                None,
                None,
                Some(5),
                None,
            ),
            expr(
                "a018_corr10",
                "ts_corr",
                None,
                Some("close"),
                Some("open"),
                Some(10),
                None,
            ),
            expr(
                "a018_sum1",
                "elem_add",
                None,
                Some("a018_std5"),
                Some("a018_diff"),
                None,
                None,
            ),
            expr(
                "a018_sum",
                "elem_add",
                None,
                Some("a018_sum1"),
                Some("a018_corr10"),
                None,
                None,
            ),
            expr(
                "a018_rank",
                "cs_rank",
                Some("a018_sum"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "alpha018",
                "elem_mul",
                None,
                Some("-1"),
                Some("a018_rank"),
                None,
                None,
            ),
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);
    let streaming = retain_from_ts(&streaming, 40);
    let offline = retain_from_ts(&offline, 40);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_alpha101_alpha046_with_intermediate_nodes() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9501_u32, 9502, 9503, 9504, 9505];
    let ts_points: Vec<i64> = (1..=260).collect();
    let outputs = vec![
        "a046_lag20".to_string(),
        "a046_lag10".to_string(),
        "a046_lag1".to_string(),
        "a046_diff1".to_string(),
        "a046_diff2".to_string(),
        "a046_term1".to_string(),
        "a046_term2".to_string(),
        "a046_metric".to_string(),
        "a046_cond_hi".to_string(),
        "a046_cond_lo".to_string(),
        "a046_neg_close_lag1".to_string(),
        "a046_inner".to_string(),
        "alpha046".to_string(),
    ];

    let metric_expr =
        "(ts_lag(close, 20) - ts_lag(close, 10)) / 10 - (ts_lag(close, 10) - close) / 10";
    let request = FactorRequest {
        exprs: vec![
            "ts_lag(close, 20)".to_string(),
            "ts_lag(close, 10)".to_string(),
            "ts_lag(close, 1)".to_string(),
            "ts_lag(close, 20) - ts_lag(close, 10)".to_string(),
            "ts_lag(close, 10) - close".to_string(),
            "(ts_lag(close, 20) - ts_lag(close, 10)) / 10".to_string(),
            "(ts_lag(close, 10) - close) / 10".to_string(),
            metric_expr.to_string(),
            format!("0.25 < {metric_expr}"),
            format!("{metric_expr} < 0"),
            "-1 * (close - ts_lag(close, 1))".to_string(),
            format!("elem_where({metric_expr} < 0, 1, -1 * (close - ts_lag(close, 1)))"),
            format!(
                "elem_where(0.25 < {metric_expr}, -1, elem_where({metric_expr} < 0, 1, -1 * (close - ts_lag(close, 1))))"
            ),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.expr_count, outputs.len(), "{manifest:?}");
    assert!(manifest.cse_hit_count >= 12, "{manifest:?}");

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

    let accel = [0.020_f64, -0.010, 0.005, 0.015, -0.004];
    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let t = ts as f64;
            let close = 80.0
                + 4.0 * base
                + (0.35 + 0.05 * base) * t
                + accel[inst_idx] * t * t
                + (t * (0.015 + 0.003 * base)).sin() * 0.03;
            let open = close - 0.12;
            let high = close + 0.24;
            let low = close - 0.28;
            let volume = 900.0 + 40.0 * base + t * (1.2 + 0.2 * base);

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

    let offline_all = run_polars_offline(&LegacyOfflinePayload {
        rows: offline_rows,
        expressions: vec![
            expr(
                "a046_lag20",
                "ts_lag",
                Some("close"),
                None,
                None,
                None,
                Some(20),
            ),
            expr(
                "a046_lag10",
                "ts_lag",
                Some("close"),
                None,
                None,
                None,
                Some(10),
            ),
            expr(
                "a046_lag1",
                "ts_lag",
                Some("close"),
                None,
                None,
                None,
                Some(1),
            ),
            expr(
                "a046_diff1",
                "elem_sub",
                None,
                Some("a046_lag20"),
                Some("a046_lag10"),
                None,
                None,
            ),
            expr(
                "a046_diff2",
                "elem_sub",
                None,
                Some("a046_lag10"),
                Some("close"),
                None,
                None,
            ),
            expr(
                "a046_term1",
                "elem_div",
                None,
                Some("a046_diff1"),
                Some("10"),
                None,
                None,
            ),
            expr(
                "a046_term2",
                "elem_div",
                None,
                Some("a046_diff2"),
                Some("10"),
                None,
                None,
            ),
            expr(
                "a046_metric",
                "elem_sub",
                None,
                Some("a046_term1"),
                Some("a046_term2"),
                None,
                None,
            ),
            expr(
                "a046_cond_hi",
                "elem_lt",
                None,
                Some("0.25"),
                Some("a046_metric"),
                None,
                None,
            ),
            expr(
                "a046_cond_lo",
                "elem_lt",
                None,
                Some("a046_metric"),
                Some("0"),
                None,
                None,
            ),
            expr(
                "a046_close_lag1",
                "elem_sub",
                None,
                Some("close"),
                Some("a046_lag1"),
                None,
                None,
            ),
            expr(
                "a046_neg_close_lag1",
                "elem_mul",
                None,
                Some("-1"),
                Some("a046_close_lag1"),
                None,
                None,
            ),
            expr(
                "a046_inner",
                "elem_where",
                Some("a046_cond_lo"),
                Some("1"),
                Some("a046_neg_close_lag1"),
                None,
                None,
            ),
            expr(
                "alpha046",
                "elem_where",
                Some("a046_cond_hi"),
                Some("-1"),
                Some("a046_inner"),
                None,
                None,
            ),
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);
    let streaming = retain_from_ts(&streaming, 40);
    let offline = retain_from_ts(&offline, 40);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}
