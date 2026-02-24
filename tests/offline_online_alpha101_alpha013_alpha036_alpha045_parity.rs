use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, Engine, FactorRequest, InputFieldCatalog, OnlineFactorEngine, Planner,
    SimplePlanner,
};
use std::collections::{BTreeMap, HashMap};

mod common;
use common::alpha101::*;

fn bar_catalog() -> InputFieldCatalog {
    InputFieldCatalog::new(vec![
        "bar.open".to_string(),
        "bar.high".to_string(),
        "bar.low".to_string(),
        "bar.close".to_string(),
        "bar.volume".to_string(),
    ])
}

#[test]
fn polars_offline_matches_online_alpha101_alpha013_with_intermediate_nodes() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9601_u32, 9602, 9603, 9604, 9605, 9606, 9607, 9608, 9609];
    let ts_points: Vec<i64> = (1..=360).collect();
    let outputs = vec![
        "a013_close_rank".to_string(),
        "a013_volume_rank".to_string(),
        "a013_cov5".to_string(),
        "a013_cov_rank".to_string(),
        "alpha013".to_string(),
    ];
    let expr_cov = "ts_cov(cs_rank(close), cs_rank(volume), 5)";
    let request = FactorRequest {
        exprs: vec![
            "cs_rank(close)".to_string(),
            "cs_rank(volume)".to_string(),
            expr_cov.to_string(),
            format!("cs_rank({expr_cov})"),
            format!("-1 * cs_rank({expr_cov})"),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.expr_count, outputs.len(), "{manifest:?}");
    assert!(manifest.cse_hit_count >= 3, "{manifest:?}");

    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &bar_catalog(),
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
            let t = ts as f64;
            let open = 1000.0 + 10.0 * base + 2.0 * t;
            let close = open + 1.0;
            let high = open.max(close) + 0.22 + 0.014 * base;
            let low = open.min(close) - 0.20 - 0.013 * base;
            let volume = 2000.0 + 5.0 * base + 3.0 * t;

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
                "a013_close_rank",
                "cs_rank",
                Some("close"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "a013_volume_rank",
                "cs_rank",
                Some("volume"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "a013_cov5",
                "ts_cov",
                None,
                Some("a013_close_rank"),
                Some("a013_volume_rank"),
                Some(5),
                None,
            ),
            expr(
                "a013_cov_rank",
                "cs_rank",
                Some("a013_cov5"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "alpha013",
                "elem_mul",
                None,
                Some("-1"),
                Some("a013_cov_rank"),
                None,
                None,
            ),
        ],
    });
    let online = retain_from_ts(&retain_outputs(&streaming, &outputs), 120);
    let offline = retain_from_ts(&retain_outputs(&offline_all, &outputs), 120);
    assert_eq!(online.len(), offline.len(), "row count mismatch");
    for (key, lhs) in &online {
        let rhs = offline
            .get(key)
            .unwrap_or_else(|| panic!("missing key in offline baseline: {:?}", key));
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_alpha101_alpha036_with_intermediate_nodes() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9701_u32, 9702, 9703, 9704, 9705, 9706, 9707, 9708];
    let ts_points: Vec<i64> = (1..=560).collect();
    let outputs = vec![
        "a036_returns".to_string(),
        "a036_neg_returns".to_string(),
        "a036_lag_neg_ret6".to_string(),
        "a036_vwap".to_string(),
        "a036_adv20".to_string(),
        "a036_close_open".to_string(),
        "a036_open_close".to_string(),
        "a036_corr15".to_string(),
        "a036_rank_corr15".to_string(),
        "a036_term1".to_string(),
        "a036_rank_open_close".to_string(),
        "a036_term2".to_string(),
        "a036_rank_ts_rank".to_string(),
        "a036_term3".to_string(),
        "a036_rank_abs_corr6".to_string(),
        "a036_term4".to_string(),
        "a036_rank_struct".to_string(),
        "a036_term5".to_string(),
        "alpha036".to_string(),
    ];

    let request = FactorRequest {
        exprs: vec![
            "ts_delta(close, 1) / (ts_lag(close, 1) + 0.000001)".to_string(),
            "-1 * (ts_delta(close, 1) / (ts_lag(close, 1) + 0.000001))".to_string(),
            "ts_lag(-1 * (ts_delta(close, 1) / (ts_lag(close, 1) + 0.000001)), 6)".to_string(),
            "(open + high + low + close) / 4".to_string(),
            "ts_mean(volume, 20)".to_string(),
            "close - open".to_string(),
            "open - close".to_string(),
            "ts_corr(close - open, ts_lag(volume, 1), 15)".to_string(),
            "cs_rank(ts_corr(close - open, ts_lag(volume, 1), 15))".to_string(),
            "2.21 * cs_rank(ts_corr(close - open, ts_lag(volume, 1), 15))".to_string(),
            "cs_rank(open - close)".to_string(),
            "0.7 * cs_rank(open - close)".to_string(),
            "cs_rank(ts_rank(ts_lag(-1 * (ts_delta(close, 1) / (ts_lag(close, 1) + 0.000001)), 6), 5))"
                .to_string(),
            "0.73 * cs_rank(ts_rank(ts_lag(-1 * (ts_delta(close, 1) / (ts_lag(close, 1) + 0.000001)), 6), 5))"
                .to_string(),
            "cs_rank(elem_abs(ts_corr((open + high + low + close) / 4, ts_mean(volume, 20), 6)))"
                .to_string(),
            "1 * cs_rank(elem_abs(ts_corr((open + high + low + close) / 4, ts_mean(volume, 20), 6)))"
                .to_string(),
            "cs_rank((ts_sum(close, 200) / 200 - open) * (close - open))".to_string(),
            "0.6 * cs_rank((ts_sum(close, 200) / 200 - open) * (close - open))".to_string(),
            "2.21 * cs_rank(ts_corr(close - open, ts_lag(volume, 1), 15)) + 0.7 * cs_rank(open - close) + 0.73 * cs_rank(ts_rank(ts_lag(-1 * (ts_delta(close, 1) / (ts_lag(close, 1) + 0.000001)), 6), 5)) + cs_rank(elem_abs(ts_corr((open + high + low + close) / 4, ts_mean(volume, 20), 6))) + 0.6 * cs_rank((ts_sum(close, 200) / 200 - open) * (close - open))".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.expr_count, outputs.len(), "{manifest:?}");
    assert!(manifest.cse_hit_count >= 12, "{manifest:?}");

    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &bar_catalog(),
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
            let t = ts as f64;
            let open = 95.0
                + 3.3 * base
                + t * (0.065 + 0.007 * base)
                + (t * (0.010 + 0.0014 * base) + base * 0.18).sin() * (0.95 + 0.05 * base);
            let close = open
                + 0.09 * base
                + (t * (0.016 + 0.0017 * base) + base * 0.24).cos() * (0.76 + 0.06 * base)
                + t * t * (0.000008 * base);
            let high = open.max(close) + 0.29 + base * 0.015;
            let low = open.min(close) - 0.27 - base * 0.013;
            let volume = 2100.0
                + 120.0 * base
                + t * (2.9 + 0.19 * base)
                + (t * (0.012 + 0.0011 * base) + base * 0.22).cos() * 85.0
                + (t * 0.007 * base).sin() * 11.0;

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

    let offline_all =
        run_polars_offline(&build_legacy_payload_from_logical(offline_rows, &logical));
    let online = retain_from_ts(&retain_outputs(&streaming, &outputs), 280);
    let offline = retain_from_ts(&retain_outputs(&offline_all, &outputs), 280);
    assert_eq!(online.len(), offline.len(), "row count mismatch");
    for (key, lhs) in &online {
        let rhs = offline
            .get(key)
            .unwrap_or_else(|| panic!("missing key in offline baseline: {:?}", key));
        assert_close(*lhs, *rhs, key);
    }
}

#[test]
fn polars_offline_matches_online_alpha101_alpha045_with_intermediate_nodes() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9801_u32, 9802, 9803, 9804, 9805, 9806, 9807];
    let ts_points: Vec<i64> = (1..=440).collect();
    let outputs = vec![
        "a045_close_lag5".to_string(),
        "a045_sum_lag5_20".to_string(),
        "a045_mean_lag5_20".to_string(),
        "a045_rank_left".to_string(),
        "a045_corr_close_volume_2".to_string(),
        "a045_sum_close_5".to_string(),
        "a045_sum_close_20".to_string(),
        "a045_corr_sum_2".to_string(),
        "a045_rank_right".to_string(),
        "a045_mul_left".to_string(),
        "a045_mul_all".to_string(),
        "alpha045".to_string(),
    ];
    let expr_mul = "cs_rank(ts_sum(ts_lag(close, 5), 20) / 20) * ts_corr(close, volume, 2) * cs_rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2))";
    let request = FactorRequest {
        exprs: vec![
            "ts_lag(close, 5)".to_string(),
            "ts_sum(ts_lag(close, 5), 20)".to_string(),
            "ts_sum(ts_lag(close, 5), 20) / 20".to_string(),
            "cs_rank(ts_sum(ts_lag(close, 5), 20) / 20)".to_string(),
            "ts_corr(close, volume, 2)".to_string(),
            "ts_sum(close, 5)".to_string(),
            "ts_sum(close, 20)".to_string(),
            "ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2)".to_string(),
            "cs_rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2))".to_string(),
            "cs_rank(ts_sum(ts_lag(close, 5), 20) / 20) * ts_corr(close, volume, 2)".to_string(),
            expr_mul.to_string(),
            format!("-1 * ({expr_mul})"),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.expr_count, outputs.len(), "{manifest:?}");
    assert!(manifest.cse_hit_count >= 10, "{manifest:?}");

    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &bar_catalog(),
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
            let t = ts as f64;
            let open = 800.0 + 7.0 * base + 3.0 * t;
            let close = open + 2.0;
            let high = open.max(close) + 0.25 + base * 0.012;
            let low = open.min(close) - 0.24 - base * 0.011;
            let volume = 1500.0 + 9.0 * base + 4.0 * t;

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
                "a045_close_lag5",
                "ts_lag",
                Some("close"),
                None,
                None,
                None,
                Some(5),
            ),
            expr(
                "a045_sum_lag5_20",
                "ts_sum",
                Some("a045_close_lag5"),
                None,
                None,
                Some(20),
                None,
            ),
            expr(
                "a045_mean_lag5_20",
                "elem_div",
                None,
                Some("a045_sum_lag5_20"),
                Some("20"),
                None,
                None,
            ),
            expr(
                "a045_rank_left",
                "cs_rank",
                Some("a045_mean_lag5_20"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "a045_corr_close_volume_2",
                "ts_corr",
                None,
                Some("close"),
                Some("volume"),
                Some(2),
                None,
            ),
            expr(
                "a045_sum_close_5",
                "ts_sum",
                Some("close"),
                None,
                None,
                Some(5),
                None,
            ),
            expr(
                "a045_sum_close_20",
                "ts_sum",
                Some("close"),
                None,
                None,
                Some(20),
                None,
            ),
            expr(
                "a045_corr_sum_2",
                "ts_corr",
                None,
                Some("a045_sum_close_5"),
                Some("a045_sum_close_20"),
                Some(2),
                None,
            ),
            expr(
                "a045_rank_right",
                "cs_rank",
                Some("a045_corr_sum_2"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "a045_mul_left",
                "elem_mul",
                None,
                Some("a045_rank_left"),
                Some("a045_corr_close_volume_2"),
                None,
                None,
            ),
            expr(
                "a045_mul_all",
                "elem_mul",
                None,
                Some("a045_mul_left"),
                Some("a045_rank_right"),
                None,
                None,
            ),
            expr(
                "alpha045",
                "elem_mul",
                None,
                Some("-1"),
                Some("a045_mul_all"),
                None,
                None,
            ),
        ],
    });
    let online = retain_from_ts(&retain_outputs(&streaming, &outputs), 320);
    let offline = retain_from_ts(&retain_outputs(&offline_all, &outputs), 320);
    assert_eq!(online.len(), offline.len(), "row count mismatch");
    for (key, lhs) in &online {
        let rhs = offline
            .get(key)
            .unwrap_or_else(|| panic!("missing key in offline baseline: {:?}", key));
        assert_close(*lhs, *rhs, key);
    }
}
