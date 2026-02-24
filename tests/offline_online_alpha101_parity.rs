use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, Engine, FactorRequest, InputFieldCatalog, OnlineFactorEngine, Planner,
    SimplePlanner,
};
use std::collections::{BTreeMap, HashMap};

mod common;
use common::alpha101::*;

#[test]
fn polars_offline_matches_online_phase_b_boolean_and_scale_ops() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![7101_u32, 7102, 7103];
    let ts_points: Vec<i64> = (1..=8).collect();
    let outputs = vec![
        "where_cmp".to_string(),
        "to_int_cmp".to_string(),
        "signed_pow".to_string(),
        "cs_scale_close".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "elem_where(close < open, volume, close)".to_string(),
            "elem_to_int((close > open) | (open == close))".to_string(),
            "elem_signed_power(close - open, 2)".to_string(),
            "cs_scale(close)".to_string(),
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
            let base = (inst_idx + 1) as f64;
            let open = 50.0 + base * 0.8 + ts as f64 * 0.1;
            let delta = match inst_idx {
                0 => -0.8 + ts as f64 * 0.02,
                1 => 0.6 - ts as f64 * 0.01,
                _ => 0.0,
            };
            let close = open + delta;
            let volume = 100.0 + base * 10.0 + ts as f64 * 1.5;
            engine
                .on_event(&bar_event_ohlcv(
                    ts,
                    instrument_slot,
                    open,
                    open.max(close) + 0.1,
                    open.min(close) - 0.1,
                    close,
                    volume,
                ))
                .expect("bar event should succeed");
            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("open".to_string(), open),
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
                "__lt_open",
                "elem_lt",
                None,
                Some("close"),
                Some("open"),
                None,
                None,
            ),
            expr(
                "where_cmp",
                "elem_where",
                Some("__lt_open"),
                Some("volume"),
                Some("close"),
                None,
                None,
            ),
            expr(
                "__gt_open",
                "elem_gt",
                None,
                Some("close"),
                Some("open"),
                None,
                None,
            ),
            expr(
                "__eq_open",
                "elem_eq",
                None,
                Some("open"),
                Some("close"),
                None,
                None,
            ),
            expr(
                "__gt_or_eq",
                "elem_or",
                None,
                Some("__gt_open"),
                Some("__eq_open"),
                None,
                None,
            ),
            expr(
                "to_int_cmp",
                "elem_to_int",
                Some("__gt_or_eq"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__diff_close_open",
                "elem_sub",
                None,
                Some("close"),
                Some("open"),
                None,
                None,
            ),
            expr(
                "signed_pow",
                "elem_signed_power",
                None,
                Some("__diff_close_open"),
                Some("2"),
                None,
                None,
            ),
            expr(
                "cs_scale_close",
                "cs_scale",
                Some("close"),
                None,
                None,
                None,
                None,
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
fn polars_offline_matches_online_alpha101_conditional_comparison_subset() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![8101_u32, 8102, 8103, 8104];
    let ts_points: Vec<i64> = (1..=260).collect();
    let outputs = vec![
        "alpha_cond_021".to_string(),
        "alpha_cond_009".to_string(),
        "alpha_cond_023".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "elem_where(ts_mean(close, 8) + ts_std(close, 8) < ts_mean(close, 2), -1, elem_where(ts_mean(close, 2) < ts_mean(close, 8) - ts_std(close, 8), 1, elem_where((1 < volume / ts_mean(volume, 20)) | (volume / ts_mean(volume, 20) == 1), 1, -1)))".to_string(),
            "elem_where(0 < ts_min(ts_delta(close, 1), 5), ts_delta(close, 1), elem_where(ts_max(ts_delta(close, 1), 5) < 0, ts_delta(close, 1), -1 * ts_delta(close, 1)))".to_string(),
            "elem_where(ts_mean(high, 20) < high, -1, 0)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert!(
        manifest.cse_hit_count >= 6,
        "expect CSE in alpha subset, got manifest={manifest:?}"
    );
    let catalog = InputFieldCatalog::new(vec![
        "bar.close".to_string(),
        "bar.volume".to_string(),
        "bar.high".to_string(),
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
            let close = 70.0
                + 1.2 * base
                + ts as f64 * (0.13 + 0.01 * base)
                + (ts as f64 * (0.07 + 0.02 * base)).sin() * 0.9;
            let open = close + (ts as f64 * (0.11 + 0.01 * base)).cos() * 0.25;
            let high = open.max(close) + 0.2;
            let low = open.min(close) - 0.2;
            let volume = 900.0
                + 35.0 * base
                + ts as f64 * (1.7 + 0.1 * base)
                + (ts as f64 * (0.05 + 0.01 * base)).cos() * 45.0;
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
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                    ("high".to_string(), high),
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
                "__a21_m8",
                "ts_mean",
                Some("close"),
                None,
                None,
                Some(8),
                None,
            ),
            expr(
                "__a21_s8",
                "ts_std",
                Some("close"),
                None,
                None,
                Some(8),
                None,
            ),
            expr(
                "__a21_m2",
                "ts_mean",
                Some("close"),
                None,
                None,
                Some(2),
                None,
            ),
            expr(
                "__a21_lhs",
                "elem_add",
                None,
                Some("__a21_m8"),
                Some("__a21_s8"),
                None,
                None,
            ),
            expr(
                "__a21_cond_a",
                "elem_lt",
                None,
                Some("__a21_lhs"),
                Some("__a21_m2"),
                None,
                None,
            ),
            expr(
                "__a21_rhs",
                "elem_sub",
                None,
                Some("__a21_m8"),
                Some("__a21_s8"),
                None,
                None,
            ),
            expr(
                "__a21_cond_b",
                "elem_lt",
                None,
                Some("__a21_m2"),
                Some("__a21_rhs"),
                None,
                None,
            ),
            expr(
                "__a21_adv20",
                "ts_mean",
                Some("volume"),
                None,
                None,
                Some(20),
                None,
            ),
            expr(
                "__a21_vratio",
                "elem_div",
                None,
                Some("volume"),
                Some("__a21_adv20"),
                None,
                None,
            ),
            expr(
                "__a21_cond_c1",
                "elem_lt",
                None,
                Some("1"),
                Some("__a21_vratio"),
                None,
                None,
            ),
            expr(
                "__a21_cond_c2",
                "elem_eq",
                None,
                Some("__a21_vratio"),
                Some("1"),
                None,
                None,
            ),
            expr(
                "__a21_cond_c",
                "elem_or",
                None,
                Some("__a21_cond_c1"),
                Some("__a21_cond_c2"),
                None,
                None,
            ),
            expr(
                "__a21_inner_c",
                "elem_where",
                Some("__a21_cond_c"),
                Some("1"),
                Some("-1"),
                None,
                None,
            ),
            expr(
                "__a21_inner_b",
                "elem_where",
                Some("__a21_cond_b"),
                Some("1"),
                Some("__a21_inner_c"),
                None,
                None,
            ),
            expr(
                "alpha_cond_021",
                "elem_where",
                Some("__a21_cond_a"),
                Some("-1"),
                Some("__a21_inner_b"),
                None,
                None,
            ),
            expr(
                "__a9_d1",
                "ts_delta",
                Some("close"),
                None,
                None,
                None,
                Some(1),
            ),
            expr(
                "__a9_min_d1_5",
                "ts_min",
                Some("__a9_d1"),
                None,
                None,
                Some(5),
                None,
            ),
            expr(
                "__a9_max_d1_5",
                "ts_max",
                Some("__a9_d1"),
                None,
                None,
                Some(5),
                None,
            ),
            expr(
                "__a9_cond_min",
                "elem_lt",
                None,
                Some("0"),
                Some("__a9_min_d1_5"),
                None,
                None,
            ),
            expr(
                "__a9_cond_max",
                "elem_lt",
                None,
                Some("__a9_max_d1_5"),
                Some("0"),
                None,
                None,
            ),
            expr(
                "__a9_neg_d1",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a9_d1"),
                None,
                None,
            ),
            expr(
                "__a9_inner",
                "elem_where",
                Some("__a9_cond_max"),
                Some("__a9_d1"),
                Some("__a9_neg_d1"),
                None,
                None,
            ),
            expr(
                "alpha_cond_009",
                "elem_where",
                Some("__a9_cond_min"),
                Some("__a9_d1"),
                Some("__a9_inner"),
                None,
                None,
            ),
            expr(
                "__a23_m20",
                "ts_mean",
                Some("high"),
                None,
                None,
                Some(20),
                None,
            ),
            expr(
                "__a23_cond",
                "elem_lt",
                None,
                Some("__a23_m20"),
                Some("high"),
                None,
                None,
            ),
            expr(
                "alpha_cond_023",
                "elem_where",
                Some("__a23_cond"),
                Some("-1"),
                Some("0"),
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
fn polars_offline_matches_online_alpha101_conditional_comparison_subset_b() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9101_u32, 9102, 9103, 9104];
    let ts_points: Vec<i64> = (1..=320).collect();
    let outputs = vec![
        "alpha_cond_010".to_string(),
        "alpha_cond_061_like".to_string(),
        "alpha_cond_075_like".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "cs_rank(elem_where(0 < ts_min(ts_delta(close, 1), 4), ts_delta(close, 1), elem_where(ts_max(ts_delta(close, 1), 4) < 0, ts_delta(close, 1), -1 * ts_delta(close, 1))))".to_string(),
            "elem_lt(cs_rank(close - ts_min(close, 16)), cs_rank(ts_delta(close, 5)))"
                .to_string(),
            "elem_lt(cs_rank(ts_mean(volume, 4)), cs_rank(ts_mean(volume, 12)))"
                .to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert!(
        manifest.cse_hit_count >= 4,
        "expect CSE in alpha subset-b, got manifest={manifest:?}"
    );
    let catalog = InputFieldCatalog::new(vec![
        "bar.close".to_string(),
        "bar.low".to_string(),
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
            let close = 90.0
                + 1.5 * base
                + ts as f64 * (0.09 + 0.01 * base)
                + (ts as f64 * (0.03 + 0.01 * base)).sin() * 1.2;
            let low = close - 0.35 - (ts as f64 * (0.02 + 0.005 * base)).abs().sin() * 0.15;
            let open = close + (ts as f64 * (0.08 + 0.01 * base)).cos() * 0.18;
            let high = open.max(close) + 0.22;
            let volume = 1500.0
                + 50.0 * base
                + ts as f64 * (2.1 + 0.2 * base)
                + (ts as f64 * (0.04 + 0.01 * base)).cos() * 60.0;
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
                    ("close".to_string(), close),
                    ("low".to_string(), low),
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
                "__a10_d1",
                "ts_delta",
                Some("close"),
                None,
                None,
                None,
                Some(1),
            ),
            expr(
                "__a10_min_d1_4",
                "ts_min",
                Some("__a10_d1"),
                None,
                None,
                Some(4),
                None,
            ),
            expr(
                "__a10_max_d1_4",
                "ts_max",
                Some("__a10_d1"),
                None,
                None,
                Some(4),
                None,
            ),
            expr(
                "__a10_cond_min",
                "elem_lt",
                None,
                Some("0"),
                Some("__a10_min_d1_4"),
                None,
                None,
            ),
            expr(
                "__a10_cond_max",
                "elem_lt",
                None,
                Some("__a10_max_d1_4"),
                Some("0"),
                None,
                None,
            ),
            expr(
                "__a10_neg_d1",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a10_d1"),
                None,
                None,
            ),
            expr(
                "__a10_inner",
                "elem_where",
                Some("__a10_cond_max"),
                Some("__a10_d1"),
                Some("__a10_neg_d1"),
                None,
                None,
            ),
            expr(
                "__a10_raw",
                "elem_where",
                Some("__a10_cond_min"),
                Some("__a10_d1"),
                Some("__a10_inner"),
                None,
                None,
            ),
            expr(
                "alpha_cond_010",
                "cs_rank",
                Some("__a10_raw"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a61_min_close16",
                "ts_min",
                Some("close"),
                None,
                None,
                Some(16),
                None,
            ),
            expr(
                "__a61_lhs_raw",
                "elem_sub",
                None,
                Some("close"),
                Some("__a61_min_close16"),
                None,
                None,
            ),
            expr(
                "__a61_lhs_rank",
                "cs_rank",
                Some("__a61_lhs_raw"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a61_rhs_raw",
                "ts_delta",
                Some("close"),
                None,
                None,
                None,
                Some(5),
            ),
            expr(
                "__a61_rhs_rank",
                "cs_rank",
                Some("__a61_rhs_raw"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "alpha_cond_061_like",
                "elem_lt",
                None,
                Some("__a61_lhs_rank"),
                Some("__a61_rhs_rank"),
                None,
                None,
            ),
            expr(
                "__a75_lhs_ts",
                "ts_mean",
                Some("volume"),
                None,
                None,
                Some(4),
                None,
            ),
            expr(
                "__a75_lhs_rank",
                "cs_rank",
                Some("__a75_lhs_ts"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a75_rhs_ts",
                "ts_mean",
                Some("volume"),
                None,
                None,
                Some(12),
                None,
            ),
            expr(
                "__a75_rhs_rank",
                "cs_rank",
                Some("__a75_rhs_ts"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "alpha_cond_075_like",
                "elem_lt",
                None,
                Some("__a75_lhs_rank"),
                Some("__a75_rhs_rank"),
                None,
                None,
            ),
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);
    let streaming = retain_from_ts(&streaming, 220);
    let offline = retain_from_ts(&offline, 220);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}
