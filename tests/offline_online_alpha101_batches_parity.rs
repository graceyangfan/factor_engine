use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, Engine, FactorRequest, InputFieldCatalog, OnlineFactorEngine, Planner,
    SimplePlanner,
};
use std::collections::{BTreeMap, HashMap};

mod common;
use common::alpha101::*;

#[test]
fn polars_offline_matches_online_alpha101_real_batch_c() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9201_u32, 9202, 9203, 9204, 9205];
    let ts_points: Vec<i64> = (1..=180).collect();
    let outputs = vec![
        "alpha003".to_string(),
        "alpha004".to_string(),
        "alpha006".to_string(),
        "alpha012".to_string(),
        "alpha020".to_string(),
        "alpha038".to_string(),
        "alpha040".to_string(),
        "alpha044".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "-1 * ts_corr(cs_rank(open), cs_rank(volume), 10)".to_string(),
            "-1 * ts_rank(cs_rank(low), 9)".to_string(),
            "-1 * ts_corr(open, volume, 10)".to_string(),
            "elem_sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1))".to_string(),
            "-1 * cs_rank(open - ts_lag(high, 1)) * cs_rank(open - ts_lag(close, 1)) * cs_rank(open - ts_lag(low, 1))".to_string(),
            "-1 * cs_rank(ts_rank(close, 10)) * cs_rank(close / open)".to_string(),
            "-1 * cs_rank(ts_std(high, 10)) * ts_corr(high, volume, 10)".to_string(),
            "-1 * ts_corr(high, cs_rank(volume), 5)".to_string(),
        ],
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.expr_count, outputs.len(), "{manifest:?}");
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
            let open = 110.0
                + 1.1 * base
                + ts as f64 * (0.11 + 0.01 * base)
                + (ts as f64 * (0.03 + 0.004 * base)).sin() * 0.9;
            let close = open + (ts as f64 * (0.05 + 0.007 * base)).cos() * 0.35;
            let high = open.max(close) + 0.25 + base * 0.01;
            let low = open.min(close) - 0.22 - base * 0.01;
            let volume = 2500.0
                + 80.0 * base
                + ts as f64 * (3.2 + 0.2 * base)
                + (ts as f64 * ts as f64) * 0.0005;
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
                "__a003_r_open",
                "cs_rank",
                Some("open"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a003_r_volume",
                "cs_rank",
                Some("volume"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a003_corr",
                "ts_corr",
                None,
                Some("__a003_r_open"),
                Some("__a003_r_volume"),
                Some(10),
                None,
            ),
            expr(
                "alpha003",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a003_corr"),
                None,
                None,
            ),
            expr(
                "__a004_r_low",
                "cs_rank",
                Some("low"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a004_ts_rank",
                "ts_rank",
                Some("__a004_r_low"),
                None,
                None,
                Some(9),
                None,
            ),
            expr(
                "alpha004",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a004_ts_rank"),
                None,
                None,
            ),
            expr(
                "__a006_corr",
                "ts_corr",
                None,
                Some("open"),
                Some("volume"),
                Some(10),
                None,
            ),
            expr(
                "alpha006",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a006_corr"),
                None,
                None,
            ),
            expr(
                "__a012_dv1",
                "ts_delta",
                Some("volume"),
                None,
                None,
                None,
                Some(1),
            ),
            expr(
                "__a012_sign_dv1",
                "elem_sign",
                Some("__a012_dv1"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a012_dc1",
                "ts_delta",
                Some("close"),
                None,
                None,
                None,
                Some(1),
            ),
            expr(
                "__a012_neg_dc1",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a012_dc1"),
                None,
                None,
            ),
            expr(
                "alpha012",
                "elem_mul",
                None,
                Some("__a012_sign_dv1"),
                Some("__a012_neg_dc1"),
                None,
                None,
            ),
            expr(
                "__a020_lag_h1",
                "ts_lag",
                Some("high"),
                None,
                None,
                None,
                Some(1),
            ),
            expr(
                "__a020_lag_c1",
                "ts_lag",
                Some("close"),
                None,
                None,
                None,
                Some(1),
            ),
            expr(
                "__a020_lag_l1",
                "ts_lag",
                Some("low"),
                None,
                None,
                None,
                Some(1),
            ),
            expr(
                "__a020_dh",
                "elem_sub",
                None,
                Some("open"),
                Some("__a020_lag_h1"),
                None,
                None,
            ),
            expr(
                "__a020_dc",
                "elem_sub",
                None,
                Some("open"),
                Some("__a020_lag_c1"),
                None,
                None,
            ),
            expr(
                "__a020_dl",
                "elem_sub",
                None,
                Some("open"),
                Some("__a020_lag_l1"),
                None,
                None,
            ),
            expr(
                "__a020_rh",
                "cs_rank",
                Some("__a020_dh"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a020_rc",
                "cs_rank",
                Some("__a020_dc"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a020_rl",
                "cs_rank",
                Some("__a020_dl"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a020_mul2",
                "elem_mul",
                None,
                Some("__a020_rh"),
                Some("__a020_rc"),
                None,
                None,
            ),
            expr(
                "__a020_mul3",
                "elem_mul",
                None,
                Some("__a020_mul2"),
                Some("__a020_rl"),
                None,
                None,
            ),
            expr(
                "alpha020",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a020_mul3"),
                None,
                None,
            ),
            expr(
                "__a038_rank_close10",
                "ts_rank",
                Some("close"),
                None,
                None,
                Some(10),
                None,
            ),
            expr(
                "__a038_r1",
                "cs_rank",
                Some("__a038_rank_close10"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a038_close_div_open",
                "elem_div",
                None,
                Some("close"),
                Some("open"),
                None,
                None,
            ),
            expr(
                "__a038_r2",
                "cs_rank",
                Some("__a038_close_div_open"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a038_mul",
                "elem_mul",
                None,
                Some("__a038_r1"),
                Some("__a038_r2"),
                None,
                None,
            ),
            expr(
                "alpha038",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a038_mul"),
                None,
                None,
            ),
            expr(
                "__a040_std_h10",
                "ts_std",
                Some("high"),
                None,
                None,
                Some(10),
                None,
            ),
            expr(
                "__a040_r",
                "cs_rank",
                Some("__a040_std_h10"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a040_corr",
                "ts_corr",
                None,
                Some("high"),
                Some("volume"),
                Some(10),
                None,
            ),
            expr(
                "__a040_mul",
                "elem_mul",
                None,
                Some("__a040_r"),
                Some("__a040_corr"),
                None,
                None,
            ),
            expr(
                "alpha040",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a040_mul"),
                None,
                None,
            ),
            expr(
                "__a044_volume_rank",
                "cs_rank",
                Some("volume"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a044_corr",
                "ts_corr",
                None,
                Some("high"),
                Some("__a044_volume_rank"),
                Some(5),
                None,
            ),
            expr(
                "alpha044",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a044_corr"),
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
fn polars_offline_matches_online_alpha101_real_batch_d() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9301_u32, 9302, 9303, 9304, 9305];
    let ts_points: Vec<i64> = (1..=320).collect();
    let outputs = vec![
        "alpha053".to_string(),
        "alpha015".to_string(),
        "alpha016".to_string(),
        "alpha055".to_string(),
        "alpha022".to_string(),
        "alpha026".to_string(),
        "alpha030".to_string(),
        "alpha043".to_string(),
        "alpha054".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "-1 * ts_delta((close - low - (high - close)) / (close - low), 9)".to_string(),
            "-1 * ts_sum(cs_rank(ts_corr(cs_rank(high), cs_rank(volume), 3)), 3)".to_string(),
            "-1 * cs_rank(ts_cov(cs_rank(high), cs_rank(volume), 5))".to_string(),
            "-1 * ts_corr(cs_rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), cs_rank(volume), 6)".to_string(),
            "-1 * ts_delta(ts_corr(high, volume, 5), 5) * cs_rank(ts_std(close, 20))".to_string(),
            "-1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)".to_string(),
            "(1 - cs_rank(elem_sign(close - ts_lag(close, 1)) + elem_sign(ts_lag(close, 1) - ts_lag(close, 2)) + elem_sign(ts_lag(close, 2) - ts_lag(close, 3)))) * ts_sum(volume, 5) / ts_sum(volume, 20)".to_string(),
            "ts_rank(volume / ts_mean(volume, 20), 20) * ts_rank(-1 * ts_delta(close, 7), 8)"
                .to_string(),
            "-1 * ((low - close) * elem_pow(open, 5)) / ((low - high) * elem_pow(close, 5))"
                .to_string(),
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
            let open = 140.0
                + 1.25 * base
                + ts as f64 * (0.08 + 0.01 * base)
                + (ts as f64 * (0.021 + 0.004 * base)).sin() * 1.1;
            let close = open + (ts as f64 * (0.035 + 0.006 * base)).cos() * 0.42;
            let high = open.max(close) + 0.28 + base * 0.015;
            let low = open.min(close) - 0.24 - base * 0.012;
            let volume = 3200.0
                + 90.0 * base
                + ts as f64 * (2.8 + 0.25 * base)
                + (ts as f64 * (0.018 + 0.003 * base)).cos() * 85.0;
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
                "__rank_high",
                "cs_rank",
                Some("high"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__rank_volume",
                "cs_rank",
                Some("volume"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a053_close_low",
                "elem_sub",
                None,
                Some("close"),
                Some("low"),
                None,
                None,
            ),
            expr(
                "__a053_high_close",
                "elem_sub",
                None,
                Some("high"),
                Some("close"),
                None,
                None,
            ),
            expr(
                "__a053_num",
                "elem_sub",
                None,
                Some("__a053_close_low"),
                Some("__a053_high_close"),
                None,
                None,
            ),
            expr(
                "__a053_ratio",
                "elem_div",
                None,
                Some("__a053_num"),
                Some("__a053_close_low"),
                None,
                None,
            ),
            expr(
                "__a053_delta9",
                "ts_delta",
                Some("__a053_ratio"),
                None,
                None,
                None,
                Some(9),
            ),
            expr(
                "alpha053",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a053_delta9"),
                None,
                None,
            ),
            expr(
                "__a015_corr",
                "ts_corr",
                None,
                Some("__rank_high"),
                Some("__rank_volume"),
                Some(3),
                None,
            ),
            expr(
                "__a015_rank_corr",
                "cs_rank",
                Some("__a015_corr"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a015_sum",
                "ts_sum",
                Some("__a015_rank_corr"),
                None,
                None,
                Some(3),
                None,
            ),
            expr(
                "alpha015",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a015_sum"),
                None,
                None,
            ),
            expr(
                "__a016_cov",
                "ts_cov",
                None,
                Some("__rank_high"),
                Some("__rank_volume"),
                Some(5),
                None,
            ),
            expr(
                "__a016_rank_cov",
                "cs_rank",
                Some("__a016_cov"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "alpha016",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a016_rank_cov"),
                None,
                None,
            ),
            expr(
                "__a055_min_low12",
                "ts_min",
                Some("low"),
                None,
                None,
                Some(12),
                None,
            ),
            expr(
                "__a055_max_high12",
                "ts_max",
                Some("high"),
                None,
                None,
                Some(12),
                None,
            ),
            expr(
                "__a055_num",
                "elem_sub",
                None,
                Some("close"),
                Some("__a055_min_low12"),
                None,
                None,
            ),
            expr(
                "__a055_den",
                "elem_sub",
                None,
                Some("__a055_max_high12"),
                Some("__a055_min_low12"),
                None,
                None,
            ),
            expr(
                "__a055_ratio",
                "elem_div",
                None,
                Some("__a055_num"),
                Some("__a055_den"),
                None,
                None,
            ),
            expr(
                "__a055_rank_ratio",
                "cs_rank",
                Some("__a055_ratio"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a055_corr",
                "ts_corr",
                None,
                Some("__a055_rank_ratio"),
                Some("__rank_volume"),
                Some(6),
                None,
            ),
            expr(
                "alpha055",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a055_corr"),
                None,
                None,
            ),
            expr(
                "__a022_corr",
                "ts_corr",
                None,
                Some("high"),
                Some("volume"),
                Some(5),
                None,
            ),
            expr(
                "__a022_dcorr5",
                "ts_delta",
                Some("__a022_corr"),
                None,
                None,
                None,
                Some(5),
            ),
            expr(
                "__a022_std20",
                "ts_std",
                Some("close"),
                None,
                None,
                Some(20),
                None,
            ),
            expr(
                "__a022_rank_std20",
                "cs_rank",
                Some("__a022_std20"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a022_mul",
                "elem_mul",
                None,
                Some("__a022_dcorr5"),
                Some("__a022_rank_std20"),
                None,
                None,
            ),
            expr(
                "alpha022",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a022_mul"),
                None,
                None,
            ),
            expr(
                "__a026_rank_v5",
                "ts_rank",
                Some("volume"),
                None,
                None,
                Some(5),
                None,
            ),
            expr(
                "__a026_rank_h5",
                "ts_rank",
                Some("high"),
                None,
                None,
                Some(5),
                None,
            ),
            expr(
                "__a026_corr",
                "ts_corr",
                None,
                Some("__a026_rank_v5"),
                Some("__a026_rank_h5"),
                Some(5),
                None,
            ),
            expr(
                "__a026_max",
                "ts_max",
                Some("__a026_corr"),
                None,
                None,
                Some(3),
                None,
            ),
            expr(
                "alpha026",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a026_max"),
                None,
                None,
            ),
            expr(
                "__a030_lag1",
                "ts_lag",
                Some("close"),
                None,
                None,
                None,
                Some(1),
            ),
            expr(
                "__a030_lag2",
                "ts_lag",
                Some("close"),
                None,
                None,
                None,
                Some(2),
            ),
            expr(
                "__a030_lag3",
                "ts_lag",
                Some("close"),
                None,
                None,
                None,
                Some(3),
            ),
            expr(
                "__a030_d01",
                "elem_sub",
                None,
                Some("close"),
                Some("__a030_lag1"),
                None,
                None,
            ),
            expr(
                "__a030_d12",
                "elem_sub",
                None,
                Some("__a030_lag1"),
                Some("__a030_lag2"),
                None,
                None,
            ),
            expr(
                "__a030_d23",
                "elem_sub",
                None,
                Some("__a030_lag2"),
                Some("__a030_lag3"),
                None,
                None,
            ),
            expr(
                "__a030_s1",
                "elem_sign",
                Some("__a030_d01"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a030_s2",
                "elem_sign",
                Some("__a030_d12"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a030_s3",
                "elem_sign",
                Some("__a030_d23"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a030_sum12",
                "elem_add",
                None,
                Some("__a030_s1"),
                Some("__a030_s2"),
                None,
                None,
            ),
            expr(
                "__a030_sum123",
                "elem_add",
                None,
                Some("__a030_sum12"),
                Some("__a030_s3"),
                None,
                None,
            ),
            expr(
                "__a030_rank",
                "cs_rank",
                Some("__a030_sum123"),
                None,
                None,
                None,
                None,
            ),
            expr(
                "__a030_one_minus",
                "elem_sub",
                None,
                Some("1"),
                Some("__a030_rank"),
                None,
                None,
            ),
            expr(
                "__a030_sum_v5",
                "ts_sum",
                Some("volume"),
                None,
                None,
                Some(5),
                None,
            ),
            expr(
                "__a030_sum_v20",
                "ts_sum",
                Some("volume"),
                None,
                None,
                Some(20),
                None,
            ),
            expr(
                "__a030_ratio",
                "elem_div",
                None,
                Some("__a030_sum_v5"),
                Some("__a030_sum_v20"),
                None,
                None,
            ),
            expr(
                "alpha030",
                "elem_mul",
                None,
                Some("__a030_one_minus"),
                Some("__a030_ratio"),
                None,
                None,
            ),
            expr(
                "__a043_adv20",
                "ts_mean",
                Some("volume"),
                None,
                None,
                Some(20),
                None,
            ),
            expr(
                "__a043_vratio",
                "elem_div",
                None,
                Some("volume"),
                Some("__a043_adv20"),
                None,
                None,
            ),
            expr(
                "__a043_left",
                "ts_rank",
                Some("__a043_vratio"),
                None,
                None,
                Some(20),
                None,
            ),
            expr(
                "__a043_dc7",
                "ts_delta",
                Some("close"),
                None,
                None,
                None,
                Some(7),
            ),
            expr(
                "__a043_neg_dc7",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a043_dc7"),
                None,
                None,
            ),
            expr(
                "__a043_right",
                "ts_rank",
                Some("__a043_neg_dc7"),
                None,
                None,
                Some(8),
                None,
            ),
            expr(
                "alpha043",
                "elem_mul",
                None,
                Some("__a043_left"),
                Some("__a043_right"),
                None,
                None,
            ),
            expr(
                "__a054_low_close",
                "elem_sub",
                None,
                Some("low"),
                Some("close"),
                None,
                None,
            ),
            expr(
                "__a054_open_pow5",
                "elem_pow",
                None,
                Some("open"),
                Some("5"),
                None,
                None,
            ),
            expr(
                "__a054_num",
                "elem_mul",
                None,
                Some("__a054_low_close"),
                Some("__a054_open_pow5"),
                None,
                None,
            ),
            expr(
                "__a054_low_high",
                "elem_sub",
                None,
                Some("low"),
                Some("high"),
                None,
                None,
            ),
            expr(
                "__a054_close_pow5",
                "elem_pow",
                None,
                Some("close"),
                Some("5"),
                None,
                None,
            ),
            expr(
                "__a054_den",
                "elem_mul",
                None,
                Some("__a054_low_high"),
                Some("__a054_close_pow5"),
                None,
                None,
            ),
            expr(
                "__a054_frac",
                "elem_div",
                None,
                Some("__a054_num"),
                Some("__a054_den"),
                None,
                None,
            ),
            expr(
                "alpha054",
                "elem_mul",
                None,
                Some("-1"),
                Some("__a054_frac"),
                None,
                None,
            ),
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);
    let streaming = retain_from_ts(&streaming, 80);
    let offline = retain_from_ts(&offline, 80);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}
