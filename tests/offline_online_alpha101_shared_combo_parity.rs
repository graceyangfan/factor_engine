use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, Engine, FactorRequest, InputFieldCatalog, OnlineFactorEngine, Planner,
    SimplePlanner,
};
use std::collections::{BTreeMap, HashMap};

mod common;
use common::alpha101::*;

#[test]
fn polars_offline_matches_online_ts_stats_shared_combinations() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9701_u32, 9702, 9703, 9704, 9705, 9706];
    let ts_points: Vec<i64> = (1..=300).collect();
    let outputs = vec![
        "u_std20".to_string(),
        "u_var20".to_string(),
        "b_cov20".to_string(),
        "b_corr20".to_string(),
        "b_beta20".to_string(),
        "b_linreg20".to_string(),
        "combo_sum".to_string(),
        "combo_ratio".to_string(),
        "combo_rank".to_string(),
    ];
    let combo_sum = "ts_std(close, 20) + ts_corr(close, open, 20)";
    let request = FactorRequest {
        exprs: vec![
            "ts_std(close, 20)".to_string(),
            "ts_var(close, 20)".to_string(),
            "ts_cov(close, open, 20)".to_string(),
            "ts_corr(close, open, 20)".to_string(),
            "ts_beta(close, open, 20)".to_string(),
            "ts_linear_regression(close, open, 20)".to_string(),
            combo_sum.to_string(),
            "ts_cov(close, open, 20) / (ts_var(open, 20) + 0.000001)".to_string(),
            format!("cs_rank({combo_sum})"),
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
            let t = ts as f64;
            let open = 95.0
                + 3.1 * base
                + t * (0.05 + 0.008 * base)
                + (t * (0.019 + 0.002 * base)).sin() * (0.7 + 0.08 * base);
            let close = open
                + base * 0.13
                + (t * (0.026 + 0.003 * base)).cos() * (0.4 + 0.06 * base)
                + (t * t) * (0.00003 * base);
            let high = open.max(close) + 0.19 + base * 0.01;
            let low = open.min(close) - 0.18 - base * 0.01;
            let volume = 1600.0 + 65.0 * base + t * (2.2 + 0.14 * base);

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
                "u_std20",
                "ts_std",
                Some("close"),
                None,
                None,
                Some(20),
                None,
            ),
            expr(
                "u_var20",
                "ts_var",
                Some("close"),
                None,
                None,
                Some(20),
                None,
            ),
            expr(
                "b_cov20",
                "ts_cov",
                None,
                Some("close"),
                Some("open"),
                Some(20),
                None,
            ),
            expr(
                "b_corr20",
                "ts_corr",
                None,
                Some("close"),
                Some("open"),
                Some(20),
                None,
            ),
            expr(
                "b_beta20",
                "ts_beta",
                None,
                Some("close"),
                Some("open"),
                Some(20),
                None,
            ),
            expr(
                "b_linreg20",
                "ts_linear_regression",
                None,
                Some("close"),
                Some("open"),
                Some(20),
                None,
            ),
            expr(
                "combo_sum",
                "elem_add",
                None,
                Some("u_std20"),
                Some("b_corr20"),
                None,
                None,
            ),
            expr(
                "open_var20",
                "ts_var",
                Some("open"),
                None,
                None,
                Some(20),
                None,
            ),
            expr(
                "open_var20_eps",
                "elem_add",
                None,
                Some("open_var20"),
                Some("0.000001"),
                None,
                None,
            ),
            expr(
                "combo_ratio",
                "elem_div",
                None,
                Some("b_cov20"),
                Some("open_var20_eps"),
                None,
                None,
            ),
            expr(
                "combo_rank",
                "cs_rank",
                Some("combo_sum"),
                None,
                None,
                None,
                None,
            ),
        ],
    });
    let offline = retain_outputs(&offline_all, &outputs);
    let streaming = retain_from_ts(&streaming, 50);
    let offline = retain_from_ts(&offline, 50);

    assert_eq!(streaming.len(), offline.len(), "key count mismatch");
    for (key, lhs) in &streaming {
        let rhs = offline.get(key).expect("offline key missing");
        assert_close(*lhs, *rhs, key);
    }
}
