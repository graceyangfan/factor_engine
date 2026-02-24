use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, Engine, FactorRequest, InputFieldCatalog, OnlineFactorEngine, Planner,
    SimplePlanner,
};
use std::collections::{BTreeMap, HashMap};

mod common;
use common::alpha101::*;

#[test]
fn polars_offline_matches_online_ts_stats_single_ops() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9601_u32, 9602, 9603, 9604, 9605];
    let ts_points: Vec<i64> = (1..=260).collect();
    let outputs = vec![
        "std20".to_string(),
        "var20".to_string(),
        "cov20".to_string(),
        "corr20".to_string(),
        "beta20".to_string(),
        "linreg20".to_string(),
    ];
    let request = FactorRequest {
        exprs: vec![
            "ts_std(close, 20)".to_string(),
            "ts_var(close, 20)".to_string(),
            "ts_cov(close, open, 20)".to_string(),
            "ts_corr(close, open, 20)".to_string(),
            "ts_beta(close, open, 20)".to_string(),
            "ts_linear_regression(close, open, 20)".to_string(),
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
            let t = ts as f64;
            let open = 60.0
                + 2.5 * base
                + t * (0.08 + 0.01 * base)
                + (t * (0.021 + 0.003 * base)).sin() * (0.6 + 0.1 * base);
            let close = open
                + base * 0.11
                + (t * (0.031 + 0.004 * base)).cos() * (0.35 + 0.08 * base)
                + (t * t) * (0.00005 * base);
            let high = open.max(close) + 0.22 + base * 0.01;
            let low = open.min(close) - 0.23 - base * 0.01;
            let volume = 1100.0 + 45.0 * base + t * (1.5 + 0.15 * base);

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
            expr("std20", "ts_std", Some("close"), None, None, Some(20), None),
            expr("var20", "ts_var", Some("close"), None, None, Some(20), None),
            expr(
                "cov20",
                "ts_cov",
                None,
                Some("close"),
                Some("open"),
                Some(20),
                None,
            ),
            expr(
                "corr20",
                "ts_corr",
                None,
                Some("close"),
                Some("open"),
                Some(20),
                None,
            ),
            expr(
                "beta20",
                "ts_beta",
                None,
                Some("close"),
                Some("open"),
                Some(20),
                None,
            ),
            expr(
                "linreg20",
                "ts_linear_regression",
                None,
                Some("close"),
                Some("open"),
                Some(20),
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
