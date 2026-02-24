use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, Engine, FactorRequest, InputFieldCatalog, OnlineFactorEngine, Planner,
    SimplePlanner,
};
use std::collections::{BTreeMap, HashMap};

mod common;
use common::alpha101::*;

#[test]
fn polars_offline_matches_online_alpha101_batch_f_more_missing_set() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9901_u32, 9902, 9903, 9904, 9905, 9906, 9907, 9908];
    let ts_points: Vec<i64> = (1..=700).collect();

    let outputs = vec![
        "alpha066".to_string(),
        "alpha068".to_string(),
        "alpha085".to_string(),
    ];

    let exprs = vec![
        "-1 * (cs_rank(ts_decay_linear(ts_delta((open + high + low + close) / 4, 4), 7)) + ts_rank(ts_decay_linear((low - ((open + high + low + close) / 4)) / (open - (high + low) / 2), 11), 7))".to_string(),
        "-1 * elem_to_int(ts_rank(ts_corr(cs_rank(high), cs_rank(ts_mean(volume, 15)), 9), 14) < cs_rank(ts_delta(close * 0.518371 + low * (1 - 0.518371), 1)))".to_string(),
        "elem_pow(cs_rank(ts_corr(high * 0.876703 + close * (1 - 0.876703), ts_mean(volume, 30), 10)), cs_rank(ts_corr(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7)))".to_string(),
    ];

    let request = FactorRequest {
        exprs,
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
            let trend = 0.028 + 0.0041 * base;
            let open = 55.0
                + 2.1 * base
                + trend * t
                + (t * (0.012 + 0.0018 * base) + base * 0.33).sin() * (0.9 + 0.06 * base)
                + (t * (0.004 + 0.0005 * base) + base * 0.19).cos() * (0.25 + 0.02 * base);
            let close = open
                + 0.14 * base
                + (t * (0.018 + 0.0021 * base) + base * 0.27).cos() * (0.7 + 0.05 * base)
                + (t * t) * (0.000009 * base);
            let high = open.max(close)
                + 0.29
                + base * 0.015
                + (t * (0.007 + 0.0007 * base)).sin().abs() * 0.06;
            let low = open.min(close)
                - 0.27
                - base * 0.013
                - (t * (0.008 + 0.0006 * base)).cos().abs() * 0.05;
            let volume = 3500.0
                + 120.0 * base
                + t * (3.4 + 0.23 * base)
                + (t * (0.013 + 0.0018 * base) + base * 0.41).cos() * 72.0
                + (t * (0.010 + 0.0010 * base)).sin() * 15.0;

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
    let online = retain_from_ts(&retain_outputs(&streaming, &outputs), 420);
    let offline = retain_from_ts(&retain_outputs(&offline_all, &outputs), 420);
    assert_eq!(online.len(), offline.len(), "row count mismatch");

    for (key, lhs) in &online {
        let rhs = offline
            .get(key)
            .unwrap_or_else(|| panic!("missing key in offline baseline: {:?}", key));
        assert_close(*lhs, *rhs, key);
    }
}
