use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, Engine, FactorRequest, InputFieldCatalog, OnlineFactorEngine, Planner,
    SimplePlanner,
};
use std::collections::{BTreeMap, HashMap};

mod common;
use common::alpha101::*;

#[test]
fn polars_offline_matches_online_alpha101_batch_e_more_missing_set() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9801_u32, 9802, 9803, 9804, 9805, 9806, 9807, 9808];
    let ts_points: Vec<i64> = (1..=620).collect();

    let outputs = vec![
        "alpha083".to_string(),
        "alpha084".to_string(),
        "alpha086".to_string(),
        "alpha099".to_string(),
    ];

    let exprs = vec![
        "cs_rank(ts_lag((high - low) / (ts_sum(close, 5) / 5), 2)) * cs_rank(cs_rank(volume)) / ((high - low) / (ts_sum(close, 5) / 5) / (((open + high + low + close) / 4) - close))".to_string(),
        "elem_signed_power(ts_rank(((open + high + low + close) / 4) - ts_max(((open + high + low + close) / 4), 15), 21), ts_delta(close, 5))".to_string(),
        "-1 * elem_to_int(ts_rank(ts_corr(close, ts_sum(ts_mean(volume, 20), 15), 6), 20) < cs_rank(close - ((open + high + low + close) / 4)))".to_string(),
        "-1 * elem_to_int(cs_rank(ts_corr(ts_sum((high + low) / 2, 20), ts_sum(ts_mean(volume, 60), 20), 9)) < cs_rank(ts_corr(low, volume, 6)))".to_string(),
    ];

    let request = FactorRequest {
        exprs,
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.expr_count, outputs.len(), "{manifest:?}");
    assert!(manifest.cse_hit_count >= 10, "{manifest:?}");

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
            let trend = 0.034 + 0.0036 * base;
            let open = 78.0
                + 1.7 * base
                + trend * t
                + (t * (0.011 + 0.0012 * base) + base * 0.33).sin() * (0.72 + 0.05 * base)
                + (t * (0.0041 + 0.0006 * base) + base * 0.17).cos() * (0.18 + 0.02 * base);
            let close = open
                + 0.11 * base
                + (t * (0.015 + 0.0020 * base) + base * 0.28).cos() * (0.57 + 0.04 * base)
                + (t * t) * (0.000010 * base)
                + (t * (0.020 + 0.0014 * base) + base * 0.12).sin() * 0.17;
            let high = open.max(close)
                + 0.25
                + base * 0.014
                + (t * (0.0068 + 0.0007 * base)).sin().abs() * 0.05;
            let low = open.min(close)
                - 0.23
                - base * 0.011
                - (t * (0.0073 + 0.0008 * base)).cos().abs() * 0.05;
            let volume = 3300.0
                + 115.0 * base
                + t * (3.2 + 0.21 * base)
                + (t * (0.013 + 0.0016 * base) + base * 0.41).cos() * 65.0
                + (t * 0.008 * base).sin() * 11.0
                + (t * (0.052 + 0.0025 * base) + base * 0.14).sin() * 30.0
                + ((t * t) * (0.00016 + 0.00002 * base) + base * 0.08).cos() * 9.0;

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
    let online = retain_from_ts(&retain_outputs(&streaming, &outputs), 360);
    let offline = retain_from_ts(&retain_outputs(&offline_all, &outputs), 360);
    assert_eq!(online.len(), offline.len(), "row count mismatch");

    for (key, lhs) in &online {
        let rhs = offline
            .get(key)
            .unwrap_or_else(|| panic!("missing key in offline baseline: {:?}", key));
        assert_close(*lhs, *rhs, key);
    }
}
