use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, Engine, FactorRequest, InputFieldCatalog, OnlineFactorEngine, Planner,
    SimplePlanner,
};
use std::collections::{BTreeMap, HashMap};

mod common;
use common::alpha101::*;

#[test]
fn polars_offline_matches_online_alpha101_batch_d_higher_order_set() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9601_u32, 9602, 9603, 9604, 9605, 9606, 9607, 9608];
    let ts_points: Vec<i64> = (1..=560).collect();

    let outputs = vec![
        "alpha072".to_string(),
        "alpha074".to_string(),
        "alpha075".to_string(),
        "alpha077".to_string(),
        "alpha078".to_string(),
        "alpha101".to_string(),
    ];

    let exprs = vec![
        "cs_rank(ts_decay_linear(ts_corr((high + low) / 2, ts_mean(volume, 40), 9), 10)) / cs_rank(ts_decay_linear(ts_corr(ts_rank((open + high + low + close) / 4, 4), ts_rank(volume, 19), 7), 3))".to_string(),
        "-1 * elem_to_int(cs_rank(ts_corr(close, ts_sum(ts_mean(volume, 30), 37), 15)) < cs_rank(ts_corr(cs_rank(high * 0.0261661 + ((open + high + low + close) / 4) * (1 - 0.0261661)), cs_rank(volume), 11)))".to_string(),
        "cs_rank(ts_corr((open + high + low + close) / 4, volume, 4)) < cs_rank(ts_corr(cs_rank(low), cs_rank(ts_mean(volume, 50)), 12))".to_string(),
        "elem_min(cs_rank(ts_decay_linear((high + low) / 2 + high - (((open + high + low + close) / 4) + high), 20)), cs_rank(ts_decay_linear(ts_corr((high + low) / 2, ts_mean(volume, 40), 3), 6)))".to_string(),
        "elem_pow(cs_rank(ts_corr(ts_sum(low * 0.352233 + ((open + high + low + close) / 4) * (1 - 0.352233), 20), ts_sum(ts_mean(volume, 40), 20), 7)), cs_rank(ts_corr(cs_rank((open + high + low + close) / 4), cs_rank(volume), 6)))".to_string(),
        "(close - open) / (high - low + 0.001)".to_string(),
    ];

    let request = FactorRequest {
        exprs,
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.expr_count, outputs.len(), "{manifest:?}");
    assert!(manifest.cse_hit_count >= 15, "{manifest:?}");

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
            let drift = 0.033 + 0.0038 * base;
            let open = 92.0
                + 1.6 * base
                + drift * t
                + (t * (0.010 + 0.0014 * base) + base * 0.31).sin() * (0.68 + 0.04 * base)
                + (t * (0.0042 + 0.0006 * base) + base * 0.17).cos() * (0.17 + 0.02 * base);
            let close = open
                + 0.08 * base
                + (t * (0.016 + 0.0020 * base) + base * 0.27).cos() * (0.55 + 0.05 * base)
                + (t * t) * (0.000011 * base)
                + (t * (0.021 + 0.0013 * base) + base * 0.11).sin() * 0.19;
            let high = open.max(close)
                + 0.23
                + base * 0.012
                + (t * (0.0068 + 0.0006 * base)).sin().abs() * 0.05;
            let low = open.min(close)
                - 0.22
                - base * 0.010
                - (t * (0.0074 + 0.0007 * base)).cos().abs() * 0.05;
            let volume = 3100.0
                + 110.0 * base
                + t * (3.0 + 0.18 * base)
                + (t * (0.014 + 0.0015 * base) + base * 0.39).cos() * 55.0
                + (t * 0.007 * base).sin() * 10.0
                + (t * (0.051 + 0.0023 * base) + base * 0.13).sin() * 28.0
                + ((t * t) * (0.00017 + 0.00002 * base) + base * 0.07).cos() * 8.0;

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
    let online = retain_from_ts(&retain_outputs(&streaming, &outputs), 320);
    let offline = retain_from_ts(&retain_outputs(&offline_all, &outputs), 320);
    assert_eq!(online.len(), offline.len(), "row count mismatch");

    for (key, lhs) in &online {
        let rhs = offline
            .get(key)
            .unwrap_or_else(|| panic!("missing key in offline baseline: {:?}", key));
        if key.output == "alpha077" {
            match (*lhs, *rhs) {
                (Some(a), Some(b)) => {
                    let abs = (a - b).abs();
                    assert!(
                        abs <= 0.15,
                        "alpha077 mismatch at {:?}: lhs={} rhs={} abs={}",
                        key,
                        a,
                        b,
                        abs
                    );
                }
                _ => assert_close(*lhs, *rhs, key),
            }
            continue;
        }
        assert_close(*lhs, *rhs, key);
    }
}
