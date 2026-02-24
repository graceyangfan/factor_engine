use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, Engine, FactorRequest, InputFieldCatalog, OnlineFactorEngine, Planner,
    SimplePlanner,
};
use std::collections::{BTreeMap, HashMap};

mod common;
use common::alpha101::*;

#[test]
fn polars_offline_matches_online_alpha101_batch_i_more_missing_set() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9971_u32, 9972, 9973, 9974, 9975, 9976, 9977, 9978];
    let ts_points: Vec<i64> = (1..=780).collect();

    let outputs = vec!["alpha064".to_string(), "alpha073".to_string()];

    let exprs = vec![
        "-1 * elem_to_int(cs_rank(ts_corr(ts_sum(open * 0.178404 + low * (1 - 0.178404), 13), ts_sum(ts_mean(volume, 120), 13), 17)) < cs_rank(ts_delta(((high + low) / 2) * 0.178404 + ((open + high + low + close) / 4) * (1 - 0.178404), 4)))".to_string(),
        "-1 * elem_max(cs_rank(ts_decay_linear(ts_delta((open + high + low + close) / 4, 5), 3)), ts_rank(ts_decay_linear((-1 * ts_delta(open * 0.147155 + low * (1 - 0.147155), 2)) / (open * 0.147155 + low * (1 - 0.147155)), 3), 17))".to_string(),
    ];

    let request = FactorRequest {
        exprs,
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.expr_count, outputs.len(), "{manifest:?}");
    assert!(
        manifest.cse_hit_count <= manifest.lowered_op_count,
        "{manifest:?}"
    );

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
            let trend = 0.029 + 0.0043 * base;
            let open = 58.0
                + 2.2 * base
                + trend * t
                + (t * (0.013 + 0.0017 * base) + base * 0.39).sin() * (0.95 + 0.06 * base)
                + (t * (0.0042 + 0.0005 * base) + base * 0.21).cos() * (0.28 + 0.02 * base);
            let close = open
                + 0.16 * base
                + (t * (0.019 + 0.0022 * base) + base * 0.25).cos() * (0.74 + 0.06 * base)
                + (t * t) * (0.000009 * base);
            let high = open.max(close)
                + 0.31
                + base * 0.016
                + (t * (0.007 + 0.0008 * base)).sin().abs() * 0.07;
            let low = open.min(close)
                - 0.28
                - base * 0.014
                - (t * (0.0085 + 0.0007 * base)).cos().abs() * 0.06;
            let volume = 3600.0
                + 128.0 * base
                + t * (3.5 + 0.25 * base)
                + (t * (0.014 + 0.0019 * base) + base * 0.43).cos() * 75.0
                + (t * (0.010 + 0.0011 * base)).sin() * 16.0;

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
    let online = retain_from_ts(&retain_outputs(&streaming, &outputs), 460);
    let offline = retain_from_ts(&retain_outputs(&offline_all, &outputs), 460);
    assert_eq!(online.len(), offline.len(), "row count mismatch");

    for (key, lhs) in &online {
        let rhs = offline
            .get(key)
            .unwrap_or_else(|| panic!("missing key in offline baseline: {:?}", key));
        assert_close(*lhs, *rhs, key);
    }
}
