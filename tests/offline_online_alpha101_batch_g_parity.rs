use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, Engine, FactorRequest, InputFieldCatalog, OnlineFactorEngine, Planner,
    SimplePlanner,
};
use std::collections::{BTreeMap, HashMap};

mod common;
use common::alpha101::*;

#[test]
fn polars_offline_matches_online_alpha101_batch_g_more_missing_set() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9951_u32, 9952, 9953, 9954, 9955, 9956, 9957, 9958];
    let ts_points: Vec<i64> = (1..=720).collect();

    let outputs = vec!["alpha098".to_string()];

    let exprs = vec![
        "cs_rank(ts_decay_linear(ts_corr((open + high + low + close) / 4, ts_sum(ts_mean(volume, 5), 26), 5), 7)) - cs_rank(ts_decay_linear(ts_rank(ts_argmin(ts_corr(cs_rank(open), cs_rank(ts_mean(volume, 15)), 21), 9), 7), 8))".to_string(),
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
            let trend = 0.031 + 0.0039 * base;
            let open = 64.0
                + 1.85 * base
                + trend * t
                + (t * (0.012 + 0.0017 * base) + base * 0.35).sin() * (0.85 + 0.05 * base)
                + (t * (0.0035 + 0.0004 * base) + base * 0.23).cos() * (0.24 + 0.02 * base);
            let close = open
                + 0.12 * base
                + (t * (0.017 + 0.0020 * base) + base * 0.31).cos() * (0.64 + 0.05 * base)
                + (t * t) * (0.0000095 * base);
            let high = open.max(close)
                + 0.27
                + base * 0.014
                + (t * (0.007 + 0.0008 * base)).sin().abs() * 0.06;
            let low = open.min(close)
                - 0.26
                - base * 0.012
                - (t * (0.009 + 0.0007 * base)).cos().abs() * 0.05;
            let volume = 3400.0
                + 125.0 * base
                + t * (3.3 + 0.24 * base)
                + (t * (0.014 + 0.0019 * base) + base * 0.39).cos() * 70.0
                + (t * (0.010 + 0.0011 * base)).sin() * 14.0;

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
    let online = retain_from_ts(&retain_outputs(&streaming, &outputs), 430);
    let offline = retain_from_ts(&retain_outputs(&offline_all, &outputs), 430);
    assert_eq!(online.len(), offline.len(), "row count mismatch");

    for (key, lhs) in &online {
        let rhs = offline
            .get(key)
            .unwrap_or_else(|| panic!("missing key in offline baseline: {:?}", key));
        assert_close(*lhs, *rhs, key);
    }
}
