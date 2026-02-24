use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, Engine, FactorRequest, InputFieldCatalog, OnlineFactorEngine, Planner,
    SimplePlanner,
};
use std::collections::{BTreeMap, HashMap};

mod common;
use common::alpha101::*;

#[test]
fn polars_offline_matches_online_alpha101_batch_h_more_missing_set() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9961_u32, 9962, 9963, 9964, 9965, 9966, 9967, 9968];
    let ts_points: Vec<i64> = (1..=760).collect();

    let outputs = vec![
        "alpha071".to_string(),
        "alpha081".to_string(),
        "alpha088".to_string(),
        "alpha092".to_string(),
        "alpha094".to_string(),
        "alpha095".to_string(),
        "alpha096".to_string(),
    ];

    let exprs = vec![
        "elem_max(ts_rank(ts_decay_linear(ts_corr(ts_rank(close, 3), ts_rank(ts_mean(volume, 180), 12), 18), 4), 16), ts_rank(ts_decay_linear(elem_pow(cs_rank(low + open - (((open + high + low + close) / 4) + ((open + high + low + close) / 4))), 2), 16), 4))".to_string(),
        "-1 * elem_to_int(cs_rank(elem_log(ts_product(cs_rank(elem_pow(cs_rank(ts_corr((open + high + low + close) / 4, ts_sum(ts_mean(volume, 10), 50), 8)), 4)), 15))) < cs_rank(ts_corr(cs_rank((open + high + low + close) / 4), cs_rank(volume), 5)))".to_string(),
        "elem_min(cs_rank(ts_decay_linear(cs_rank(open) + cs_rank(low) - (cs_rank(high) + cs_rank(close)), 8)), ts_rank(ts_decay_linear(ts_corr(ts_rank(close, 8), ts_rank(ts_mean(volume, 60), 21), 8), 7), 3))".to_string(),
        "elem_min(ts_rank(ts_decay_linear(((high + low) / 2 + close) < (low + open), 15), 19), ts_rank(ts_decay_linear(ts_corr(cs_rank(low), cs_rank(ts_mean(volume, 30)), 8), 7), 7))".to_string(),
        "-1 * elem_pow(cs_rank(((open + high + low + close) / 4) - ts_min(((open + high + low + close) / 4), 12)), ts_rank(ts_corr(ts_rank(((open + high + low + close) / 4), 20), ts_rank(ts_mean(volume, 60), 4), 18), 3))".to_string(),
        "cs_rank(open - ts_min(open, 12)) < ts_rank(elem_pow(cs_rank(ts_corr(ts_sum((high + low) / 2, 19), ts_sum(ts_mean(volume, 40), 19), 13)), 5), 12)".to_string(),
        "-1 * elem_max(ts_rank(ts_decay_linear(ts_corr(cs_rank((open + high + low + close) / 4), cs_rank(volume), 4), 4), 8), ts_rank(ts_decay_linear(ts_argmax(ts_corr(ts_rank(close, 7), ts_rank(ts_mean(volume, 60), 4), 4), 13), 14), 13))".to_string(),
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
            let trend = 0.032 + 0.0037 * base;
            let open = 72.0
                + 1.75 * base
                + trend * t
                + (t * (0.011 + 0.0016 * base) + base * 0.37).sin() * (0.82 + 0.05 * base)
                + (t * (0.0038 + 0.00045 * base) + base * 0.19).cos() * (0.23 + 0.02 * base);
            let close = open
                + 0.13 * base
                + (t * (0.017 + 0.0019 * base) + base * 0.29).cos() * (0.62 + 0.05 * base)
                + (t * t) * (0.0000092 * base);
            let high = open.max(close)
                + 0.28
                + base * 0.014
                + (t * (0.007 + 0.0008 * base)).sin().abs() * 0.06;
            let low = open.min(close)
                - 0.25
                - base * 0.012
                - (t * (0.009 + 0.0007 * base)).cos().abs() * 0.05;
            let volume = 3450.0
                + 123.0 * base
                + t * (3.25 + 0.22 * base)
                + (t * (0.014 + 0.00185 * base) + base * 0.41).cos() * 69.0
                + (t * (0.010 + 0.0010 * base)).sin() * 13.0;

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
    let online = retain_from_ts(&retain_outputs(&streaming, &outputs), 450);
    let offline = retain_from_ts(&retain_outputs(&offline_all, &outputs), 450);
    assert_eq!(online.len(), offline.len(), "row count mismatch");

    for (key, lhs) in &online {
        let rhs = offline
            .get(key)
            .unwrap_or_else(|| panic!("missing key in offline baseline: {:?}", key));
        assert_close(*lhs, *rhs, key);
    }
}
