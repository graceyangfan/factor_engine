use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, Engine, FactorRequest, InputFieldCatalog, OnlineFactorEngine, Planner,
    SimplePlanner,
};
use std::collections::{BTreeMap, HashMap};

mod common;
use common::alpha101::*;

#[test]
fn polars_offline_matches_online_alpha101_batch_c_parity() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![9501_u32, 9502, 9503, 9504, 9505, 9506, 9507, 9508];
    let ts_points: Vec<i64> = (1..=520).collect();

    let outputs = vec![
        "alpha031".to_string(),
        "alpha032".to_string(),
        "alpha037".to_string(),
        "alpha039".to_string(),
        "alpha047".to_string(),
        "alpha049".to_string(),
        "alpha050".to_string(),
        "alpha051".to_string(),
        "alpha052".to_string(),
        "alpha057".to_string(),
        "alpha060".to_string(),
        "alpha061".to_string(),
    ];

    let exprs = vec![
        "cs_rank(cs_rank(cs_rank(ts_decay_linear(-1 * cs_rank(cs_rank(ts_delta(close, 10))), 10)))) + cs_rank(-1 * ts_delta(close, 3)) + elem_sign(cs_scale(ts_corr(ts_mean(volume, 20), low, 12)))".to_string(),
        "cs_scale(ts_sum(close, 7) / 7 - close) + 20 * cs_scale(ts_corr((open + high + low + close) / 4, ts_lag(close, 5), 230))".to_string(),
        "cs_rank(ts_corr(ts_lag(open - close, 1), close, 200)) + cs_rank(open - close)".to_string(),
        "-1 * cs_rank(ts_delta(close, 7) * (1 - cs_rank(ts_decay_linear(volume / ts_mean(volume, 20), 9)))) * (1 + cs_rank(ts_sum((ts_delta(close, 1) / (ts_lag(close, 1) + 0.000001)), 250)))".to_string(),
        "cs_rank(1 / close) * volume / ts_mean(volume, 20) * (high * cs_rank(high - close) / (ts_sum(high, 5) / 5)) - cs_rank(((open + high + low + close) / 4) - ts_lag(((open + high + low + close) / 4), 5))".to_string(),
        "elem_where(((ts_lag(close, 20) - ts_lag(close, 10)) / 10 - (ts_lag(close, 10) - close) / 10) < -0.1, 1, -1 * (close - ts_lag(close, 1)))".to_string(),
        "-1 * ts_max(cs_rank(ts_corr(cs_rank(volume), cs_rank((open + high + low + close) / 4), 5)), 5)".to_string(),
        "elem_where(((ts_lag(close, 20) - ts_lag(close, 10)) / 10 - (ts_lag(close, 10) - close) / 10) < -0.05, 1, -1 * (close - ts_lag(close, 1)))".to_string(),
        "(-1 * ts_min(low, 5) + ts_lag(ts_min(low, 5), 5)) * cs_rank((ts_sum((ts_delta(close, 1) / (ts_lag(close, 1) + 0.000001)), 240) - ts_sum((ts_delta(close, 1) / (ts_lag(close, 1) + 0.000001)), 20)) / 220) * ts_rank(volume, 5)".to_string(),
        "-1 * ((close - ((open + high + low + close) / 4)) / ts_decay_linear(cs_rank(ts_argmax(close, 30)), 2))".to_string(),
        "-1 * (2 * cs_scale(cs_rank((close - low - (high - close)) / (high - low) * volume)) - cs_scale(cs_rank(ts_argmax(close, 10))))".to_string(),
        "cs_rank(((open + high + low + close) / 4) - ts_min(((open + high + low + close) / 4), 16)) < cs_rank(ts_corr(((open + high + low + close) / 4), ts_mean(volume, 180), 18))".to_string(),
    ];

    let request = FactorRequest {
        exprs,
        outputs: outputs.clone(),
        opts: CompileOptions::default(),
    };

    let planner = SimplePlanner;
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.expr_count, outputs.len(), "{manifest:?}");
    assert!(manifest.cse_hit_count >= 30, "{manifest:?}");

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
            let drift = 0.038 + 0.004 * base;
            let open = 70.0
                + 1.8 * base
                + drift * t
                + (t * (0.011 + 0.0015 * base) + base * 0.37).sin() * (0.75 + 0.05 * base);
            let close = open
                + 0.12 * base
                + (t * (0.017 + 0.0023 * base) + base * 0.29).cos() * (0.58 + 0.06 * base)
                + (t * t) * (0.000012 * base);
            let high = open.max(close) + 0.26 + base * 0.013;
            let low = open.min(close) - 0.24 - base * 0.011;
            let volume = 2600.0
                + 95.0 * base
                + t * (3.1 + 0.22 * base)
                + (t * (0.015 + 0.0018 * base) + base * 0.41).cos() * 62.0
                + (t * 0.009 * base).sin() * 8.0;

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
    let online = retain_from_ts(&retain_outputs(&streaming, &outputs), 300);
    let offline = retain_from_ts(&retain_outputs(&offline_all, &outputs), 300);
    assert_eq!(online.len(), offline.len(), "row count mismatch");

    for (key, lhs) in &online {
        let rhs = offline
            .get(key)
            .unwrap_or_else(|| panic!("missing key in offline baseline: {:?}", key));
        assert_close(*lhs, *rhs, key);
    }
}
