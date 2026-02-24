use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, DataLite, Engine, EventEnvelope, FactorRequest, InputFieldCatalog,
    OnlineFactorEngine, Payload, Planner, SimplePlanner, SourceKind,
};
use std::collections::{BTreeMap, HashMap};

mod common;
use common::alpha101::*;

fn data_event(ts: i64, instrument_slot: u32, source_slot: u16, industry: f64) -> EventEnvelope {
    EventEnvelope {
        ts_event_ns: ts,
        ts_init_ns: ts,
        seq: 2,
        source_slot,
        instrument_slot,
        quality_flags: 0,
        payload: Payload::Data(DataLite {
            values: vec![("industry".to_string(), industry)],
        }),
    }
}

fn pseudo_noise(ts: i64, instrument_slot: u32, salt: u64) -> f64 {
    let mut x = (ts as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add((instrument_slot as u64).wrapping_mul(1442695040888963407))
        .wrapping_add(salt.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    let u = (x as f64) / (u64::MAX as f64);
    u * 2.0 - 1.0
}

#[test]
fn polars_offline_matches_online_alpha101_alpha089_with_neutralize_term() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![10011_u32, 10012, 10013, 10014, 10015, 10016, 10017, 10018];
    let ts_points: Vec<i64> = (1..=900).collect();

    let outputs = vec!["alpha089".to_string()];
    let vwap = "(open + high + low + close) / 4";
    let lhs = "ts_rank(ts_decay_linear(ts_corr(low * 0.967285 + low * 0.032715, ts_mean(volume, 10), 7), 6), 4)";
    let rhs = format!(
        "ts_rank(ts_decay_linear(ts_delta(cs_neutralize({vwap}, group=data.industry), 3), 10), 15)"
    );
    let exprs = vec![format!("{lhs} - {rhs}")];

    let request = FactorRequest {
        exprs,
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
        "data.industry".to_string(),
    ]);
    let physical = planner
        .bind(
            &logical,
            &Universe::new(universe.clone()),
            &catalog,
            AdvancePolicy::StrictAllReady,
        )
        .expect("bind should succeed");
    let data_source_slot = physical
        .source_kinds
        .iter()
        .position(|kind| *kind == SourceKind::Data)
        .expect("data source slot should exist") as u16;

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let mut offline_rows = Vec::new();
    let mut streaming = HashMap::new();
    for &ts in &ts_points {
        for (inst_idx, &instrument_slot) in universe.iter().enumerate() {
            let base = (inst_idx + 1) as f64;
            let t = ts as f64;
            let n1 = pseudo_noise(ts, instrument_slot, 1);
            let n2 = pseudo_noise(ts, instrument_slot, 2);
            let n3 = pseudo_noise(ts, instrument_slot, 3);
            let open = 48.0
                + 2.3 * base
                + 0.032 * t
                + (t * (0.018 + 0.0013 * base) + base * 0.31).sin() * (1.05 + 0.04 * base)
                + n1 * 0.48;
            let close = open
                + (t * (0.014 + 0.0015 * base) + base * 0.27).cos() * (0.88 + 0.03 * base)
                + n2 * 0.44;
            let high = open.max(close) + 0.31 + n3.abs() * 0.17;
            let low = open.min(close) - 0.29 - n1.abs() * 0.16;
            let volume = 3200.0
                + 140.0 * base
                + t * (3.1 + 0.19 * base)
                + (t * (0.067 + 0.003 * base) + base * 0.22).sin() * 55.0
                + (n2 - n3) * 42.0;
            let industry = (inst_idx % 2) as f64 + 11.0;

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
            engine
                .on_event(&data_event(ts, instrument_slot, data_source_slot, industry))
                .expect("data event should succeed");

            offline_rows.push(OfflineInputRow {
                ts,
                instrument_slot,
                fields: BTreeMap::from([
                    ("open".to_string(), open),
                    ("high".to_string(), high),
                    ("low".to_string(), low),
                    ("close".to_string(), close),
                    ("volume".to_string(), volume),
                    ("industry".to_string(), industry),
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
