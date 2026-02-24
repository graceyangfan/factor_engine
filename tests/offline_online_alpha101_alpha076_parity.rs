use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, DataLite, Engine, EventEnvelope, FactorRequest, InputFieldCatalog,
    OnlineFactorEngine, Payload, Planner, SimplePlanner, SourceKind,
};
use std::collections::{BTreeMap, HashMap};

mod common;
use common::alpha101::*;

fn data_event(ts: i64, instrument_slot: u32, source_slot: u16, sector: f64) -> EventEnvelope {
    EventEnvelope {
        ts_event_ns: ts,
        ts_init_ns: ts,
        seq: 2,
        source_slot,
        instrument_slot,
        quality_flags: 0,
        payload: Payload::Data(DataLite {
            values: vec![("sector".to_string(), sector)],
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
fn polars_offline_matches_online_alpha101_alpha076_with_neutralize_term() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![
        10101_u32, 10102, 10103, 10104, 10105, 10106, 10107, 10108, 10109, 10110, 10111, 10112,
    ];
    let ts_points: Vec<i64> = (1..=920).collect();

    let outputs = vec!["alpha076".to_string()];
    let vwap = "(open + high + low + close) / 4";
    let lhs = format!("cs_rank(ts_decay_linear(ts_delta({vwap}, 1), 12))");
    let rhs = "ts_rank(ts_decay_linear(ts_rank(ts_corr(cs_neutralize(low, group=data.sector), ts_mean(volume, 81), 8), 20), 17), 19)";
    let exprs = vec![format!("-1 * elem_max({lhs}, {rhs})")];

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
        "data.sector".to_string(),
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
            let n4 = pseudo_noise(ts, instrument_slot, 4);

            let open = 41.0
                + 1.7 * base
                + 0.029 * t
                + (t * (0.015 + 0.0011 * base) + base * 0.19).sin() * (1.35 + 0.03 * base)
                + n1 * (0.92 + 0.05 * base);
            let close = open
                + (t * (0.017 + 0.0013 * base) + base * 0.21).cos() * (1.08 + 0.04 * base)
                + n2 * (0.83 + 0.03 * base);
            let high = open.max(close) + 0.37 + n3.abs() * 0.23;
            let low = open.min(close) - 0.33 - n4.abs() * 0.21;
            let volume = 2950.0
                + 118.0 * base
                + t * (3.4 + 0.21 * base)
                + (t * (0.061 + 0.0032 * base) + base * 0.33).sin() * 92.0
                + (t * (0.027 + 0.0017 * base) + base * 0.11).cos() * 47.0
                + (n1 - n3) * 88.0;
            let sector = (inst_idx % 4) as f64 + 1.0;

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
                .on_event(&data_event(ts, instrument_slot, data_source_slot, sector))
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
                    ("sector".to_string(), sector),
                ]),
            });
        }

        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all =
        run_polars_offline(&build_legacy_payload_from_logical(offline_rows, &logical));
    let online = retain_from_ts(&retain_outputs(&streaming, &outputs), 500);
    let offline = retain_from_ts(&retain_outputs(&offline_all, &outputs), 500);
    assert_eq!(online.len(), offline.len(), "row count mismatch");

    for (key, lhs) in &online {
        let rhs = offline
            .get(key)
            .unwrap_or_else(|| panic!("missing key in offline baseline: {:?}", key));
        match (*lhs, *rhs) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                let abs = (a - b).abs();
                assert!(
                    abs <= 3.0e-2,
                    "mismatch at {:?}: lhs={} rhs={} abs={}",
                    key,
                    a,
                    b,
                    abs
                );
            }
            _ => panic!(
                "null/finite mismatch at {:?}: lhs={:?} rhs={:?}",
                key, lhs, rhs
            ),
        }
    }
}
