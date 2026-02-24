use factor_engine::types::{CompileOptions, Universe};
use factor_engine::{
    AdvancePolicy, DataLite, Engine, EventEnvelope, FactorRequest, InputFieldCatalog,
    OnlineFactorEngine, Payload, Planner, SimplePlanner, SourceKind,
};
use std::collections::{BTreeMap, HashMap};

mod common;
use common::alpha101::*;

fn data_event(
    ts: i64,
    instrument_slot: u32,
    source_slot: u16,
    cap: f64,
    sector: f64,
    industry: f64,
    subindustry: f64,
) -> EventEnvelope {
    EventEnvelope {
        ts_event_ns: ts,
        ts_init_ns: ts,
        seq: 2,
        source_slot,
        instrument_slot,
        quality_flags: 0,
        payload: Payload::Data(DataLite {
            values: vec![
                ("cap".to_string(), cap),
                ("sector".to_string(), sector),
                ("industry".to_string(), industry),
                ("subindustry".to_string(), subindustry),
            ],
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
fn polars_offline_matches_online_alpha101_batch_j_final_complex_set() {
    if !python_polars_available() {
        eprintln!("skip: python3/polars unavailable");
        return;
    }

    let universe = vec![
        9981_u32, 9982, 9983, 9984, 9985, 9986, 9987, 9988, 9989, 9990,
    ];
    let ts_points: Vec<i64> = (1..=840).collect();

    let outputs = vec![
        "alpha029".to_string(),
        "alpha048".to_string(),
        "alpha056".to_string(),
        "alpha058".to_string(),
        "alpha059".to_string(),
        "alpha062".to_string(),
        "alpha063".to_string(),
        "alpha065".to_string(),
        "alpha067".to_string(),
        "alpha069".to_string(),
        "alpha070".to_string(),
        "alpha079".to_string(),
        "alpha080".to_string(),
        "alpha082".to_string(),
        "alpha087".to_string(),
        "alpha090".to_string(),
        "alpha091".to_string(),
        "alpha093".to_string(),
        "alpha097".to_string(),
        "alpha100".to_string(),
    ];

    let vwap = "((open + high + low + close) / 4)";
    let ret = "(ts_delta(close, 1) / (ts_lag(close, 1) + 0.000001))";

    let exprs = vec![
        format!(
            "elem_min(ts_product(cs_rank(cs_rank(cs_scale(elem_log(ts_sum(ts_min(cs_rank(cs_rank(-1 * cs_rank(ts_delta(close - 1, 5)))), 2), 1))))), 1), 5) + ts_rank(ts_lag(-1 * {ret}, 6), 5)"
        ),
        format!(
            "cs_neutralize((ts_corr(ts_delta(close, 1), ts_delta(ts_lag(close, 1), 1), 250) * ts_delta(close, 1)) / (close + 0.000001), group=data.subindustry) / ts_sum(elem_pow(ts_delta(close, 1) / (ts_lag(close, 1) + 0.000001), 2), 250)"
        ),
        format!(
            "-1 * cs_rank(ts_sum({ret}, 10) / ts_sum(ts_sum({ret}, 2), 3)) * cs_rank({ret} * data.cap)"
        ),
        format!(
            "-1 * ts_rank(ts_decay_linear(ts_corr(cs_neutralize({vwap}, group=data.sector), volume, 4), 8), 6)"
        ),
        format!(
            "-1 * ts_rank(ts_decay_linear(ts_corr(cs_neutralize({vwap} * 0.728317 + {vwap} * (1 - 0.728317), group=data.industry), volume, 4), 16), 8)"
        ),
        format!(
            "-1 * elem_to_int(cs_rank(ts_corr({vwap}, ts_sum(ts_mean(volume, 20), 22), 10)) < cs_rank((cs_rank(open) + cs_rank(open)) < (cs_rank((high + low) / 2) + cs_rank(high))))"
        ),
        format!(
            "-1 * (cs_rank(ts_decay_linear(ts_delta(cs_neutralize(close, group=data.industry), 2), 8)) - cs_rank(ts_decay_linear(ts_corr({vwap} * 0.318108 + open * (1 - 0.318108), ts_sum(ts_mean(volume, 180), 37), 14), 12)))"
        ),
        format!(
            "-1 * elem_to_int(cs_rank(ts_corr(open * 0.00817205 + {vwap} * (1 - 0.00817205), ts_sum(ts_mean(volume, 60), 9), 6)) < cs_rank(open - ts_min(open, 14)))"
        ),
        format!(
            "-1 * elem_pow(cs_rank(high - ts_min(high, 2)), cs_rank(ts_corr(cs_neutralize({vwap}, group=data.sector), cs_neutralize(ts_mean(volume, 20), group=data.subindustry), 6)))"
        ),
        format!(
            "-1 * elem_pow(cs_rank(ts_max(ts_delta(cs_neutralize({vwap}, group=data.industry), 3), 5)), ts_rank(ts_corr(close * 0.490655 + {vwap} * (1 - 0.490655), ts_mean(volume, 20), 5), 9))"
        ),
        format!(
            "-1 * elem_pow(cs_rank(ts_delta({vwap}, 1)), ts_rank(ts_corr(cs_neutralize(close, group=data.industry), ts_mean(volume, 50), 18), 18))"
        ),
        format!(
            "elem_to_int(cs_rank(ts_delta(cs_neutralize(close * 0.60733 + open * (1 - 0.60733), group=data.sector), 1)) < cs_rank(ts_corr(ts_rank({vwap}, 4), ts_rank(ts_mean(volume, 150), 9), 15)))"
        ),
        format!(
            "-1 * elem_pow(cs_rank(elem_sign(ts_delta(open * 0.868128 + high * (1 - 0.868128), 4))), ts_rank(ts_corr(high, ts_mean(volume, 10), 5), 6))"
        ),
        format!(
            "-1 * elem_min(cs_rank(ts_decay_linear(ts_delta(open, 1), 15)), ts_rank(ts_decay_linear(ts_corr(cs_neutralize(volume, group=data.sector), open * 0.634196 + open * (1 - 0.634196), 17), 7), 13))"
        ),
        format!(
            "-1 * elem_max(cs_rank(ts_decay_linear(ts_delta(close * 0.369701 + {vwap} * (1 - 0.369701), 2), 3)), ts_rank(ts_decay_linear(elem_abs(ts_corr(cs_neutralize(ts_mean(volume, 81), group=data.industry), close, 13)), 5), 14))"
        ),
        format!(
            "-1 * elem_pow(cs_rank(close - ts_max(close, 5)), ts_rank(ts_corr(cs_neutralize(ts_mean(volume, 40), group=data.subindustry), low, 5), 3))"
        ),
        format!(
            "-1 * (ts_rank(ts_decay_linear(ts_decay_linear(ts_corr(cs_neutralize(close, group=data.industry), volume, 10), 16), 4), 5) - cs_rank(ts_decay_linear(ts_corr({vwap}, ts_mean(volume, 30), 4), 3)))"
        ),
        format!(
            "ts_rank(ts_decay_linear(ts_corr(cs_neutralize({vwap}, group=data.industry), ts_mean(volume, 81), 17), 20), 8) / cs_rank(ts_decay_linear(ts_delta(close * 0.524434 + {vwap} * (1 - 0.524434), 3), 16))"
        ),
        format!(
            "-1 * (cs_rank(ts_decay_linear(ts_delta(cs_neutralize(low * 0.721001 + {vwap} * (1 - 0.721001), group=data.industry), 3), 20)) - ts_rank(ts_decay_linear(ts_rank(ts_corr(ts_rank(low, 8), ts_rank(ts_mean(volume, 60), 17), 5), 19), 16), 7))"
        ),
        format!(
            "-1 * ((1.5 * cs_scale(cs_neutralize(cs_neutralize(cs_rank(((close - low - (high - close)) / (high - low + 0.000001)) * volume), group=data.subindustry), group=data.subindustry)) - cs_scale(cs_neutralize(ts_corr(close, cs_rank(ts_mean(volume, 20)), 5) - cs_rank(ts_argmin(close, 30)), group=data.subindustry))) * (volume / ts_mean(volume, 20)))"
        ),
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
        "data.cap".to_string(),
        "data.sector".to_string(),
        "data.industry".to_string(),
        "data.subindustry".to_string(),
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
            let trend = 0.028 + 0.0042 * base;
            let open = 64.0
                + 2.4 * base
                + trend * t
                + (t * (0.012 + 0.0019 * base) + base * 0.42).sin() * (1.02 + 0.07 * base)
                + (t * (0.0046 + 0.00055 * base) + base * 0.23).cos() * (0.31 + 0.02 * base)
                + n1 * (0.55 + 0.03 * base);
            let close = open
                + 0.18 * base
                + (t * (0.018 + 0.0024 * base) + base * 0.27).cos() * (0.77 + 0.06 * base)
                + (t * t) * (0.0000087 * base)
                + n2 * (0.42 + 0.025 * base);
            let high = open.max(close)
                + 0.33
                + base * 0.017
                + (t * (0.0075 + 0.0008 * base)).sin().abs() * 0.08
                + n3.abs() * 0.12;
            let low = open.min(close)
                - 0.30
                - base * 0.015
                - (t * (0.0092 + 0.00075 * base)).cos().abs() * 0.07
                - n4.abs() * 0.11;
            let volume = 3720.0
                + 136.0 * base
                + t * (3.7 + 0.27 * base)
                + (t * (0.0145 + 0.00195 * base) + base * 0.47).cos() * 81.0
                + (t * (0.0105 + 0.00115 * base)).sin() * 18.0
                + (t * (0.071 + 0.0032 * base) + base * 0.17).sin() * 35.0
                + ((t * t) * (0.00027 + 0.00002 * base) + base * 0.11).cos() * 12.0
                + (n1 + n3) * 47.0;

            let sector = (inst_idx % 3) as f64 + 1.0;
            let industry = (inst_idx % 3) as f64 + 11.0;
            let subindustry = (inst_idx % 4) as f64 + 21.0;
            let cap = 1_200_000.0
                + base * 13_500.0
                + t * (46.0 + 1.3 * base)
                + (t * (0.008 + 0.0005 * base) + base * 0.31).cos() * 2600.0
                + (n2 - n4) * 3400.0;

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
                .on_event(&data_event(
                    ts,
                    instrument_slot,
                    data_source_slot,
                    cap,
                    sector,
                    industry,
                    subindustry,
                ))
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
                    ("cap".to_string(), cap),
                    ("sector".to_string(), sector),
                    ("industry".to_string(), industry),
                    ("subindustry".to_string(), subindustry),
                ]),
            });
        }

        let frame = engine.advance(ts).expect("advance should succeed");
        record_frame_results(&mut streaming, &frame, &outputs, &universe, ts);
    }

    let offline_all =
        run_polars_offline(&build_legacy_payload_from_logical(offline_rows, &logical));
    let online = retain_from_ts(&retain_outputs(&streaming, &outputs), 520);
    let offline = retain_from_ts(&retain_outputs(&offline_all, &outputs), 520);
    assert_eq!(online.len(), offline.len(), "row count mismatch");

    for (key, lhs) in &online {
        let rhs = offline
            .get(key)
            .unwrap_or_else(|| panic!("missing key in offline baseline: {:?}", key));
        assert_close(*lhs, *rhs, key);
    }
}
