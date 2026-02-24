use super::*;

#[test]
fn advance_rejects_non_monotonic_ts() {
    let planner = SimplePlanner;
    let request = ts_only_request();
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    engine
        .on_event(&bar_event(6_000, 101, 10.0, 100.0))
        .expect("event should succeed");
    engine.advance(6_000).expect("advance should succeed");

    let err = engine.advance(5_999).expect_err("must reject ts rollback");
    assert!(matches!(
        err,
        EngineError::NonMonotonicAdvance {
            current_ts_ns: 5_999,
            last_ts_ns: 6_000
        }
    ));
}

#[test]
fn nested_ts_chain_executes_with_derived_field_propagation() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["ts_std(ts_mean(close, 2), 2)".to_string()]);
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    engine
        .on_event(&bar_event(1_000, 101, 10.0, 100.0))
        .expect("event should succeed");
    engine
        .on_event(&bar_event(2_000, 101, 14.0, 100.0))
        .expect("event should succeed");
    let frame2 = engine.advance(2_000).expect("advance should succeed");
    let factor_idx = frame2.factor_idx("f0_ts_std").expect("root factor index");
    assert!(!frame2.valid_mask[factor_idx]);

    engine
        .on_event(&bar_event(3_000, 101, 18.0, 100.0))
        .expect("event should succeed");
    let frame3 = engine.advance(3_000).expect("advance should succeed");
    let factor_idx = frame3.factor_idx("f0_ts_std").expect("root factor index");
    assert!(frame3.valid_mask[factor_idx]);
    assert!(approx_eq(
        frame3
            .factor_value(0, "f0_ts_std")
            .expect("factor value by root name"),
        2.8284271247461903
    ));
}

#[test]
fn elem_infix_expression_executes_with_precedence() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["close + volume * close".to_string()]);
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    engine
        .on_event(&bar_event(1_000, 101, 2.0, 3.0))
        .expect("event should succeed");
    let frame = engine.advance(1_000).expect("advance should succeed");
    assert!(approx_eq(
        frame
            .factor_value(0, "f0_elem_add")
            .expect("factor value by root name"),
        8.0
    ));
}

#[test]
fn elem_infix_expression_executes_with_scalar_literal() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["close * 0.5 + volume".to_string()]);
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    engine
        .on_event(&bar_event(1_000, 101, 10.0, 3.0))
        .expect("event should succeed");
    let frame = engine.advance(1_000).expect("advance should succeed");
    assert!(approx_eq(
        frame
            .factor_value(0, "f0_elem_add")
            .expect("factor value by root name"),
        8.0
    ));
}

#[test]
fn elem_infix_expression_executes_sub_and_div_with_precedence() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["close - volume / close".to_string()]);
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    engine
        .on_event(&bar_event(1_000, 101, 2.0, 3.0))
        .expect("event should succeed");
    let frame = engine.advance(1_000).expect("advance should succeed");
    assert!(approx_eq(
        frame
            .factor_value(0, "f0_elem_sub")
            .expect("factor value by root name"),
        0.5
    ));
}

#[test]
fn elem_div_by_zero_marks_output_invalid() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["close / volume".to_string()]);
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    engine
        .on_event(&bar_event(1_000, 101, 2.0, 0.0))
        .expect("event should succeed");
    let frame = engine.advance(1_000).expect("advance should succeed");
    let idx = frame.factor_idx("f0_elem_div").expect("root factor index");
    assert!(!frame.valid_mask[idx]);
}

#[test]
fn elem_unary_minus_executes_as_mul_neg_one() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["-close".to_string()]);
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    engine
        .on_event(&bar_event(1_000, 101, 2.0, 3.0))
        .expect("event should succeed");
    let frame = engine.advance(1_000).expect("advance should succeed");
    assert!(approx_eq(
        frame
            .factor_value(0, "f0_elem_mul")
            .expect("factor value by root name"),
        -2.0
    ));
}

#[test]
fn ts_univariate_moments_share_profile_and_match_expected_values() {
    let planner = SimplePlanner;
    let request = ts_moments_request();
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![1]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    assert_eq!(engine.debug_ts_univariate_profile_count(), 1);
    assert_eq!(engine.debug_ts_univariate_consumer_count(0), Some(4));

    for (ts, close) in [(1_i64, 1.0_f64), (2, 2.0), (3, 3.0), (4, 4.0)] {
        engine
            .on_event(&bar_event(ts, 1, close, 100.0))
            .expect("event should succeed");
    }
    let frame = engine.advance(4).expect("advance should succeed");
    let (mean, std, skew, kurt) = naive_moments(&[1.0, 2.0, 3.0, 4.0]);

    assert!(approx_eq(frame.value_at(0, 0).expect("mean"), mean));
    assert!(approx_eq(frame.value_at(0, 1).expect("std"), std));
    assert!(approx_eq(frame.value_at(0, 2).expect("skew"), skew));
    assert!(approx_eq(frame.value_at(0, 3).expect("kurt"), kurt));
}

#[test]
fn bivariate_kernels_support_corr_and_linear_regression_slope() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec![
        "ts_corr(close, volume, 3)".to_string(),
        "ts_linear_regression(close, volume, 3)".to_string(),
    ]);
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![9]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    assert_eq!(engine.debug_ts_bivariate_profile_count(), 1);
    assert_eq!(engine.debug_ts_bivariate_consumer_count(0), Some(2));
    for (ts, close, volume) in [(1_i64, 1.0_f64, 2.0_f64), (2, 2.0, 4.0), (3, 3.0, 6.0)] {
        engine
            .on_event(&bar_event(ts, 9, close, volume))
            .expect("event should succeed");
    }
    let frame = engine.advance(3).expect("advance should succeed");
    assert!(approx_eq(frame.value_at(0, 0).expect("corr"), 1.0));
    assert!(approx_eq(frame.value_at(0, 1).expect("slope"), 2.0));
}

#[test]
fn delta_marks_non_finite_result_invalid() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["ts_delta(close, 1)".to_string()]);
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![1]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    engine
        .on_event(&bar_event(10_000, 1, 10.0, 1.0))
        .expect("event should succeed");
    engine
        .on_event(&bar_event(10_100, 1, f64::NAN, 1.0))
        .expect("event should succeed");
    let frame = engine.advance(10_100).expect("advance should succeed");
    assert!(!frame.is_valid_at(0, 0));
    assert!(frame.value_at(0, 0).expect("delta output").is_nan());
}

#[test]
fn cs_rank_treats_non_finite_inputs_as_invalid() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["cs_rank(close)".to_string()]);
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![1, 2]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    engine
        .on_event(&bar_event(11_000, 1, f64::INFINITY, 1.0))
        .expect("event should succeed");
    engine
        .on_event(&bar_event(11_000, 2, 5.0, 1.0))
        .expect("event should succeed");

    let frame = engine.advance(11_000).expect("advance should succeed");
    assert!(!frame.is_valid_at(0, 0));
    assert!(frame.value_at(0, 0).expect("rank output").is_nan());
    assert!(frame.is_valid_at(1, 0));
    assert!(approx_eq(frame.value_at(1, 0).expect("rank value"), 0.0));
}

#[test]
fn force_policy_allows_advance_when_not_ready() {
    let planner = SimplePlanner;
    let request = three_factor_request();
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![1, 2]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::ForceWithLast)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    engine
        .on_event(&bar_event(2_000, 1, 11.0, 101.0))
        .expect("event should succeed");
    assert!(!engine.is_graph_ready(2_000));

    let frame = engine.advance(2_000).expect("force advance should succeed");
    assert_eq!(frame.instrument_count, 2);
    assert_eq!(
        frame.quality_flags[0] & QUALITY_FORCED_ADVANCE,
        QUALITY_FORCED_ADVANCE
    );
    assert_eq!(
        frame.quality_flags[1] & QUALITY_FORCED_ADVANCE,
        QUALITY_FORCED_ADVANCE
    );
}

#[test]
fn strict_and_force_policies_diverge_on_partial_cross_source_ticks_and_converge_when_ready() {
    let planner = SimplePlanner;
    let request = cross_source_policy_request();
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert!(
        manifest.cse_hit_count >= 2,
        "expected shared cross-source subgraph, got manifest={manifest:?}"
    );
    let universe = Universe::new(vec![1, 2]);
    let catalog = InputFieldCatalog::new(vec![
        "bar.close".to_string(),
        "quote_tick.bid_price".to_string(),
    ]);
    let strict_plan = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("strict bind should succeed");
    let force_plan = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::ForceWithLast)
        .expect("force bind should succeed");

    let mut strict_engine = OnlineFactorEngine::default();
    strict_engine
        .load(strict_plan)
        .expect("strict load should succeed");
    let mut force_engine = OnlineFactorEngine::default();
    force_engine
        .load(force_plan)
        .expect("force load should succeed");

    for ts in 1_i64..=6_i64 {
        for &instrument in &[1_u32, 2_u32] {
            let base = instrument as f64;
            let close = 10.0 + base * 1.5 + ts as f64 * 0.7;
            let bid = 20.0 + base * 1.1 + ts as f64 * 0.5;

            let bar = bar_event(ts, instrument, close, 100.0 + base);
            strict_engine
                .on_event(&bar)
                .expect("strict bar event should succeed");
            force_engine
                .on_event(&bar)
                .expect("force bar event should succeed");

            // ts=2 and ts=5 intentionally miss one instrument quote -> not ready for strict.
            let skip_quote = (ts == 2 && instrument == 2) || (ts == 5 && instrument == 1);
            if !skip_quote {
                let quote = quote_event(ts, instrument, bid, bid + 0.01);
                strict_engine
                    .on_event(&quote)
                    .expect("strict quote event should succeed");
                force_engine
                    .on_event(&quote)
                    .expect("force quote event should succeed");
            }
        }

        let force_frame = force_engine
            .advance(ts)
            .expect("force advance should always succeed");
        let strict_result = strict_engine.advance(ts);
        let strict_not_ready_tick = ts == 2 || ts == 5;
        if strict_not_ready_tick {
            assert!(
                matches!(strict_result, Err(EngineError::NotReady { ts_ns }) if ts_ns == ts),
                "strict should reject partial cross-source tick ts={ts}, got {strict_result:?}"
            );
            assert!(force_frame
                .quality_flags
                .iter()
                .all(|flag| { (flag & QUALITY_FORCED_ADVANCE) == QUALITY_FORCED_ADVANCE }));
            continue;
        }

        let strict_frame = strict_result.expect("strict advance should succeed when ready");
        for (lhs, rhs) in strict_frame.values.iter().zip(force_frame.values.iter()) {
            assert!(
                approx_eq(*lhs, *rhs),
                "strict/force mismatch on ready tick ts={ts}: lhs={lhs}, rhs={rhs}"
            );
        }
        assert_eq!(strict_frame.valid_mask, force_frame.valid_mask);
        assert_eq!(strict_frame.quality_flags, force_frame.quality_flags);
        assert!(strict_frame
            .quality_flags
            .iter()
            .all(|flag| (flag & QUALITY_FORCED_ADVANCE) == 0));
    }
}

#[test]
fn runtime_alpha101_canonical_subset_executes_and_outputs_finite_values() {
    let planner = SimplePlanner;
    let request = alpha101_canonical_subset_request();
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![101, 202, 303]);
    let catalog = InputFieldCatalog::new(vec![
        "bar.open".to_string(),
        "bar.high".to_string(),
        "bar.low".to_string(),
        "bar.close".to_string(),
        "bar.volume".to_string(),
    ]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    let mut last_frame = None;
    for ts in 1_i64..=24_i64 {
        for (idx, instrument) in [101_u32, 202_u32, 303_u32].into_iter().enumerate() {
            let base = idx as f64 + 1.0;
            let open = 20.0 + ts as f64 * 0.05 + (ts as f64 * 0.37 + base * 1.1).sin() * 2.0;
            let close = open + (ts as f64 * 0.19 + base * 0.7).cos() * 0.3;
            let high = open.max(close) + 0.25 + base * 0.02;
            let low = open.min(close) - 0.22 - base * 0.02;
            let volume =
                110.0 + ts as f64 * 0.4 + (ts as f64 * 0.23 + base * 2.3).cos() * 30.0 + base * 3.0;
            engine
                .on_event(&bar_event_ohlcv(
                    ts, instrument, open, high, low, close, volume,
                ))
                .expect("event should succeed");
        }
        last_frame = Some(engine.advance(ts).expect("advance should succeed"));
    }

    let frame = last_frame.expect("last frame");
    for factor_name in &request.outputs {
        let factor_idx = frame.factor_idx(factor_name).expect("factor idx");
        let mut valid_count = 0usize;
        for instrument_idx in 0..frame.instrument_count {
            if frame.is_valid_at(instrument_idx, factor_idx) {
                let value = frame
                    .value_at(instrument_idx, factor_idx)
                    .expect("factor value should exist");
                assert!(
                    value.is_finite(),
                    "expect finite output for factor `{factor_name}` at instrument {instrument_idx}, got {value}"
                );
                valid_count += 1;
            }
        }
        assert!(
            valid_count > 0,
            "expect at least one valid output for factor `{factor_name}` at final frame"
        );
    }
}

#[test]
fn runtime_phase_b_boolean_and_scale_ops_values_are_expected() {
    let planner = SimplePlanner;
    let request = phase_b_boolean_and_scale_request();
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![101, 102, 103]);
    let catalog = InputFieldCatalog::new(vec![
        "bar.open".to_string(),
        "bar.close".to_string(),
        "bar.volume".to_string(),
    ]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    for (instrument, open, close, volume) in [
        (101_u32, 10.0_f64, 9.0_f64, 100.0_f64),
        (102_u32, 10.0_f64, 12.0_f64, 200.0_f64),
        (103_u32, 10.0_f64, 10.0_f64, 300.0_f64),
    ] {
        engine
            .on_event(&bar_event_ohlcv(
                1, instrument, open, open, open, close, volume,
            ))
            .expect("event should succeed");
    }
    let frame = engine.advance(1).expect("advance should succeed");

    let where_idx = frame.factor_idx("where_cmp").expect("where idx");
    assert!(approx_eq(
        frame.value_at(0, where_idx).expect("inst0 where"),
        100.0
    ));
    assert!(approx_eq(
        frame.value_at(1, where_idx).expect("inst1 where"),
        12.0
    ));
    assert!(approx_eq(
        frame.value_at(2, where_idx).expect("inst2 where"),
        10.0
    ));

    let to_int_idx = frame.factor_idx("to_int_cmp").expect("to_int idx");
    assert!(approx_eq(
        frame.value_at(0, to_int_idx).expect("inst0 to_int"),
        0.0
    ));
    assert!(approx_eq(
        frame.value_at(1, to_int_idx).expect("inst1 to_int"),
        1.0
    ));
    assert!(approx_eq(
        frame.value_at(2, to_int_idx).expect("inst2 to_int"),
        1.0
    ));

    let signed_idx = frame.factor_idx("signed_pow").expect("signed_pow idx");
    assert!(approx_eq(
        frame.value_at(0, signed_idx).expect("inst0 signed"),
        -1.0
    ));
    assert!(approx_eq(
        frame.value_at(1, signed_idx).expect("inst1 signed"),
        4.0
    ));
    assert!(approx_eq(
        frame.value_at(2, signed_idx).expect("inst2 signed"),
        0.0
    ));

    let scale_idx = frame
        .factor_idx("cs_scale_close")
        .expect("cs_scale_close idx");
    assert!(approx_eq(
        frame.value_at(0, scale_idx).expect("inst0 scale"),
        9.0 / 31.0
    ));
    assert!(approx_eq(
        frame.value_at(1, scale_idx).expect("inst1 scale"),
        12.0 / 31.0
    ));
    assert!(approx_eq(
        frame.value_at(2, scale_idx).expect("inst2 scale"),
        10.0 / 31.0
    ));
}

#[test]
fn event_quality_flags_propagate_and_reset_after_advance() {
    let planner = SimplePlanner;
    let request = ts_only_request();
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![1]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    engine
        .on_event(&bar_event_with_flags(
            2_500,
            1,
            11.0,
            101.0,
            QUALITY_REVISION,
        ))
        .expect("event should succeed");
    let frame = engine.advance(2_500).expect("advance should succeed");
    assert_eq!(frame.quality_flags[0] & QUALITY_REVISION, QUALITY_REVISION);

    engine
        .on_event(&bar_event(2_600, 1, 12.0, 102.0))
        .expect("event should succeed");
    let frame = engine.advance(2_600).expect("advance should succeed");
    assert_eq!(frame.quality_flags[0], 0);
}

#[test]
fn data_payload_routes_by_name_and_updates_dependent_nodes() {
    let planner = SimplePlanner;
    let request = FactorRequest {
        exprs: vec!["ts_mean(foo, 2)".to_string()],
        outputs: Vec::new(),
        opts: CompileOptions {
            default_source_kind: SourceKind::Data,
        },
    };
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![7]);
    let catalog = InputFieldCatalog::new(vec!["data.foo".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    engine
        .on_event(&data_event(3_100, 7, vec![("bar", 1.0), ("foo", 10.0)]))
        .expect("event should succeed");
    let frame = engine.advance(3_100).expect("advance should succeed");
    assert!(!frame.is_valid_at(0, 0));

    engine
        .on_event(&data_event(3_200, 7, vec![("foo", 14.0), ("other", 5.0)]))
        .expect("event should succeed");
    let frame = engine.advance(3_200).expect("advance should succeed");
    assert!(frame.is_valid_at(0, 0));
    assert_eq!(frame.value_at(0, 0), Some(12.0));
}

#[test]
fn feature_frame_factor_lookup_behaves_as_expected() {
    let frame = crate::FeatureFrame::new(
        3_000,
        2,
        vec!["f0".to_string(), "f1".to_string()],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![true, true, true, false],
        vec![0, 0],
    );
    assert_eq!(frame.factor_idx("f1"), Some(1));
    assert_eq!(frame.factor_value(1, "f1"), Some(4.0));
    assert_eq!(frame.factor_values("f0"), Some(vec![1.0, 3.0]));
    assert_eq!(frame.factor_values("missing"), None);
}

#[test]
fn load_fails_for_unknown_bar_field_accessor() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["ts_mean(foo, 3)".to_string()]);
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![1]);
    let catalog = InputFieldCatalog::new(vec!["bar.foo".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    let err = engine.load(physical).expect_err("load should fail");
    match err {
        EngineError::UnsupportedFieldAccessor { source_kind, field } => {
            assert_eq!(source_kind, "bar");
            assert_eq!(field, "foo");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn with_advanced_frame_provides_zero_copy_view() {
    let planner = SimplePlanner;
    let request = three_factor_request();
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![101, 202]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    engine
        .on_event(&bar_event(7_000, 101, 10.0, 100.0))
        .expect("event should succeed");
    engine
        .on_event(&bar_event(7_000, 202, 20.0, 200.0))
        .expect("event should succeed");

    let got = engine
        .with_advanced_frame(7_000, |frame| {
            assert_eq!(frame.factor_idx("f2_cs_rank"), Some(2));
            frame.factor_value(1, "f2_cs_rank").unwrap_or(f64::NAN)
        })
        .expect("advance should succeed");
    assert_eq!(got, 1.0);
}

#[test]
fn advance_and_with_advanced_frame_have_parity() {
    let planner = SimplePlanner;
    let request = three_factor_request();
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![11, 22, 33]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine_a = OnlineFactorEngine::default();
    engine_a
        .load(physical.clone())
        .expect("load should succeed");
    let mut engine_b = OnlineFactorEngine::default();
    engine_b.load(physical).expect("load should succeed");

    let events = [
        bar_event(9_000, 11, 10.0, 100.0),
        bar_event(9_000, 22, 20.0, 200.0),
        bar_event(9_000, 33, 30.0, 300.0),
    ];
    for event in &events {
        engine_a.on_event(event).expect("event should succeed");
        engine_b.on_event(event).expect("event should succeed");
    }

    let frame = engine_a.advance(9_000).expect("advance should succeed");
    let borrowed_snapshot = engine_b
        .with_advanced_frame(9_000, |view| {
            (
                view.values.to_vec(),
                view.valid_mask.to_vec(),
                view.quality_flags.to_vec(),
                view.factor_names.to_vec(),
            )
        })
        .expect("advance should succeed");

    assert_eq!(frame.values.len(), borrowed_snapshot.0.len());
    for (lhs, rhs) in frame.values.iter().zip(&borrowed_snapshot.0) {
        assert!(lhs == rhs || (lhs.is_nan() && rhs.is_nan()));
    }
    assert_eq!(frame.valid_mask, borrowed_snapshot.1);
    assert_eq!(frame.quality_flags, borrowed_snapshot.2);
    assert_eq!(frame.factor_names.as_ref(), borrowed_snapshot.3.as_slice());
}

#[test]
fn repeated_advance_same_ts_skips_multi_reexecution_until_new_event() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["cs_rank(close)".to_string()]);
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![11, 22]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    engine
        .on_event(&bar_event(9_500, 11, 10.0, 100.0))
        .expect("event should succeed");
    engine
        .on_event(&bar_event(9_500, 22, 20.0, 200.0))
        .expect("event should succeed");
    engine.advance(9_500).expect("advance should succeed");
    assert_eq!(engine.debug_multi_kernel_exec_count(), 1);

    engine.advance(9_500).expect("advance should succeed");
    assert_eq!(engine.debug_multi_kernel_exec_count(), 1);

    engine
        .on_event(&bar_event(9_500, 11, 30.0, 300.0))
        .expect("event should succeed");
    engine.advance(9_500).expect("advance should succeed");
    assert_eq!(engine.debug_multi_kernel_exec_count(), 2);
}

#[test]
fn advance_into_buffers_matches_advance_and_reuses_capacity() {
    let planner = SimplePlanner;
    let request = three_factor_request();
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![1, 2]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine_a = OnlineFactorEngine::default();
    engine_a
        .load(physical.clone())
        .expect("load should succeed");
    let mut engine_b = OnlineFactorEngine::default();
    engine_b.load(physical).expect("load should succeed");
    let mut buffers = FeatureFrameBuffers::default();

    for event in [
        bar_event(9_800, 1, 10.0, 100.0),
        bar_event(9_800, 2, 20.0, 200.0),
    ] {
        engine_a.on_event(&event).expect("event should succeed");
        engine_b.on_event(&event).expect("event should succeed");
    }
    let frame = engine_a.advance(9_800).expect("advance should succeed");
    let shape = engine_b
        .advance_into_buffers(9_800, &mut buffers)
        .expect("advance_into_buffers should succeed");
    assert_eq!(shape, (frame.instrument_count, frame.factor_count));
    assert_eq!(buffers.values.len(), frame.values.len());
    for (lhs, rhs) in buffers.values.iter().zip(frame.values.iter()) {
        assert!(lhs == rhs || (lhs.is_nan() && rhs.is_nan()));
    }
    assert_eq!(buffers.valid_mask, frame.valid_mask);
    assert_eq!(buffers.quality_flags, frame.quality_flags);

    let cap_values = buffers.values.capacity();
    let cap_valid = buffers.valid_mask.capacity();
    let cap_quality = buffers.quality_flags.capacity();

    for event in [
        bar_event(9_900, 1, 30.0, 300.0),
        bar_event(9_900, 2, 40.0, 400.0),
    ] {
        engine_b.on_event(&event).expect("event should succeed");
    }
    engine_b
        .advance_into_buffers(9_900, &mut buffers)
        .expect("advance_into_buffers should succeed");
    assert_eq!(buffers.values.capacity(), cap_values);
    assert_eq!(buffers.valid_mask.capacity(), cap_valid);
    assert_eq!(buffers.quality_flags.capacity(), cap_quality);
}

#[test]
fn advance_in_place_reuses_owned_feature_frame_allocations() {
    let planner = SimplePlanner;
    let request = three_factor_request();
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![1, 2]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine_a = OnlineFactorEngine::default();
    engine_a
        .load(physical.clone())
        .expect("load should succeed");
    let mut engine_b = OnlineFactorEngine::default();
    engine_b.load(physical).expect("load should succeed");

    let mut reusable = crate::FeatureFrame::new(0, 0, vec![], vec![], vec![], vec![]);

    for event in [
        bar_event(9_810, 1, 10.0, 100.0),
        bar_event(9_810, 2, 20.0, 200.0),
    ] {
        engine_a.on_event(&event).expect("event should succeed");
        engine_b.on_event(&event).expect("event should succeed");
    }
    let expected = engine_a.advance(9_810).expect("advance should succeed");
    engine_b
        .advance_in_place(9_810, &mut reusable)
        .expect("advance_in_place should succeed");
    assert_eq!(reusable.values.len(), expected.values.len());
    for (lhs, rhs) in reusable.values.iter().zip(expected.values.iter()) {
        assert!(lhs == rhs || (lhs.is_nan() && rhs.is_nan()));
    }
    assert_eq!(reusable.valid_mask, expected.valid_mask);
    assert_eq!(reusable.quality_flags, expected.quality_flags);
    assert_eq!(
        reusable.factor_names.as_ref(),
        expected.factor_names.as_ref()
    );
    let cap_values = reusable.values.capacity();
    let cap_valid = reusable.valid_mask.capacity();
    let cap_quality = reusable.quality_flags.capacity();

    for event in [
        bar_event(9_820, 1, 30.0, 300.0),
        bar_event(9_820, 2, 40.0, 400.0),
    ] {
        engine_b.on_event(&event).expect("event should succeed");
    }
    engine_b
        .advance_in_place(9_820, &mut reusable)
        .expect("advance_in_place should succeed");
    assert_eq!(reusable.values.capacity(), cap_values);
    assert_eq!(reusable.valid_mask.capacity(), cap_valid);
    assert_eq!(reusable.quality_flags.capacity(), cap_quality);
}

#[test]
fn cs_rank_scratch_capacity_is_reused_across_advances() {
    let planner = SimplePlanner;
    let request = three_factor_request();
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let instrument_slots: Vec<u32> = (0..64).map(|i| i + 1).collect();
    let universe = Universe::new(instrument_slots.clone());
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    for (idx, slot) in instrument_slots.iter().copied().enumerate() {
        engine
            .on_event(&bar_event(
                10_000,
                slot,
                100.0 + idx as f64,
                1_000.0 + idx as f64,
            ))
            .expect("event should succeed");
    }
    engine.advance(10_000).expect("advance should succeed");
    let cap_after_first = engine
        .debug_cs_rank_scratch_capacity()
        .expect("scratch capacity should exist");

    for (idx, slot) in instrument_slots.iter().copied().enumerate() {
        engine
            .on_event(&bar_event(
                11_000,
                slot,
                200.0 + idx as f64,
                2_000.0 + idx as f64,
            ))
            .expect("event should succeed");
    }
    engine.advance(11_000).expect("advance should succeed");
    let cap_after_second = engine
        .debug_cs_rank_scratch_capacity()
        .expect("scratch capacity should exist");

    assert!(cap_after_first >= instrument_slots.len());
    assert_eq!(cap_after_first, cap_after_second);
}

#[test]
fn scratch_is_not_allocated_when_plan_does_not_need_it() {
    let planner = SimplePlanner;
    let request = ts_only_request();
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![11, 22]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    assert_eq!(engine.debug_cs_rank_scratch_capacity(), None);

    engine
        .on_event(&bar_event(12_000, 11, 10.0, 100.0))
        .expect("event should succeed");
    engine
        .on_event(&bar_event(12_000, 22, 20.0, 200.0))
        .expect("event should succeed");
    engine.advance(12_000).expect("advance should succeed");

    assert_eq!(engine.debug_cs_rank_scratch_capacity(), None);
}

#[test]
fn cs_zscore_uses_tmp_scratch_and_produces_expected_values() {
    let planner = SimplePlanner;
    let request = cs_zscore_request();
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.node_count, 1);
    let universe = Universe::new(vec![101, 202, 303]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    assert_eq!(engine.debug_cs_rank_scratch_capacity(), None);
    assert!(
        engine
            .debug_tmp_f64_scratch_capacity()
            .expect("tmp_f64 scratch should exist")
            >= 3
    );
    assert!(
        engine
            .debug_tmp_usize_scratch_capacity()
            .expect("tmp_usize scratch should exist")
            >= 3
    );

    engine
        .on_event(&bar_event(13_000, 101, 10.0, 100.0))
        .expect("event should succeed");
    engine
        .on_event(&bar_event(13_000, 202, 20.0, 200.0))
        .expect("event should succeed");
    engine
        .on_event(&bar_event(13_000, 303, 30.0, 300.0))
        .expect("event should succeed");
    let frame = engine.advance(13_000).expect("advance should succeed");

    assert_eq!(frame.factor_names[0], "f0_cs_zscore");
    let tol = 1e-12;
    assert!((frame.value_at(0, 0).expect("v0") + 1.224_744_871_391_589).abs() < tol);
    assert!(frame.value_at(1, 0).expect("v1").abs() < tol);
    assert!((frame.value_at(2, 0).expect("v2") - 1.224_744_871_391_589).abs() < tol);
}

#[test]
fn ts_rank_uses_tmp_f64_scratch_and_matches_expected_values() {
    let planner = SimplePlanner;
    let request = ts_rank_request();
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.node_count, 1);
    let universe = Universe::new(vec![42]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    assert_eq!(engine.debug_cs_rank_scratch_capacity(), None);
    assert_eq!(engine.debug_tmp_usize_scratch_capacity(), None);
    assert!(
        engine
            .debug_tmp_f64_scratch_capacity()
            .expect("tmp_f64 scratch should exist")
            >= 1
    );

    engine
        .on_event(&bar_event(14_000, 42, 2.0, 100.0))
        .expect("event should succeed");
    let frame = engine.advance(14_000).expect("advance should succeed");
    assert_eq!(frame.factor_names[0], "f0_ts_rank");
    assert!(!frame.is_valid_at(0, 0));
    assert!(frame.value_at(0, 0).expect("value should exist").is_nan());

    engine
        .on_event(&bar_event(15_000, 42, 1.0, 100.0))
        .expect("event should succeed");
    let frame = engine.advance(15_000).expect("advance should succeed");
    assert!(!frame.is_valid_at(0, 0));
    assert!(frame.value_at(0, 0).expect("value should exist").is_nan());

    engine
        .on_event(&bar_event(16_000, 42, 3.0, 100.0))
        .expect("event should succeed");
    let frame = engine.advance(16_000).expect("advance should succeed");
    assert!(frame.is_valid_at(0, 0));
    assert_eq!(frame.value_at(0, 0), Some(1.0));

    engine
        .on_event(&bar_event(17_000, 42, 2.0, 100.0))
        .expect("event should succeed");
    let frame = engine.advance(17_000).expect("advance should succeed");
    assert!(frame.is_valid_at(0, 0));
    assert_eq!(frame.value_at(0, 0), Some(0.5));
    assert!(
        engine
            .debug_tmp_f64_scratch_capacity()
            .expect("tmp_f64 scratch should exist")
            >= 3
    );
}

#[test]
fn runtime_wave1_new_ops_values_are_expected() {
    let planner = SimplePlanner;
    let (logical, _) = planner
        .compile(&wave1_new_ops_request())
        .expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    engine
        .on_event(&bar_event(1, 101, 2.0, 5.0))
        .expect("event should succeed");
    let frame = engine.advance(1).expect("advance should succeed");
    let sum3_idx = frame.factor_idx("sum3").expect("sum3 idx");
    let sum3_maxcv_idx = frame.factor_idx("sum3_maxcv").expect("sum3_maxcv idx");
    assert!(!frame.is_valid_at(0, sum3_idx));
    assert_eq!(frame.factor_value(0, "pow_close2"), Some(4.0));
    assert_eq!(frame.factor_value(0, "min_cv"), Some(2.0));
    assert_eq!(frame.factor_value(0, "max_cv"), Some(5.0));
    assert!(!frame.is_valid_at(0, sum3_maxcv_idx));

    engine
        .on_event(&bar_event(2, 101, 4.0, 3.0))
        .expect("event should succeed");
    let frame = engine.advance(2).expect("advance should succeed");
    assert!(!frame.is_valid_at(0, sum3_idx));
    assert_eq!(frame.factor_value(0, "pow_close2"), Some(16.0));
    assert_eq!(frame.factor_value(0, "min_cv"), Some(3.0));
    assert_eq!(frame.factor_value(0, "max_cv"), Some(4.0));
    assert!(!frame.is_valid_at(0, sum3_maxcv_idx));

    engine
        .on_event(&bar_event(3, 101, 1.0, 7.0))
        .expect("event should succeed");
    let frame = engine.advance(3).expect("advance should succeed");
    assert_eq!(frame.factor_value(0, "sum3"), Some(7.0));
    assert_eq!(frame.factor_value(0, "min3"), Some(1.0));
    assert_eq!(frame.factor_value(0, "max3"), Some(4.0));
    assert_eq!(frame.factor_value(0, "pow_close2"), Some(1.0));
    assert_eq!(frame.factor_value(0, "min_cv"), Some(1.0));
    assert_eq!(frame.factor_value(0, "max_cv"), Some(7.0));
    assert_eq!(frame.factor_value(0, "sum3_maxcv"), Some(16.0));
}

#[test]
fn runtime_wave2_ts_stats_ops_values_are_expected() {
    let planner = SimplePlanner;
    let (logical, _) = planner
        .compile(&wave2_ts_stats_request())
        .expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    engine
        .on_event(&bar_event(1, 101, 2.0, 5.0))
        .expect("event should succeed");
    let frame = engine.advance(1).expect("advance should succeed");
    let lag_idx = frame.factor_idx("lag1").expect("lag1 idx");
    assert!(!frame.is_valid_at(0, lag_idx));

    engine
        .on_event(&bar_event(2, 101, 4.0, 3.0))
        .expect("event should succeed");
    let frame = engine.advance(2).expect("advance should succeed");
    assert_eq!(frame.factor_value(0, "lag1"), Some(2.0));

    engine
        .on_event(&bar_event(3, 101, 1.0, 7.0))
        .expect("event should succeed");
    let frame = engine.advance(3).expect("advance should succeed");
    assert_eq!(frame.factor_value(0, "lag1"), Some(4.0));
    assert_eq!(frame.factor_value(0, "cov3"), Some(-3.0));
    assert!(!frame.is_valid_at(0, frame.factor_idx("z_lag1_3").expect("z_lag1_3 idx")));

    engine
        .on_event(&bar_event(4, 101, 5.0, 2.0))
        .expect("event should succeed");
    let frame = engine.advance(4).expect("advance should succeed");
    assert_eq!(frame.factor_value(0, "lag1"), Some(1.0));
    assert_eq!(frame.factor_value(0, "cov3"), Some(-5.5));
    assert!((frame.factor_value(0, "z3").expect("z3") - 0.800_640_769_025_435_7).abs() < 1e-12);
    assert!(
        (frame.factor_value(0, "z_lag1_3").expect("z_lag1_3") + 0.872_871_560_943_969_4).abs()
            < 1e-12
    );
}

#[test]
fn runtime_wave3_var_beta_ops_values_are_expected() {
    let planner = SimplePlanner;
    let (logical, _) = planner
        .compile(&wave3_var_beta_request())
        .expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let samples = [
        (1_i64, 2.0_f64, 5.0_f64),
        (2, 4.0, 3.0),
        (3, 1.0, 7.0),
        (4, 5.0, 2.0),
        (5, 3.0, 6.0),
    ];
    for (ts, close, volume) in samples {
        engine
            .on_event(&bar_event(ts, 101, close, volume))
            .expect("event should succeed");
        let frame = engine.advance(ts).expect("advance should succeed");
        if ts < 4 {
            assert!(!frame.is_valid_at(0, frame.factor_idx("var4").expect("var4 idx")));
            continue;
        }
        if ts == 4 {
            assert!(
                (frame.factor_value(0, "var4").expect("var4") - 3.333_333_333_333_333_5).abs()
                    < 1e-12
            );
            assert!(
                (frame.factor_value(0, "beta_cv_4").expect("beta_cv_4") + 0.813_559_322_033_898_4)
                    .abs()
                    < 1e-12
            );
            assert!((frame.factor_value(0, "beta_vc_4").expect("beta_vc_4") + 1.2).abs() < 1e-12);
            assert!(
                !frame.is_valid_at(0, frame.factor_idx("var_mean3_3").expect("var_mean3_3 idx"))
            );
            continue;
        }
        assert!(
            (frame.factor_value(0, "var4").expect("var4") - 2.916_666_666_666_666_5).abs() < 1e-12
        );
        assert!(
            (frame.factor_value(0, "beta_cv_4").expect("beta_cv_4") + 0.676_470_588_235_294_2)
                .abs()
                < 1e-12
        );
        assert!(
            (frame.factor_value(0, "beta_vc_4").expect("beta_vc_4") + 1.314_285_714_285_714_3)
                .abs()
                < 1e-12
        );
        assert!(
            (frame.factor_value(0, "var_mean3_3").expect("var_mean3_3") - 0.259_259_259_259_259_24)
                .abs()
                < 1e-12
        );
    }
}

#[test]
fn runtime_wave4_ewm_ops_values_are_expected() {
    let planner = SimplePlanner;
    let (logical, _) = planner
        .compile(&wave4_ewm_request())
        .expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let samples = [
        (1_i64, 2.0_f64, 5.0_f64),
        (2, 4.0, 3.0),
        (3, 1.0, 7.0),
        (4, 5.0, 2.0),
    ];
    for (ts, close, volume) in samples {
        engine
            .on_event(&bar_event(ts, 101, close, volume))
            .expect("event should succeed");
        let frame = engine.advance(ts).expect("advance should succeed");
        if ts < 3 {
            assert!(!frame.is_valid_at(0, frame.factor_idx("ewm_mean3").expect("ewm_mean3 idx")));
            continue;
        }
        if ts == 3 {
            assert!((frame.factor_value(0, "ewm_mean3").expect("ewm_mean3") - 2.0).abs() < 1e-12);
            assert!(
                (frame.factor_value(0, "ewm_var3").expect("ewm_var3") - 1.714_285_714_285_714_4)
                    .abs()
                    < 1e-12
            );
            assert!(
                (frame.factor_value(0, "ewm_cov3").expect("ewm_cov3") + 2.285_714_285_714_286_5)
                    .abs()
                    < 1e-12
            );
            assert!(!frame.is_valid_at(
                0,
                frame
                    .factor_idx("ewm_mean_lag1_3")
                    .expect("ewm_mean_lag1_3 idx")
            ));
            assert!(!frame.is_valid_at(
                0,
                frame
                    .factor_idx("ewm_var_lag1_3")
                    .expect("ewm_var_lag1_3 idx")
            ));
            continue;
        }
        assert!(
            (frame.factor_value(0, "ewm_mean3").expect("ewm_mean3") - 3.714_285_714_285_714_4)
                .abs()
                < 1e-12
        );
        assert!(
            (frame.factor_value(0, "ewm_var3").expect("ewm_var3") - 3.061_224_489_795_917_8).abs()
                < 1e-12
        );
        assert!(
            (frame.factor_value(0, "ewm_cov3").expect("ewm_cov3") + 3.836_734_693_877_550_4).abs()
                < 1e-12
        );
        assert!(
            (frame
                .factor_value(0, "ewm_mean_lag1_3")
                .expect("ewm_mean_lag1_3")
                - 2.0)
                .abs()
                < 1e-12
        );
        assert!(
            (frame
                .factor_value(0, "ewm_var_lag1_3")
                .expect("ewm_var_lag1_3")
                - 1.714_285_714_285_714_4)
                .abs()
                < 1e-12
        );
    }
}

#[test]
fn runtime_wave5_quantile_ops_values_are_expected() {
    let planner = SimplePlanner;
    let (logical, _) = planner
        .compile(&wave5_quantile_request())
        .expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let samples = [
        (1_i64, 2.0_f64, 5.0_f64),
        (2, 4.0, 3.0),
        (3, 1.0, 7.0),
        (4, 5.0, 2.0),
        (5, 3.0, 6.0),
        (6, 6.0, 1.0),
    ];
    for (ts, close, volume) in samples {
        engine
            .on_event(&bar_event(ts, 101, close, volume))
            .expect("event should succeed");
        let frame = engine.advance(ts).expect("advance should succeed");
        if ts < 4 {
            assert!(!frame.is_valid_at(0, frame.factor_idx("q25_4").expect("q25_4 idx")));
            continue;
        }
        if ts == 4 {
            assert!((frame.factor_value(0, "q25_4").expect("q25_4") - 1.75).abs() < 1e-12);
            assert!(!frame.is_valid_at(0, frame.factor_idx("q50_lag1_4").expect("q50_lag1_4 idx")));
            assert!(!frame.is_valid_at(0, frame.factor_idx("q75_lag1_4").expect("q75_lag1_4 idx")));
            continue;
        }
        if ts == 5 {
            assert!((frame.factor_value(0, "q25_4").expect("q25_4") - 2.5).abs() < 1e-12);
            assert!((frame.factor_value(0, "q50_lag1_4").expect("q50_lag1_4") - 3.0).abs() < 1e-12);
            assert!(
                (frame.factor_value(0, "q75_lag1_4").expect("q75_lag1_4") - 4.25).abs() < 1e-12
            );
            continue;
        }
        assert!((frame.factor_value(0, "q25_4").expect("q25_4") - 2.5).abs() < 1e-12);
        assert!((frame.factor_value(0, "q50_lag1_4").expect("q50_lag1_4") - 3.5).abs() < 1e-12);
        assert!((frame.factor_value(0, "q75_lag1_4").expect("q75_lag1_4") - 4.25).abs() < 1e-12);
    }
}

#[test]
fn runtime_wave6_argext_ops_values_are_expected() {
    let planner = SimplePlanner;
    let (logical, _) = planner
        .compile(&wave6_argext_request())
        .expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let samples = [
        (1_i64, 2.0_f64, 5.0_f64),
        (2, 4.0, 3.0),
        (3, 1.0, 7.0),
        (4, 5.0, 2.0),
        (5, 3.0, 6.0),
        (6, 6.0, 1.0),
    ];
    for (ts, close, volume) in samples {
        engine
            .on_event(&bar_event(ts, 101, close, volume))
            .expect("event should succeed");
        let frame = engine.advance(ts).expect("advance should succeed");
        if ts < 4 {
            assert!(!frame.is_valid_at(0, frame.factor_idx("argmax4").expect("argmax4 idx")));
            continue;
        }
        if ts == 4 {
            assert_eq!(frame.factor_value(0, "argmax4"), Some(0.0));
            assert_eq!(frame.factor_value(0, "argmin4"), Some(1.0));
            assert!(!frame.is_valid_at(
                0,
                frame
                    .factor_idx("argmax_lag1_4")
                    .expect("argmax_lag1_4 idx")
            ));
            continue;
        }
        if ts == 5 {
            assert_eq!(frame.factor_value(0, "argmax4"), Some(1.0));
            assert_eq!(frame.factor_value(0, "argmin4"), Some(2.0));
            assert_eq!(frame.factor_value(0, "argmax_lag1_4"), Some(0.0));
            continue;
        }
        assert_eq!(frame.factor_value(0, "argmax4"), Some(0.0));
        assert_eq!(frame.factor_value(0, "argmin4"), Some(3.0));
        assert_eq!(frame.factor_value(0, "argmax_lag1_4"), Some(1.0));
    }
}

#[test]
fn runtime_wave7_elem_unary_ops_values_are_expected() {
    let planner = SimplePlanner;
    let (logical, _) = planner
        .compile(&wave7_elem_unary_request())
        .expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let samples = [
        (1_i64, 2.0_f64, 5.0_f64),
        (2, 4.0, 3.0),
        (3, 1.0, 7.0),
        (4, 5.0, 2.0),
        (5, 3.0, 6.0),
    ];
    for (ts, close, volume) in samples {
        engine
            .on_event(&bar_event(ts, 101, close, volume))
            .expect("event should succeed");
        let frame = engine.advance(ts).expect("advance should succeed");
        if ts == 1 {
            assert!(!frame.is_valid_at(0, frame.factor_idx("abs_delta1").expect("abs_delta1 idx")));
        } else if ts == 2 {
            assert_eq!(frame.factor_value(0, "abs_delta1"), Some(2.0));
        } else if ts == 3 {
            assert_eq!(frame.factor_value(0, "abs_delta1"), Some(3.0));
            assert_eq!(frame.factor_value(0, "sign_dev3"), Some(-1.0));
        } else if ts == 4 {
            assert_eq!(frame.factor_value(0, "abs_delta1"), Some(4.0));
            assert_eq!(frame.factor_value(0, "sign_dev3"), Some(1.0));
        } else {
            let abs_delta = frame.factor_value(0, "abs_delta1").expect("abs_delta1");
            assert!(
                (abs_delta - 2.0).abs() < 1e-12,
                "abs_delta1 at ts={ts} was {abs_delta}"
            );
            assert_eq!(frame.factor_value(0, "sign_dev3"), Some(1.0));
        }
        assert!(
            (frame.factor_value(0, "log_close").expect("log_close") - close.ln()).abs() < 1e-12
        );
    }
}

#[test]
fn runtime_wave8_elem_extended_ops_values_are_expected() {
    let planner = SimplePlanner;
    let (logical, _) = planner
        .compile(&wave8_elem_extended_request())
        .expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let samples = [
        (1_i64, 2.0_f64, 5.0_f64),
        (2, 4.0, 3.0),
        (3, 1.0, 7.0),
        (4, 5.0, 2.0),
        (5, 3.0, 6.0),
    ];
    for (ts, close, volume) in samples {
        engine
            .on_event(&bar_event(ts, 101, close, volume))
            .expect("event should succeed");
        let frame = engine.advance(ts).expect("advance should succeed");
        if ts == 1 {
            assert!(!frame.is_valid_at(0, frame.factor_idx("exp_lag1").expect("exp_lag1 idx")));
            assert!(!frame.is_valid_at(
                0,
                frame
                    .factor_idx("sqrt_abs_delta1")
                    .expect("sqrt_abs_delta1 idx")
            ));
        } else if ts == 2 {
            assert!(
                (frame.factor_value(0, "exp_lag1").expect("exp_lag1")
                    - std::f64::consts::E.powf(2.0))
                .abs()
                    < 1e-12
            );
            assert!(
                (frame
                    .factor_value(0, "sqrt_abs_delta1")
                    .expect("sqrt_abs_delta1")
                    - 2.0_f64.sqrt())
                .abs()
                    < 1e-12
            );
        } else if ts == 3 {
            assert!(
                (frame.factor_value(0, "exp_lag1").expect("exp_lag1")
                    - std::f64::consts::E.powf(4.0))
                .abs()
                    < 1e-10
            );
            assert!(
                (frame
                    .factor_value(0, "sqrt_abs_delta1")
                    .expect("sqrt_abs_delta1")
                    - 3.0_f64.sqrt())
                .abs()
                    < 1e-12
            );
        } else if ts == 4 {
            assert!(
                (frame.factor_value(0, "exp_lag1").expect("exp_lag1")
                    - std::f64::consts::E.powf(1.0))
                .abs()
                    < 1e-12
            );
            assert!(
                (frame
                    .factor_value(0, "sqrt_abs_delta1")
                    .expect("sqrt_abs_delta1")
                    - 2.0)
                    .abs()
                    < 1e-12
            );
        } else {
            assert!(
                (frame.factor_value(0, "exp_lag1").expect("exp_lag1")
                    - std::f64::consts::E.powf(5.0))
                .abs()
                    < 1e-9
            );
            assert!(
                (frame
                    .factor_value(0, "sqrt_abs_delta1")
                    .expect("sqrt_abs_delta1")
                    - 2.0_f64.sqrt())
                .abs()
                    < 1e-12
            );
        }
        let expected_clip = close.clamp(2.5, 4.5);
        assert!(
            (frame.factor_value(0, "clip_close").expect("clip_close") - expected_clip).abs()
                < 1e-12
        );
    }
}

#[test]
fn runtime_wave9_elem_conditional_ops_values_are_expected() {
    let planner = SimplePlanner;
    let (logical, _) = planner
        .compile(&wave9_elem_conditional_request())
        .expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let samples = [
        (1_i64, 2.0_f64, 10.0_f64),
        (2, 4.0, 11.0),
        (3, 1.0, 12.0),
        (4, 5.0, 13.0),
        (5, 3.0, 14.0),
    ];
    for (ts, close, volume) in samples {
        engine
            .on_event(&bar_event(ts, 101, close, volume))
            .expect("event should succeed");
        let frame = engine.advance(ts).expect("advance should succeed");
        assert_eq!(frame.factor_value(0, "where_one"), Some(close));
        assert_eq!(frame.factor_value(0, "fillna_div0"), Some(0.0));
        assert_eq!(frame.factor_value(0, "where_zero"), Some(volume));
    }
}

#[test]
fn runtime_wave10_decay_linear_values_are_expected() {
    let planner = SimplePlanner;
    let (logical, _) = planner
        .compile(&wave10_decay_linear_request())
        .expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let closes = [2.0_f64, 4.0, 1.0, 5.0, 3.0];
    for (idx, close) in closes.into_iter().enumerate() {
        let ts = (idx + 1) as i64;
        engine
            .on_event(&bar_event(ts, 101, close, 10.0 + ts as f64))
            .expect("event should succeed");
        let frame = engine.advance(ts).expect("advance should succeed");

        if ts < 4 {
            assert!(!frame.is_valid_at(0, frame.factor_idx("decay4").expect("decay4 idx")));
            continue;
        }

        let expected = match ts {
            4 => (4.0 * 5.0 + 3.0 * 1.0 + 2.0 * 4.0 + 1.0 * 2.0) / 10.0,
            5 => (4.0 * 3.0 + 3.0 * 5.0 + 2.0 * 1.0 + 1.0 * 4.0) / 10.0,
            _ => unreachable!(),
        };
        assert!((frame.factor_value(0, "decay4").expect("decay4") - expected).abs() < 1e-12);
        if ts == 4 {
            assert!(
                !frame.is_valid_at(0, frame.factor_idx("decay4_lag1").expect("decay4_lag1 idx"))
            );
        } else {
            let expected_lag1 = (4.0 * 5.0 + 3.0 * 1.0 + 2.0 * 4.0 + 1.0 * 2.0) / 10.0;
            assert!(
                (frame.factor_value(0, "decay4_lag1").expect("decay4_lag1") - expected_lag1).abs()
                    < 1e-12
            );
        }
    }
}

#[test]
fn runtime_wave11_product_mad_values_are_expected() {
    let planner = SimplePlanner;
    let (logical, _) = planner
        .compile(&wave11_product_mad_request())
        .expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let closes = [2.0_f64, 4.0, 1.0, 5.0, 3.0];
    for (idx, close) in closes.into_iter().enumerate() {
        let ts = (idx + 1) as i64;
        engine
            .on_event(&bar_event(ts, 101, close, 10.0 + ts as f64))
            .expect("event should succeed");
        let frame = engine.advance(ts).expect("advance should succeed");

        if ts < 4 {
            assert!(!frame.is_valid_at(0, frame.factor_idx("product4").expect("product4 idx")));
            assert!(!frame.is_valid_at(0, frame.factor_idx("mad4").expect("mad4 idx")));
            continue;
        }

        let (expected_product, expected_mad) = match ts {
            4 => (40.0, 1.5),
            5 => (60.0, 1.25),
            _ => unreachable!(),
        };
        assert!(
            (frame.factor_value(0, "product4").expect("product4") - expected_product).abs() < 1e-12
        );
        assert!((frame.factor_value(0, "mad4").expect("mad4") - expected_mad).abs() < 1e-12);

        if ts == 4 {
            assert!(!frame.is_valid_at(0, frame.factor_idx("mad4_lag1").expect("mad4_lag1 idx")));
        } else {
            assert!((frame.factor_value(0, "mad4_lag1").expect("mad4_lag1") - 1.5).abs() < 1e-12);
        }
    }
}

#[test]
fn runtime_wave12_cs_preprocess_values_are_expected() {
    let planner = SimplePlanner;
    let (logical, _) = planner
        .compile(&wave12_cs_preprocess_request())
        .expect("compile should succeed");
    let universe = Universe::new(vec![101, 102, 103]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    // ts=1: [1, 2, NaN]
    engine
        .on_event(&bar_event(1, 101, 1.0, 10.0))
        .expect("event should succeed");
    engine
        .on_event(&bar_event(1, 102, 2.0, 10.0))
        .expect("event should succeed");
    engine
        .on_event(&bar_event(1, 103, f64::NAN, 10.0))
        .expect("event should succeed");
    let frame = engine.advance(1).expect("advance should succeed");

    assert_eq!(frame.factor_value(0, "center_close"), Some(-0.5));
    assert_eq!(frame.factor_value(1, "center_close"), Some(0.5));
    assert!(!frame.is_valid_at(
        2,
        frame.factor_idx("center_close").expect("center_close idx")
    ));

    assert_eq!(frame.factor_value(0, "norm_close"), Some(0.0));
    assert_eq!(frame.factor_value(1, "norm_close"), Some(1.0));
    assert!(!frame.is_valid_at(2, frame.factor_idx("norm_close").expect("norm_close idx")));

    assert_eq!(frame.factor_value(0, "fillna_close"), Some(1.0));
    assert_eq!(frame.factor_value(1, "fillna_close"), Some(2.0));
    assert_eq!(frame.factor_value(2, "fillna_close"), Some(1.5));
}

#[test]
fn runtime_wave13_cs_clip_values_are_expected() {
    let planner = SimplePlanner;
    let (logical, _) = planner
        .compile(&wave13_cs_clip_request())
        .expect("compile should succeed");
    let universe = Universe::new(vec![101, 102, 103]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    engine
        .on_event(&bar_event(1, 101, 1.0, 10.0))
        .expect("event should succeed");
    engine
        .on_event(&bar_event(1, 102, 2.0, 10.0))
        .expect("event should succeed");
    engine
        .on_event(&bar_event(1, 103, 100.0, 10.0))
        .expect("event should succeed");
    let frame = engine.advance(1).expect("advance should succeed");

    assert!((frame.factor_value(0, "wins_close").expect("wins0") - 1.2).abs() < 1e-12);
    assert!((frame.factor_value(1, "wins_close").expect("wins1") - 2.0).abs() < 1e-12);
    assert!((frame.factor_value(2, "wins_close").expect("wins2") - 80.4).abs() < 1e-12);

    assert!((frame.factor_value(0, "pct_close").expect("pct0") - 1.0).abs() < 1e-12);
    assert!((frame.factor_value(1, "pct_close").expect("pct1") - 0.0).abs() < 1e-12);
    assert!((frame.factor_value(2, "pct_close").expect("pct2") - 100.0).abs() < 1e-12);
}

#[test]
fn runtime_wave14_cs_neutralize_values_are_expected() {
    let planner = SimplePlanner;
    let (logical, _) = planner
        .compile(&wave14_cs_neutralize_request())
        .expect("compile should succeed");
    let universe = Universe::new(vec![101, 102, 103]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    engine
        .on_event(&bar_event(1, 101, 2.0, 1.0))
        .expect("event should succeed");
    engine
        .on_event(&bar_event(1, 102, 2.0, 2.0))
        .expect("event should succeed");
    engine
        .on_event(&bar_event(1, 103, 8.0, 3.0))
        .expect("event should succeed");
    let frame = engine.advance(1).expect("advance should succeed");

    assert!((frame.factor_value(0, "neutral_ols").expect("neutral_ols0") - 1.0).abs() < 1e-12);
    assert!((frame.factor_value(1, "neutral_ols").expect("neutral_ols1") + 2.0).abs() < 1e-12);
    assert!((frame.factor_value(2, "neutral_ols").expect("neutral_ols2") - 1.0).abs() < 1e-12);

    let multi0 = frame
        .factor_value(0, "neutral_ols_multi")
        .expect("neutral_ols_multi0");
    let multi1 = frame
        .factor_value(1, "neutral_ols_multi")
        .expect("neutral_ols_multi1");
    let multi2 = frame
        .factor_value(2, "neutral_ols_multi")
        .expect("neutral_ols_multi2");
    assert!(multi0.abs() < 1e-10);
    assert!(multi1.abs() < 1e-10);
    assert!(multi2.abs() < 1e-10);
}

#[test]
fn runtime_wave15_cs_neutralize_multi3_and_alias_values_are_expected() {
    let planner = SimplePlanner;
    let (logical, _) = planner
        .compile(&wave15_cs_neutralize_multi3_request())
        .expect("compile should succeed");
    let universe = Universe::new(vec![101, 102, 103, 104]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    for (idx, instrument_slot) in [101_u32, 102_u32, 103_u32, 104_u32].into_iter().enumerate() {
        let v = (idx + 1) as f64;
        let close = 1.0 + 2.0 * v + 3.0 * v * v + 4.0 * v * v * v;
        engine
            .on_event(&bar_event(1, instrument_slot, close, v))
            .expect("event should succeed");
    }

    let frame = engine.advance(1).expect("advance should succeed");
    let alias_vals = frame
        .factor_values("neutral_alias")
        .expect("neutral_alias output");
    let alias_mean = alias_vals.iter().sum::<f64>() / alias_vals.len() as f64;
    assert!(alias_mean.abs() < 1e-10);

    for instrument_idx in 0..4 {
        let v = frame
            .factor_value(instrument_idx, "neutral_ols_multi3")
            .expect("neutral_ols_multi3");
        assert!(v.abs() < 1e-8);
    }
}

#[test]
fn runtime_cs_neutralize_supports_group_weights_and_standardize() {
    let planner = SimplePlanner;
    let request = FactorRequest {
        exprs: vec!["cs_neutralize(close, group=volume, weights=open, standardize=1)".to_string()],
        outputs: vec!["neutral_gws".to_string()],
        opts: CompileOptions::default(),
    };
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![101, 102, 103, 104]);
    let catalog = InputFieldCatalog::new(vec![
        "bar.open".to_string(),
        "bar.close".to_string(),
        "bar.volume".to_string(),
    ]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let rows = [
        (101_u32, 1.0_f64, 1.0_f64, 0.0_f64),
        (102_u32, 3.0_f64, 3.0_f64, 0.0_f64),
        (103_u32, 1.0_f64, 10.0_f64, 1.0_f64),
        (104_u32, 1.0_f64, 14.0_f64, 1.0_f64),
    ];
    for (instrument_slot, open, close, volume) in rows {
        engine
            .on_event(&EventEnvelope {
                ts_event_ns: 1,
                ts_init_ns: 1,
                seq: 1,
                source_slot: 0,
                instrument_slot,
                quality_flags: 0,
                payload: Payload::Bar(BarLite {
                    open,
                    high: close,
                    low: close,
                    close,
                    volume,
                }),
            })
            .expect("event should succeed");
    }

    let frame = engine.advance(1).expect("advance should succeed");
    assert!((frame.factor_value(0, "neutral_gws").expect("v0") + 1.7320508075688772).abs() < 1e-9);
    assert!((frame.factor_value(1, "neutral_gws").expect("v1") - 0.5773502691896257).abs() < 1e-9);
    assert!((frame.factor_value(2, "neutral_gws").expect("v2") + 1.0).abs() < 1e-9);
    assert!((frame.factor_value(3, "neutral_gws").expect("v3") - 1.0).abs() < 1e-9);
}

#[test]
fn runtime_cs_neutralize_ols_supports_group_semantics() {
    let planner = SimplePlanner;
    let request = FactorRequest {
        exprs: vec!["cs_neutralize_ols(close, open, group=volume)".to_string()],
        outputs: vec!["neutral_ols_group".to_string()],
        opts: CompileOptions::default(),
    };
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![201, 202, 203, 204]);
    let catalog = InputFieldCatalog::new(vec![
        "bar.open".to_string(),
        "bar.close".to_string(),
        "bar.volume".to_string(),
    ]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");
    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    let rows = [
        (201_u32, 1.0_f64, 3.0_f64, 0.0_f64),
        (202_u32, 2.0_f64, 5.0_f64, 0.0_f64),
        (203_u32, 1.0_f64, 12.0_f64, 1.0_f64),
        (204_u32, 2.0_f64, 14.0_f64, 1.0_f64),
    ];
    for (instrument_slot, open, close, volume) in rows {
        engine
            .on_event(&EventEnvelope {
                ts_event_ns: 1,
                ts_init_ns: 1,
                seq: 1,
                source_slot: 0,
                instrument_slot,
                quality_flags: 0,
                payload: Payload::Bar(BarLite {
                    open,
                    high: close,
                    low: close,
                    close,
                    volume,
                }),
            })
            .expect("event should succeed");
    }

    let frame = engine.advance(1).expect("advance should succeed");
    for idx in 0..4 {
        let value = frame
            .factor_value(idx, "neutral_ols_group")
            .expect("neutral_ols_group");
        assert!(value.abs() < 1e-9);
    }
}

#[test]
#[ignore = "micro-benchmark; run manually with -- --ignored --nocapture"]
fn benchmark_cross_source_policy_strict_vs_force_sparse_quotes() {
    let planner = SimplePlanner;
    let request = cross_source_policy_request();
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new((1_u32..=16_u32).collect::<Vec<_>>());
    let catalog = InputFieldCatalog::new(vec![
        "bar.close".to_string(),
        "quote_tick.bid_price".to_string(),
    ]);
    let strict_plan = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("strict bind should succeed");
    let force_plan = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::ForceWithLast)
        .expect("force bind should succeed");

    let ticks = 5_000_i64;
    let expected_not_ready = (ticks / 3) as usize;
    let mut strict_engine = OnlineFactorEngine::default();
    strict_engine
        .load(strict_plan)
        .expect("strict load should succeed");
    let mut force_engine = OnlineFactorEngine::default();
    force_engine
        .load(force_plan)
        .expect("force load should succeed");

    let t0 = Instant::now();
    let mut strict_ok = 0usize;
    let mut strict_not_ready = 0usize;
    for ts in 1_i64..=ticks {
        for instrument in 1_u32..=16_u32 {
            let base = instrument as f64;
            strict_engine
                .on_event(&bar_event(
                    ts,
                    instrument,
                    11.0 + base + ts as f64 * 0.01,
                    100.0,
                ))
                .expect("strict bar event should succeed");
            let skip_quote = instrument == 1_u32 && ts % 3 == 0;
            if !skip_quote {
                let bid = 21.0 + base + ts as f64 * 0.02;
                strict_engine
                    .on_event(&quote_event(ts, instrument, bid, bid + 0.01))
                    .expect("strict quote event should succeed");
            }
        }
        match strict_engine.advance(ts) {
            Ok(_) => strict_ok += 1,
            Err(EngineError::NotReady { .. }) => strict_not_ready += 1,
            Err(other) => panic!("unexpected strict advance error: {other:?}"),
        }
    }
    let dt_strict = t0.elapsed();

    let t1 = Instant::now();
    let mut force_ok = 0usize;
    let mut force_forced = 0usize;
    for ts in 1_i64..=ticks {
        for instrument in 1_u32..=16_u32 {
            let base = instrument as f64;
            force_engine
                .on_event(&bar_event(
                    ts,
                    instrument,
                    11.0 + base + ts as f64 * 0.01,
                    100.0,
                ))
                .expect("force bar event should succeed");
            let skip_quote = instrument == 1_u32 && ts % 3 == 0;
            if !skip_quote {
                let bid = 21.0 + base + ts as f64 * 0.02;
                force_engine
                    .on_event(&quote_event(ts, instrument, bid, bid + 0.01))
                    .expect("force quote event should succeed");
            }
        }
        let frame = force_engine
            .advance(ts)
            .expect("force advance should always succeed");
        force_ok += 1;
        if frame
            .quality_flags
            .iter()
            .any(|flag| (flag & QUALITY_FORCED_ADVANCE) == QUALITY_FORCED_ADVANCE)
        {
            force_forced += 1;
        }
    }
    let dt_force = t1.elapsed();

    assert_eq!(strict_not_ready, expected_not_ready);
    assert_eq!(strict_ok, ticks as usize - expected_not_ready);
    assert_eq!(force_ok, ticks as usize);
    assert_eq!(force_forced, expected_not_ready);

    println!(
        "policy benchmark ticks={} instruments={} strict={:?} (ok={}, not_ready={}) force={:?} (ok={}, forced={})",
        ticks, 16, dt_strict, strict_ok, strict_not_ready, dt_force, force_ok, force_forced
    );
}

#[test]
#[ignore = "micro-benchmark; run manually with -- --ignored --nocapture"]
fn benchmark_ts_rank_scratch_reuse_vs_per_call_alloc() {
    let series = synthetic_series(60_000);
    let window = 32usize;

    let t0 = Instant::now();
    let reused = ts_rank_series_with_reuse(&series, window);
    let d_reused = t0.elapsed();

    let t1 = Instant::now();
    let allocated = ts_rank_series_with_alloc(&series, window);
    let d_alloc = t1.elapsed();

    assert_eq!(reused.len(), allocated.len());
    for (lhs, rhs) in reused.iter().zip(allocated.iter()) {
        assert!(lhs == rhs || (lhs.is_nan() && rhs.is_nan()));
    }

    println!(
        "ts_rank(window={window}) reuse={:?} alloc={:?} speedup={:.3}x",
        d_reused,
        d_alloc,
        d_alloc.as_secs_f64() / d_reused.as_secs_f64()
    );
}
