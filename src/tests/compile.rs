use super::*;

#[test]
fn compile_bind_and_ready_gate_flow() {
    let planner = SimplePlanner;
    let request = three_factor_request();
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.node_count, 3);
    assert_eq!(manifest.field_count, 2);
    assert_eq!(manifest.expr_count, 3);
    assert_eq!(manifest.alias_count, 0);

    let universe = Universe::new(vec![101, 202]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    engine
        .on_event(&bar_event(1_000, 101, 10.0, 100.0))
        .expect("event should succeed");
    assert!(!engine.is_graph_ready(1_000));

    engine
        .on_event(&bar_event(1_000, 202, 20.0, 200.0))
        .expect("event should succeed");
    assert!(engine.is_graph_ready(1_000));

    let frame = engine.advance(1_000).expect("advance should succeed");
    assert_eq!(frame.instrument_count, 2);
    assert_eq!(frame.factor_count, 3);
    assert_eq!(frame.factor_names[2], "f2_cs_rank");
    // node order: ts_mean, ts_delta, cs_rank
    assert_eq!(frame.value_at(0, 2), Some(0.0));
    assert_eq!(frame.value_at(1, 2), Some(1.0));
    assert_eq!(frame.factor_value(1, "f2_cs_rank"), Some(1.0));
    assert_eq!(frame.factor_value(0, "missing_factor"), None);
    assert_eq!(frame.factor_values("f2_cs_rank"), Some(vec![0.0, 1.0]));
}

#[test]
fn ready_gate_uses_asof_latest_ts_for_required_cells() {
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
        .on_event(&bar_event(5_000, 101, 10.0, 100.0))
        .expect("event should succeed");
    engine
        .on_event(&bar_event(5_000, 202, 20.0, 200.0))
        .expect("event should succeed");

    assert!(engine.is_graph_ready(4_999));
    assert!(engine.is_graph_ready(5_000));
    assert!(!engine.is_graph_ready(5_001));
}

#[test]
fn compile_rejects_trailing_tokens_after_operator_call() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["ts_mean(close, 20) extra".to_string()]);
    let err = planner
        .compile(&request)
        .expect_err("compile should reject trailing tokens");
    assert!(matches!(
        err,
        crate::error::CompileError::InvalidExpression { .. }
    ));
}

#[test]
fn compile_supports_window_and_lag_kwargs() {
    let planner = SimplePlanner;
    planner
        .compile(&FactorRequest::new(vec![
            "ts_mean(close, window=3)".to_string()
        ]))
        .expect("window kwarg should compile");
    planner
        .compile(&FactorRequest::new(vec![
            "ts_delta(volume, lag=1)".to_string()
        ]))
        .expect("lag kwarg should compile");
    planner
        .compile(&FactorRequest::new(vec![
            "ts_corr(close, volume, window=2)".to_string(),
        ]))
        .expect("two-field window kwarg should compile");
    planner
        .compile(&FactorRequest::new(vec![
            "ts_quantile(close, window=3, q=0.5)".to_string(),
        ]))
        .expect("window+q kwargs should compile");
}

#[test]
fn compile_supports_top_level_elem_infix_add_mul() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["close + volume * close".to_string()]);
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.node_count, 2);
    assert_eq!(logical.outputs, vec![1]);
}

#[test]
fn compile_supports_top_level_unary_minus_expression() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["-close".to_string()]);
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.node_count, 1);
    assert_eq!(logical.outputs, vec![0]);
}

#[test]
fn compile_supports_wave1_new_ops() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&wave1_new_ops_request())
        .expect("compile should succeed");
    assert_eq!(manifest.expr_count, 7);
    assert!(
        manifest.cse_hit_count >= 1,
        "expected at least one CSE hit for repeated elem_max usage"
    );
    assert_eq!(logical.outputs.len(), 7);
}

#[test]
fn compile_supports_wave2_ts_stats_ops() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&wave2_ts_stats_request())
        .expect("compile should succeed");
    assert_eq!(manifest.expr_count, 4);
    assert!(
        manifest.cse_hit_count >= 1,
        "expected cse hit for repeated ts_lag(close,1)"
    );
    assert_eq!(logical.outputs.len(), 4);
}

#[test]
fn compile_supports_wave3_var_beta_ops() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&wave3_var_beta_request())
        .expect("compile should succeed");
    assert_eq!(manifest.expr_count, 4);
    assert_eq!(logical.outputs.len(), 4);
}

#[test]
fn compile_supports_wave4_ewm_ops() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&wave4_ewm_request())
        .expect("compile should succeed");
    assert_eq!(manifest.expr_count, 5);
    assert!(
        manifest.cse_hit_count >= 1,
        "expected cse hit for repeated ts_lag(close,1)"
    );
    assert_eq!(logical.outputs.len(), 5);
}

#[test]
fn compile_supports_wave5_quantile_ops() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&wave5_quantile_request())
        .expect("compile should succeed");
    assert_eq!(manifest.expr_count, 3);
    assert!(
        manifest.cse_hit_count >= 1,
        "expected cse hit for repeated ts_lag(close,1)"
    );
    assert_eq!(logical.outputs.len(), 3);
}

#[test]
fn compile_supports_wave6_argext_ops() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&wave6_argext_request())
        .expect("compile should succeed");
    assert_eq!(manifest.expr_count, 3);
    assert_eq!(logical.outputs.len(), 3);
}

#[test]
fn compile_supports_wave7_elem_unary_ops() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&wave7_elem_unary_request())
        .expect("compile should succeed");
    assert_eq!(manifest.expr_count, 3);
    assert_eq!(logical.outputs.len(), 3);
}

#[test]
fn compile_supports_wave8_elem_extended_ops() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&wave8_elem_extended_request())
        .expect("compile should succeed");
    assert_eq!(manifest.expr_count, 3);
    assert_eq!(logical.outputs.len(), 3);
}

#[test]
fn compile_supports_wave9_elem_conditional_ops() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&wave9_elem_conditional_request())
        .expect("compile should succeed");
    assert_eq!(manifest.expr_count, 3);
    assert_eq!(logical.outputs.len(), 3);
}

#[test]
fn compile_supports_wave10_decay_linear_ops() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&wave10_decay_linear_request())
        .expect("compile should succeed");
    assert_eq!(manifest.expr_count, 2);
    assert_eq!(logical.outputs.len(), 2);
}

#[test]
fn compile_supports_wave11_product_mad_ops() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&wave11_product_mad_request())
        .expect("compile should succeed");
    assert_eq!(manifest.expr_count, 3);
    assert_eq!(logical.outputs.len(), 3);
}

#[test]
fn compile_supports_wave12_cs_preprocess_ops() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&wave12_cs_preprocess_request())
        .expect("compile should succeed");
    assert_eq!(manifest.expr_count, 3);
    assert_eq!(logical.outputs.len(), 3);
}

#[test]
fn compile_supports_wave13_cs_clip_ops() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&wave13_cs_clip_request())
        .expect("compile should succeed");
    assert_eq!(manifest.expr_count, 2);
    assert_eq!(logical.outputs.len(), 2);
}

#[test]
fn compile_supports_wave14_cs_neutralize_ops() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&wave14_cs_neutralize_request())
        .expect("compile should succeed");
    assert_eq!(manifest.expr_count, 2);
    assert_eq!(logical.outputs.len(), 2);
}

#[test]
fn compile_supports_wave15_cs_neutralize_multi3_and_alias() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&wave15_cs_neutralize_multi3_request())
        .expect("compile should succeed");
    assert_eq!(manifest.expr_count, 2);
    assert_eq!(logical.outputs.len(), 2);
}

#[test]
fn compile_rejects_alpha_style_alias_operator_names() {
    let planner = SimplePlanner;
    for expr in [
        "ts_delay(close, 1)",
        "ts_std_dev(close, 3)",
        "ts_covariance(close, volume, 3)",
        "ts_arg_max(close, 3)",
        "Pow(close, 2)",
        "abs_(close)",
        "log(close)",
        "sign(close)",
        "if_else(1, close, volume)",
        "max_(close, volume)",
        "min_(close, volume)",
    ] {
        let err = planner
            .compile(&FactorRequest::new(vec![expr.to_string()]))
            .expect_err("alias should be rejected in strict canonical DSL");
        assert!(
            matches!(err, crate::error::CompileError::UnknownOperator { .. }),
            "unexpected error for `{expr}`: {err:?}"
        );
    }
}

#[test]
fn compile_supports_alpha101_canonical_subset() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&alpha101_canonical_subset_request())
        .expect("alpha101 canonical subset should compile");
    assert_eq!(manifest.expr_count, 7);
    assert_eq!(logical.outputs.len(), 7);
    assert!(
        manifest.node_count >= 7,
        "node_count should be at least outputs: {manifest:?}"
    );
}

#[test]
fn compile_supports_phase_b_boolean_and_scale_ops() {
    let planner = SimplePlanner;
    let (logical, manifest) = planner
        .compile(&phase_b_boolean_and_scale_request())
        .expect("phase-b request should compile");
    assert_eq!(manifest.expr_count, 4);
    assert_eq!(logical.outputs.len(), 4);
}

#[test]
fn compile_supports_cs_neutralize_kwargs_and_rejects_invalid_layout() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec![
        "cs_neutralize(close, group=volume, weights=open, standardize=1)".to_string(),
    ]);
    planner
        .compile(&request)
        .expect("cs_neutralize kwargs should compile");

    let err = planner
        .compile(&FactorRequest::new(vec![
            "cs_neutralize(close, group=1.0)".to_string()
        ]))
        .expect_err("scalar group should be rejected");
    assert!(matches!(
        err,
        crate::error::CompileError::InvalidExpression { .. }
    ));

    let err = planner
        .compile(&FactorRequest::new(vec![
            "cs_neutralize_ols_multi(close, open, high, low, group=volume, weights=open)"
                .to_string(),
        ]))
        .expect_err("input count exceeding MAX_NODE_INPUTS should be rejected");
    assert!(matches!(
        err,
        crate::error::CompileError::InvalidExpression { .. }
    ));
}

#[test]
fn compile_rejects_scalar_literal_for_non_elem_operator() {
    let planner = SimplePlanner;
    let err = planner
        .compile(&FactorRequest::new(vec!["ts_mean(0.5, 3)".to_string()]))
        .expect_err("scalar literal should be rejected for ts operator");
    assert!(matches!(
        err,
        crate::error::CompileError::InvalidExpression { .. }
    ));
}

#[test]
fn compile_supports_top_level_elem_infix_sub_div() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["close - volume / close".to_string()]);
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.node_count, 2);
    assert_eq!(logical.outputs, vec![1]);
}

#[test]
fn compile_supports_nested_operator_calls() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["ts_std(ts_mean(close, 2), 2)".to_string()]);
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.node_count, 2);
    assert_eq!(logical.outputs, vec![1]);

    let required = logical
        .required_fields
        .keys()
        .map(|k| format!("{:?}.{}", k.source_kind, k.field))
        .collect::<Vec<_>>();
    assert!(required.iter().any(|k| k.contains("close")));
    assert!(required.iter().any(|k| k.contains("__derived__n0")));

    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");
    assert_eq!(physical.ready_required_fields.len(), 1);
}

#[test]
fn lineage_taint_propagates_from_nested_multi_source_ancestor() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec![
        "ts_mean(ts_corr(bar.close, quote_tick.bid_price, 2), 2)".to_string(),
    ]);
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec![
        "bar.close".to_string(),
        "quote_tick.bid_price".to_string(),
    ]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");
    let root = &physical.nodes[1];
    assert!(matches!(root.exec_mode, ExecMode::BarrierSingle));
    assert!(root.lineage.has_multi_ancestor);
    assert!(root.lineage.barrier_tainted);
}

#[test]
fn compile_rejects_zero_window_or_lag_params() {
    let planner = SimplePlanner;
    let err = planner
        .compile(&FactorRequest::new(vec!["ts_mean(close, 0)".to_string()]))
        .expect_err("compile should reject zero window");
    assert!(matches!(
        err,
        crate::error::CompileError::InvalidExpression { .. }
    ));

    let err = planner
        .compile(&FactorRequest::new(vec!["ts_delta(volume, 0)".to_string()]))
        .expect_err("compile should reject zero lag");
    assert!(matches!(
        err,
        crate::error::CompileError::InvalidExpression { .. }
    ));

    let err = planner
        .compile(&FactorRequest::new(vec![
            "ts_quantile(close, 4, 1.5)".to_string()
        ]))
        .expect_err("compile should reject invalid quantile");
    assert!(matches!(
        err,
        crate::error::CompileError::InvalidExpression { .. }
    ));
}

#[test]
fn bind_partitions_nodes_by_cs_and_source_cardinality() {
    let planner = SimplePlanner;
    let request = three_factor_request();
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![101, 202]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    assert_eq!(physical.single_nodes, vec![0, 1]);
    assert_eq!(physical.multi_nodes, vec![2]);
    assert!(matches!(
        physical.nodes[2].exec_mode,
        ExecMode::BarrierMulti
    ));
    assert_eq!(physical.nodes[2].lineage.source_cardinality(), 1);
    assert!(!physical.nodes[2].lineage.has_multi_ancestor);
    assert!(physical.nodes[2].lineage.barrier_tainted);
}

#[test]
fn bind_marks_ts_two_input_same_source_as_event() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["ts_corr(bar.close, bar.volume, 3)".to_string()]);
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    assert_eq!(physical.single_nodes, vec![0]);
    assert!(physical.multi_nodes.is_empty());
    assert!(matches!(physical.nodes[0].exec_mode, ExecMode::EventSingle));
    assert_eq!(physical.nodes[0].lineage.source_cardinality(), 1);
    assert!(!physical.nodes[0].lineage.has_multi_ancestor);
    assert!(!physical.nodes[0].lineage.barrier_tainted);
}

#[test]
fn bind_marks_ts_two_input_cross_source_as_barrier() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec![
        "ts_corr(bar.close, quote_tick.bid_price, 3)".to_string()
    ]);
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec![
        "bar.close".to_string(),
        "quote_tick.bid_price".to_string(),
    ]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    assert!(physical.single_nodes.is_empty());
    assert_eq!(physical.multi_nodes, vec![0]);
    assert!(matches!(
        physical.nodes[0].exec_mode,
        ExecMode::BarrierSingle
    ));
    assert_eq!(physical.nodes[0].lineage.source_cardinality(), 2);
    assert!(physical.nodes[0].lineage.has_multi_ancestor);
    assert!(physical.nodes[0].lineage.barrier_tainted);
}

#[test]
fn barrier_fallback_runs_single_ts_kernel_for_cross_source_node() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec![
        "ts_corr(bar.close, quote_tick.bid_price, 2)".to_string()
    ]);
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec![
        "bar.close".to_string(),
        "quote_tick.bid_price".to_string(),
    ]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    for event in [
        bar_event(1_000, 101, 1.0, 10.0),
        quote_event(1_000, 101, 2.0, 2.1),
        bar_event(2_000, 101, 2.0, 10.0),
        quote_event(2_000, 101, 4.0, 4.1),
    ] {
        engine.on_event(&event).expect("event should succeed");
    }
    let frame = engine.advance(2_000).expect("advance should succeed");
    assert!(approx_eq(
        frame.value_at(0, 0).expect("corr output should be present"),
        1.0
    ));
}

#[test]
fn compile_dedups_identical_expressions_and_supports_alias_lookup() {
    let planner = SimplePlanner;
    let request = FactorRequest {
        exprs: vec![
            "ts_mean(close, 2)".to_string(),
            "ts_mean(close, 2)".to_string(),
        ],
        outputs: vec!["m_a".to_string(), "m_b".to_string()],
        opts: CompileOptions::default(),
    };
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.node_count, 1);
    assert_eq!(logical.outputs, vec![0, 0]);

    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");
    assert_eq!(physical.output_names, vec!["m_a".to_string()]);
    assert_eq!(physical.output_aliases, vec![("m_b".to_string(), 0)]);

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    engine
        .on_event(&bar_event(10_000, 101, 10.0, 100.0))
        .expect("event should succeed");
    engine
        .on_event(&bar_event(11_000, 101, 14.0, 110.0))
        .expect("event should succeed");
    let frame = engine.advance(11_000).expect("advance should succeed");
    assert_eq!(frame.factor_count, 1);
    assert!(approx_eq(
        frame
            .factor_value(0, "m_a")
            .expect("canonical output should be available"),
        12.0
    ));
    assert!(approx_eq(
        frame
            .factor_value(0, "m_b")
            .expect("alias output should be available"),
        12.0
    ));
}

#[test]
fn compile_dedups_commutative_elem_expressions_and_supports_alias_lookup() {
    let planner = SimplePlanner;
    let request = FactorRequest {
        exprs: vec!["close + volume".to_string(), "volume + close".to_string()],
        outputs: vec!["sum_ab".to_string(), "sum_ba".to_string()],
        opts: CompileOptions::default(),
    };
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.node_count, 1);
    assert_eq!(logical.outputs, vec![0, 0]);

    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");
    assert_eq!(physical.output_names, vec!["sum_ab".to_string()]);
    assert_eq!(physical.output_aliases, vec![("sum_ba".to_string(), 0)]);

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    engine
        .on_event(&bar_event(12_000, 101, 2.0, 3.0))
        .expect("event should succeed");
    let frame = engine.advance(12_000).expect("advance should succeed");
    assert!(approx_eq(
        frame
            .factor_value(0, "sum_ab")
            .expect("canonical output should be available"),
        5.0
    ));
    assert!(approx_eq(
        frame
            .factor_value(0, "sum_ba")
            .expect("alias output should be available"),
        5.0
    ));
}

#[test]
fn compile_dedups_commutative_corr_expressions_and_supports_alias_lookup() {
    let planner = SimplePlanner;
    let request = FactorRequest {
        exprs: vec![
            "ts_corr(close, volume, 3)".to_string(),
            "ts_corr(volume, close, 3)".to_string(),
        ],
        outputs: vec!["corr_ab".to_string(), "corr_ba".to_string()],
        opts: CompileOptions::default(),
    };
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.node_count, 1);
    assert_eq!(logical.outputs, vec![0, 0]);

    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");
    assert_eq!(physical.output_names, vec!["corr_ab".to_string()]);
    assert_eq!(physical.output_aliases, vec![("corr_ba".to_string(), 0)]);

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    for (ts, close, volume) in [(1_i64, 1.0_f64, 2.0_f64), (2, 2.0, 4.0), (3, 3.0, 6.0)] {
        engine
            .on_event(&bar_event(ts, 101, close, volume))
            .expect("event should succeed");
    }
    let frame = engine.advance(3).expect("advance should succeed");
    assert!(approx_eq(
        frame
            .factor_value(0, "corr_ab")
            .expect("canonical output should be available"),
        1.0
    ));
    assert!(approx_eq(
        frame
            .factor_value(0, "corr_ba")
            .expect("alias output should be available"),
        1.0
    ));
}

#[test]
fn compile_keeps_non_commutative_elem_expressions_distinct() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec![
        "close - volume".to_string(),
        "volume - close".to_string(),
    ]);
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.node_count, 2);
    assert_eq!(logical.outputs, vec![0, 1]);
}

#[test]
fn compile_keeps_non_commutative_bivariate_expressions_distinct() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec![
        "ts_linear_regression(close, volume, 3)".to_string(),
        "ts_linear_regression(volume, close, 3)".to_string(),
    ]);
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.node_count, 2);
    assert_eq!(logical.outputs, vec![0, 1]);
}

#[test]
fn compile_dedups_associative_elem_add_tree_shapes() {
    let planner = SimplePlanner;
    let request = FactorRequest {
        exprs: vec![
            "(close + volume) + close".to_string(),
            "close + (close + volume)".to_string(),
        ],
        outputs: vec!["sum_l".to_string(), "sum_r".to_string()],
        opts: CompileOptions::default(),
    };
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.node_count, 2);
    assert_eq!(logical.outputs, vec![1, 1]);

    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");
    assert_eq!(physical.output_names[1], "sum_l");
    assert_eq!(physical.output_aliases, vec![("sum_r".to_string(), 1)]);

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");
    engine
        .on_event(&bar_event(20_000, 101, 2.0, 3.0))
        .expect("event should succeed");
    let frame = engine.advance(20_000).expect("advance should succeed");
    assert!(approx_eq(
        frame
            .factor_value(0, "sum_l")
            .expect("canonical output should be available"),
        7.0
    ));
    assert!(approx_eq(
        frame
            .factor_value(0, "sum_r")
            .expect("alias output should be available"),
        7.0
    ));
}

#[test]
fn compile_folds_elem_identity_for_nested_derived_input() {
    let planner = SimplePlanner;
    let request = FactorRequest {
        exprs: vec![
            "ts_mean(close, 2)".to_string(),
            "ts_mean(close, 2) + 0".to_string(),
            "ts_mean(close, 2) * 1".to_string(),
        ],
        outputs: vec!["base".to_string(), "plus0".to_string(), "mul1".to_string()],
        opts: CompileOptions::default(),
    };
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.node_count, 1);
    assert_eq!(logical.outputs, vec![0, 0, 0]);

    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");
    assert_eq!(physical.output_names, vec!["base".to_string()]);
    assert_eq!(
        physical.output_aliases,
        vec![("plus0".to_string(), 0), ("mul1".to_string(), 0)]
    );
}

#[test]
fn compile_manifest_reports_optimization_counters_for_complex_expr_list() {
    let planner = SimplePlanner;
    let request = FactorRequest {
        exprs: vec![
            "close + volume".to_string(),
            "volume + close".to_string(),
            "(close + volume) + close".to_string(),
            "close + (close + volume)".to_string(),
            "ts_mean(close, 2)".to_string(),
            "ts_mean(close, 2) + 0".to_string(),
            "ts_corr(ts_mean(close, 2), ts_delta(volume, 1), 3)".to_string(),
            "ts_corr(ts_delta(volume, 1), ts_mean(close, 2), 3)".to_string(),
        ],
        outputs: vec![
            "e0".to_string(),
            "e1".to_string(),
            "e2".to_string(),
            "e3".to_string(),
            "e4".to_string(),
            "e5".to_string(),
            "e6".to_string(),
            "e7".to_string(),
        ],
        opts: CompileOptions::default(),
    };

    let (_logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.expr_count, request.exprs.len());
    assert!(manifest.lowered_op_count >= manifest.node_count);
    assert!(manifest.cse_hit_count >= 2);
    assert!(manifest.identity_fold_count >= 1);
    assert!(manifest.alias_count >= 2);
    assert!(manifest.summary_line().contains("compile_us="));
}

#[test]
fn compile_folds_inline_constant_subexpressions() {
    let planner = SimplePlanner;
    let request = FactorRequest {
        exprs: vec![
            "open * 0.728317 + open * (1.0 - 0.728317)".to_string(),
            "open * (1.0 - 0.967285)".to_string(),
        ],
        outputs: vec!["mix_w".to_string(), "tiny_w".to_string()],
        opts: CompileOptions::default(),
    };
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.node_count, 4, "{manifest:?}");
    assert!(
        logical.nodes.iter().all(|node| node.op != OpCode::ElemSub),
        "constant-only subtraction should be folded during lowering"
    );
}

#[test]
fn complex_expression_regression_suite_compiles_and_executes() {
    let planner = SimplePlanner;
    let request = FactorRequest {
        exprs: vec![
            "ts_mean(close, 3)".to_string(),
            "ts_delta(volume, 1)".to_string(),
            "ts_std(ts_mean(close, 2), 2)".to_string(),
            "ts_corr(ts_mean(close, 2), ts_delta(volume, 1), 3)".to_string(),
            "(close + volume * close) - close / volume".to_string(),
            "cs_rank(ts_mean(close, 2) + ts_delta(volume, 1))".to_string(),
        ],
        outputs: vec![
            "mean3".to_string(),
            "delta1".to_string(),
            "std_nested".to_string(),
            "corr_nested".to_string(),
            "elem_mix".to_string(),
            "cs_mix".to_string(),
        ],
        opts: CompileOptions::default(),
    };
    let (logical, manifest) = planner.compile(&request).expect("compile should succeed");
    assert_eq!(manifest.expr_count, request.exprs.len());
    assert_eq!(logical.outputs.len(), request.exprs.len());

    let universe = Universe::new(vec![11, 22]);
    let catalog = InputFieldCatalog::new(vec!["bar.close".to_string(), "bar.volume".to_string()]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let mut engine = OnlineFactorEngine::default();
    engine.load(physical).expect("load should succeed");

    for ts in 1_i64..=4_i64 {
        let close_a = 10.0 + ts as f64;
        let close_b = 20.0 + ts as f64;
        let volume_a = 100.0 + (ts as f64).powi(2);
        let volume_b = 200.0 + (ts as f64).powi(2);
        engine
            .on_event(&bar_event(ts, 11, close_a, volume_a))
            .expect("event A should succeed");
        engine
            .on_event(&bar_event(ts, 22, close_b, volume_b))
            .expect("event B should succeed");
    }

    let frame = engine.advance(4).expect("advance should succeed");
    for name in [
        "mean3",
        "delta1",
        "std_nested",
        "corr_nested",
        "elem_mix",
        "cs_mix",
    ] {
        let idx = frame.factor_idx(name).expect("factor should exist");
        assert!(frame.valid_mask[idx], "{name} should be valid at ts=4");
        assert!(
            frame
                .factor_value(0, name)
                .expect("instrument 0 value")
                .is_finite(),
            "{name} should be finite"
        );
    }
    assert!(
        frame
            .factor_value(1, "cs_mix")
            .expect("instrument 1 cs value should exist")
            > frame
                .factor_value(0, "cs_mix")
                .expect("instrument 0 cs value should exist")
    );
}

#[test]
fn compile_supports_explicit_source_prefix_and_normalizes_field_case() {
    let planner = SimplePlanner;
    let request = FactorRequest {
        exprs: vec!["ts_mean(Trade_Tick.Price, 3)".to_string()],
        outputs: Vec::new(),
        opts: CompileOptions {
            default_source_kind: SourceKind::Bar,
        },
    };
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let mut iter = logical.required_fields.keys();
    let key = iter.next().expect("one required field");
    assert_eq!(key.source_kind, SourceKind::TradeTick);
    assert_eq!(key.field, "price");

    let universe = Universe::new(vec![101]);
    let catalog = InputFieldCatalog::new(vec!["trade_tick.price".to_string()]);
    planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");
}

#[test]
fn compile_rejects_unknown_source_prefix() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec!["ts_mean(foobar.close, 3)".to_string()]);
    let err = planner
        .compile(&request)
        .expect_err("compile should reject unknown source");
    assert!(matches!(
        err,
        crate::error::CompileError::InvalidExpression { .. }
    ));
}

#[test]
fn history_len_policy_is_driven_by_registry_specs() {
    let planner = SimplePlanner;
    let request = FactorRequest::new(vec![
        "ts_mean(close, 20)".to_string(),
        "ts_delta(volume, 3)".to_string(),
        "ts_rank(vwap, 5)".to_string(),
        "cs_rank(mid)".to_string(),
    ]);
    let (logical, _) = planner.compile(&request).expect("compile should succeed");
    let universe = Universe::new(vec![1]);
    let catalog = InputFieldCatalog::new(vec![
        "bar.close".to_string(),
        "bar.volume".to_string(),
        "bar.vwap".to_string(),
        "bar.mid".to_string(),
    ]);
    let physical = planner
        .bind(&logical, &universe, &catalog, AdvancePolicy::StrictAllReady)
        .expect("bind should succeed");

    let history_by_field: HashMap<&str, usize> = physical
        .fields
        .iter()
        .map(|binding| (binding.key.field.as_str(), binding.history_len))
        .collect();
    assert_eq!(history_by_field.get("close"), Some(&1));
    assert_eq!(history_by_field.get("volume"), Some(&4));
    assert_eq!(history_by_field.get("vwap"), Some(&5));
    assert_eq!(history_by_field.get("mid"), Some(&1));
}

#[test]
#[ignore = "micro-benchmark; run manually with -- --ignored --nocapture"]
fn benchmark_compile_manifest_metrics() {
    let planner = SimplePlanner;
    let request = FactorRequest {
        exprs: vec![
            "ts_mean(close, 20)".to_string(),
            "ts_delta(volume, 1)".to_string(),
            "ts_std(ts_mean(close, 4), 4)".to_string(),
            "ts_corr(ts_mean(close, 4), ts_delta(volume, 1), 5)".to_string(),
            "(close + volume * close) - close / volume".to_string(),
            "cs_rank(ts_mean(close, 4) + ts_delta(volume, 1))".to_string(),
            "ts_corr(ts_delta(volume, 1), ts_mean(close, 4), 5)".to_string(),
            "close + (volume + close)".to_string(),
        ],
        outputs: vec![
            "m0".to_string(),
            "m1".to_string(),
            "m2".to_string(),
            "m3".to_string(),
            "m4".to_string(),
            "m5".to_string(),
            "m6".to_string(),
            "m7".to_string(),
        ],
        opts: CompileOptions::default(),
    };

    let rounds = 2_000usize;
    let t0 = Instant::now();
    let mut sum_compile_us = 0u128;
    let mut sum_nodes = 0usize;
    let mut sum_cse_hits = 0usize;
    let mut sum_aliases = 0usize;
    for _ in 0..rounds {
        let (_logical, manifest) = planner.compile(&request).expect("compile should succeed");
        sum_compile_us += manifest.compile_time_us as u128;
        sum_nodes += manifest.node_count;
        sum_cse_hits += manifest.cse_hit_count;
        sum_aliases += manifest.alias_count;
    }
    let wall = t0.elapsed();

    println!(
        "compile benchmark rounds={} wall={:?} avg_compile_us={:.2} avg_nodes={:.2} avg_cse_hits={:.2} avg_aliases={:.2}",
        rounds,
        wall,
        (sum_compile_us as f64) / (rounds as f64),
        (sum_nodes as f64) / (rounds as f64),
        (sum_cse_hits as f64) / (rounds as f64),
        (sum_aliases as f64) / (rounds as f64)
    );
}
