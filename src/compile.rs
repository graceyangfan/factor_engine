use crate::compile_expr::{parse_expression, BinaryOp, ExprAst, UnaryOp};
use crate::error::{BindError, CompileError};
use crate::ops::{CompileArgSpec, Domain, ExecCapability, OpCode};
use crate::ops::{HistorySpec, OperatorRegistry};
use crate::plan::{
    CompileManifest, ExecMode, FieldBinding, FieldKey, LogicalNode, LogicalParam, LogicalPlan,
    NodeLineage, PhysicalNode, PhysicalPlan, MAX_NODE_INPUTS,
};
use crate::types::{AdvancePolicy, FactorRequest, InputFieldCatalog, SourceKind, Universe};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::OnceLock;
use std::time::Instant;

pub trait Planner {
    fn compile(&self, req: &FactorRequest) -> Result<(LogicalPlan, CompileManifest), CompileError>;
    fn bind(
        &self,
        logical: &LogicalPlan,
        universe: &Universe,
        catalog: &InputFieldCatalog,
        policy: AdvancePolicy,
    ) -> Result<PhysicalPlan, BindError>;
}

#[derive(Debug, Default)]
pub struct SimplePlanner;

impl Planner for SimplePlanner {
    fn compile(&self, req: &FactorRequest) -> Result<(LogicalPlan, CompileManifest), CompileError> {
        if req.exprs.is_empty() {
            return Err(CompileError::EmptyRequest);
        }
        let started_at = Instant::now();

        let mut nodes = Vec::with_capacity(req.exprs.len());
        let mut outputs = Vec::with_capacity(req.exprs.len());
        let mut output_aliases = Vec::new();
        let mut required_fields: BTreeMap<FieldKey, usize> = BTreeMap::new();
        let mut node_by_sig: HashMap<NodeSignature, usize> = HashMap::new();
        let mut output_name_to_slot: HashMap<String, usize> = HashMap::new();
        let mut derived_field_by_node: HashMap<usize, FieldKey> = HashMap::new();
        let mut lower_stats = LowerStats::default();

        for (idx, expr) in req.exprs.iter().enumerate() {
            let ast = parse_expression(expr)?;
            let root_slot = {
                let mut ctx = LowerCtx {
                    expr,
                    default_source_kind: req.opts.default_source_kind,
                    nodes: &mut nodes,
                    required_fields: &mut required_fields,
                    node_by_sig: &mut node_by_sig,
                    derived_field_by_node: &mut derived_field_by_node,
                    lower_stats: &mut lower_stats,
                };
                ctx.lower_root(&ast)?
            };
            let output_name = req.outputs.get(idx).cloned().unwrap_or_else(|| {
                let meta = OperatorRegistry::get_by_op(nodes[root_slot].op)
                    .expect("compiled operator must exist in registry");
                format!("f{}_{}", idx, meta.name)
            });

            outputs.push(root_slot);
            register_output_name(
                &mut output_name_to_slot,
                &mut output_aliases,
                &mut nodes,
                output_name,
                root_slot,
                expr,
            )?;
        }

        let logical = LogicalPlan {
            nodes,
            outputs,
            output_aliases,
            required_fields,
        };
        let manifest = CompileManifest {
            node_count: logical.nodes.len(),
            field_count: logical.required_fields.len(),
            expr_count: req.exprs.len(),
            lowered_op_count: lower_stats.lowered_op_count,
            cse_hit_count: lower_stats.cse_hit_count,
            identity_fold_count: lower_stats.identity_fold_count,
            alias_count: logical.output_aliases.len(),
            compile_time_us: started_at.elapsed().as_micros() as u64,
        };
        if compile_manifest_debug_enabled() {
            eprintln!("[factor_engine::compile] {}", manifest.summary_line());
        }
        Ok((logical, manifest))
    }

    fn bind(
        &self,
        logical: &LogicalPlan,
        universe: &Universe,
        catalog: &InputFieldCatalog,
        policy: AdvancePolicy,
    ) -> Result<PhysicalPlan, BindError> {
        if universe.is_empty() {
            return Err(BindError::EmptyUniverse);
        }

        validate_required_fields(logical, catalog)?;
        let source_kinds = collect_source_kinds(logical);
        let source_slot_by_kind = build_source_slot_by_kind(&source_kinds);
        let (mut fields, field_slot_by_key) = build_field_bindings(logical, &source_slot_by_kind);
        let (nodes, single_nodes, multi_nodes) =
            build_physical_nodes(logical, &field_slot_by_key, &mut fields);

        let ready_required_fields =
            build_ready_required_fields(logical, &field_slot_by_key, logical.required_fields.len());

        Ok(PhysicalPlan {
            source_kinds,
            fields,
            nodes,
            single_nodes,
            multi_nodes,
            output_names: logical
                .nodes
                .iter()
                .map(|n| n.output_name.clone())
                .collect(),
            output_aliases: logical.output_aliases.clone(),
            policy,
            universe_slots: universe.instrument_slots.clone(),
            ready_required_fields,
        })
    }
}

fn validate_required_fields(
    logical: &LogicalPlan,
    catalog: &InputFieldCatalog,
) -> Result<(), BindError> {
    for key in logical.required_fields.keys() {
        if is_derived_field_name(&key.field) || is_const_field_name(&key.field) {
            continue;
        }
        if !catalog_contains_field(catalog, key) {
            return Err(BindError::MissingField {
                field: canonical_field_name(key.source_kind, &key.field),
            });
        }
    }
    Ok(())
}

fn collect_source_kinds(logical: &LogicalPlan) -> Vec<SourceKind> {
    let mut kinds = BTreeSet::new();
    for key in logical.required_fields.keys() {
        kinds.insert(key.source_kind);
    }
    kinds.into_iter().collect()
}

fn build_source_slot_by_kind(source_kinds: &[SourceKind]) -> HashMap<SourceKind, u16> {
    source_kinds
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, kind)| (kind, idx as u16))
        .collect()
}

fn build_field_bindings(
    logical: &LogicalPlan,
    source_slot_by_kind: &HashMap<SourceKind, u16>,
) -> (Vec<FieldBinding>, HashMap<FieldKey, usize>) {
    let mut fields = Vec::with_capacity(logical.required_fields.len());
    let mut field_slot_by_key = HashMap::with_capacity(logical.required_fields.len());
    for (slot, (key, history_len)) in logical.required_fields.iter().enumerate() {
        fields.push(FieldBinding {
            field_slot: slot,
            source_slot: *source_slot_by_kind
                .get(&key.source_kind)
                .expect("source table built from required fields"),
            key: key.clone(),
            history_len: *history_len,
        });
        field_slot_by_key.insert(key.clone(), slot);
    }
    (fields, field_slot_by_key)
}

fn build_physical_nodes(
    logical: &LogicalPlan,
    field_slot_by_key: &HashMap<FieldKey, usize>,
    fields: &mut [FieldBinding],
) -> (Vec<PhysicalNode>, Vec<usize>, Vec<usize>) {
    let mut nodes = Vec::with_capacity(logical.nodes.len());
    let mut single_nodes = Vec::new();
    let mut multi_nodes = Vec::new();
    for node in &logical.nodes {
        let mut input_field_slots = [0usize; MAX_NODE_INPUTS];
        for (idx, key) in node.input_fields.iter().enumerate() {
            input_field_slots[idx] = *field_slot_by_key
                .get(key)
                .expect("field slots are built from required_fields");
        }
        let input_count = node.input_fields.len() as u8;
        let meta =
            OperatorRegistry::get_by_op(node.op).expect("compiled operator must exist in registry");
        let exec_binding = resolve_exec_binding(node.lineage, meta);
        let shared_runtime_enabled = meta.shared_family.is_some()
            && matches!(exec_binding.exec_mode, ExecMode::EventSingle)
            && node.output_field.is_none();
        let runtime_history_len = resolve_runtime_history_len(
            meta,
            node.param,
            exec_binding.exec_mode,
            shared_runtime_enabled,
        );
        for field_slot in input_field_slots.iter().take(input_count as usize) {
            fields[*field_slot].history_len =
                fields[*field_slot].history_len.max(runtime_history_len);
        }
        let output_field_slot = node
            .output_field
            .as_ref()
            .and_then(|key| field_slot_by_key.get(key).copied());
        let exec_mode = exec_binding.exec_mode;
        nodes.push(PhysicalNode {
            node_id: node.node_id,
            op: node.op,
            lineage: exec_binding.lineage,
            input_field_slots,
            input_count,
            output_field_slot,
            exec_mode,
            param: node.param,
            output_slot: node.node_id,
        });
        if exec_mode.is_barrier() {
            multi_nodes.push(node.node_id);
        } else {
            single_nodes.push(node.node_id);
        }
    }
    (nodes, single_nodes, multi_nodes)
}

#[inline]
fn compile_manifest_debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FACTOR_ENGINE_COMPILE_DEBUG")
            .ok()
            .map(|raw| {
                let v = raw.trim().to_ascii_lowercase();
                matches!(v.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
    })
}

#[derive(Debug, Clone, Copy)]
struct ExecBinding {
    exec_mode: ExecMode,
    lineage: NodeLineage,
}

#[inline]
fn resolve_exec_binding(lineage: NodeLineage, meta: &crate::ops::OpMeta) -> ExecBinding {
    let op_barrier_semantic =
        meta.domain == Domain::Cs || matches!(meta.exec, ExecCapability::BarrierBatchExact);

    // CS operators stay barrier-multi by definition. Non-CS nodes become barrier-single
    // once lineage is tainted by multi-source ancestry.
    let exec_mode = if op_barrier_semantic {
        ExecMode::BarrierMulti
    } else if lineage.barrier_tainted {
        ExecMode::BarrierSingle
    } else {
        ExecMode::EventSingle
    };
    ExecBinding { exec_mode, lineage }
}

#[inline]
fn resolve_runtime_history_len(
    meta: &crate::ops::OpMeta,
    param: LogicalParam,
    exec_mode: ExecMode,
    shared_runtime_enabled: bool,
) -> usize {
    let mut base = history_len_from_spec(meta.history_spec, param);
    if meta.shared_family.is_some() && !shared_runtime_enabled {
        base = base.max(param_value(param).max(1));
    }
    if matches!(exec_mode, ExecMode::EventSingle | ExecMode::BarrierMulti) {
        return base;
    }
    // Barrier node backed by single-kernel fallback must keep enough field history
    // to evaluate the single-kernel logic at barrier time.
    let fallback = match meta.arg_spec {
        CompileArgSpec::FieldOnly
        | CompileArgSpec::TwoFields
        | CompileArgSpec::ThreeFields
        | CompileArgSpec::Fields2To4 => 1,
        CompileArgSpec::FieldWindow
        | CompileArgSpec::TwoFieldsWindow
        | CompileArgSpec::FieldWindowQuantile => param_value(param).max(1),
        CompileArgSpec::FieldLag => param_value(param).max(1) + 1,
    };
    base.max(fallback)
}

#[derive(Debug)]
struct ParsedCall {
    positional_args: Vec<String>,
    keyword_args: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum CanonicalSortKey {
    Scalar(u64),
    Field(FieldKey),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NodeSignature {
    op: OpCode,
    input_fields: Vec<FieldKey>,
    param: LogicalParam,
}

#[derive(Debug, Default)]
struct LowerStats {
    lowered_op_count: usize,
    cse_hit_count: usize,
    identity_fold_count: usize,
}

const DERIVED_FIELD_PREFIX: &str = "__derived__";
const CONST_FIELD_PREFIX: &str = "__const__";

#[inline]
fn canonical_op_name(op_name: &str) -> &str {
    // Keep DSL strict and simple: only canonical operator names are accepted.
    op_name
}

#[inline]
fn is_derived_field_name(field: &str) -> bool {
    field.starts_with(DERIVED_FIELD_PREFIX)
}

#[inline]
fn is_const_field_name(field: &str) -> bool {
    field.starts_with(CONST_FIELD_PREFIX)
}

struct LowerCtx<'a> {
    expr: &'a str,
    default_source_kind: SourceKind,
    nodes: &'a mut Vec<LogicalNode>,
    required_fields: &'a mut BTreeMap<FieldKey, usize>,
    node_by_sig: &'a mut HashMap<NodeSignature, usize>,
    derived_field_by_node: &'a mut HashMap<usize, FieldKey>,
    lower_stats: &'a mut LowerStats,
}

impl<'a> LowerCtx<'a> {
    fn lower_root(&mut self, ast: &ExprAst) -> Result<usize, CompileError> {
        match ast {
            ExprAst::Call { .. } => self.lower_call(ast),
            ExprAst::Binary { .. } => self.lower_binary(ast),
            ExprAst::Unary { .. } => self.lower_unary(ast),
            _ => Err(CompileError::InvalidExpression {
                expr: self.expr.to_string(),
                reason:
                    "top-level expression must be an operator call or arithmetic (`+ - * /`) expression"
                        .to_string(),
            }),
        }
    }

    fn lower_call(&mut self, ast: &ExprAst) -> Result<usize, CompileError> {
        let ExprAst::Call { name, args, kwargs } = ast else {
            return Err(CompileError::InvalidExpression {
                expr: self.expr.to_string(),
                reason: "internal lowering error: expected call node".to_string(),
            });
        };
        if name.trim().is_empty() {
            return Err(CompileError::InvalidExpression {
                expr: self.expr.to_string(),
                reason: "empty operator name".to_string(),
            });
        }

        let op_name = canonical_op_name(&name.to_ascii_lowercase()).to_string();

        let mut positional_args = Vec::with_capacity(args.len());
        for arg in args {
            positional_args.push(self.lower_arg_token(arg)?);
        }
        let mut keyword_args = BTreeMap::new();
        for (name, value) in kwargs {
            keyword_args.insert(name.clone(), self.lower_arg_token(value)?);
        }
        let parsed = ParsedCall {
            positional_args,
            keyword_args,
        };
        self.lower_parsed_call(&op_name, parsed)
    }

    fn lower_arg_token(&mut self, ast: &ExprAst) -> Result<String, CompileError> {
        if let Some(value) = evaluate_const_expr(ast) {
            return normalize_number_token(value);
        }
        match ast {
            ExprAst::Identifier(id) => Ok(id.clone()),
            ExprAst::Number(v) => normalize_number_token(*v),
            ExprAst::Unary { op, expr: inner } => match (op, inner.as_ref()) {
                (UnaryOp::Plus, ExprAst::Number(v)) => normalize_number_token(*v),
                (UnaryOp::Minus, ExprAst::Number(v)) => normalize_number_token(-*v),
                _ => {
                    let node_id = self.lower_unary(ast)?;
                    self.lowered_node_token(node_id)
                }
            },
            ExprAst::Call { .. } => self
                .lower_call(ast)
                .and_then(|node_id| self.lowered_node_token(node_id)),
            ExprAst::Binary { .. } => self
                .lower_binary(ast)
                .and_then(|node_id| self.lowered_node_token(node_id)),
        }
    }

    fn lowered_node_token(&mut self, node_id: usize) -> Result<String, CompileError> {
        let key = self.ensure_output_field(node_id)?;
        Ok(canonical_field_name(key.source_kind, &key.field))
    }

    fn lower_unary(&mut self, ast: &ExprAst) -> Result<usize, CompileError> {
        let ExprAst::Unary { op, expr: inner } = ast else {
            return Err(CompileError::InvalidExpression {
                expr: self.expr.to_string(),
                reason: "internal lowering error: expected unary node".to_string(),
            });
        };
        match op {
            UnaryOp::Plus => match inner.as_ref() {
                ExprAst::Call { .. } => self.lower_call(inner),
                ExprAst::Binary { .. } => self.lower_binary(inner),
                ExprAst::Unary { .. } => self.lower_unary(inner),
                ExprAst::Identifier(_) | ExprAst::Number(_) => self.lower_signed_expr(inner, 1.0),
            },
            UnaryOp::Minus => self.lower_signed_expr(inner, -1.0),
            UnaryOp::Not => {
                let parsed = ParsedCall {
                    positional_args: vec![self.lower_arg_token(inner)?],
                    keyword_args: BTreeMap::new(),
                };
                self.lower_parsed_call("elem_not", parsed)
            }
        }
    }

    fn lower_signed_expr(&mut self, inner: &ExprAst, sign: f64) -> Result<usize, CompileError> {
        let parsed = ParsedCall {
            positional_args: vec![self.lower_arg_token(inner)?, normalize_number_token(sign)?],
            keyword_args: BTreeMap::new(),
        };
        self.lower_parsed_call("elem_mul", parsed)
    }

    fn lower_binary(&mut self, ast: &ExprAst) -> Result<usize, CompileError> {
        let ExprAst::Binary { op, lhs, rhs } = ast else {
            return Err(CompileError::InvalidExpression {
                expr: self.expr.to_string(),
                reason: "internal lowering error: expected binary node".to_string(),
            });
        };

        if matches!(op, BinaryOp::Add | BinaryOp::Mul) {
            return self.lower_associative_binary(ast, *op);
        }

        let op_name = match op {
            BinaryOp::Add => "elem_add",
            BinaryOp::Sub => "elem_sub",
            BinaryOp::Mul => "elem_mul",
            BinaryOp::Div => "elem_div",
            BinaryOp::Lt => "elem_lt",
            BinaryOp::Le => "elem_le",
            BinaryOp::Gt => "elem_gt",
            BinaryOp::Ge => "elem_ge",
            BinaryOp::Eq => "elem_eq",
            BinaryOp::Ne => "elem_ne",
            BinaryOp::And => "elem_and",
            BinaryOp::Or => "elem_or",
        };

        let parsed = ParsedCall {
            positional_args: vec![self.lower_arg_token(lhs)?, self.lower_arg_token(rhs)?],
            keyword_args: BTreeMap::new(),
        };
        self.lower_parsed_call(op_name, parsed)
    }

    fn lower_associative_binary(
        &mut self,
        ast: &ExprAst,
        op: BinaryOp,
    ) -> Result<usize, CompileError> {
        let mut terms = Vec::new();
        collect_associative_terms(ast, op, &mut terms);
        if terms.len() < 2 {
            return self.lower_arg_expr_to_node_id(terms[0]);
        }

        let op_name = match op {
            BinaryOp::Add => "elem_add",
            BinaryOp::Mul => "elem_mul",
            _ => unreachable!("only add/mul are associative"),
        };

        let mut lowered = Vec::with_capacity(terms.len());
        for term in terms {
            let token = self.lower_arg_token(term)?;
            lowered.push((self.canonical_sort_key(&token)?, token));
        }
        lowered.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));

        let mut iter = lowered.into_iter().map(|(_, token)| token);
        let first = iter
            .next()
            .expect("associative op should have at least 2 terms");
        let second = iter
            .next()
            .expect("associative op should have at least 2 terms");
        let mut node_id = self.lower_parsed_call(
            op_name,
            ParsedCall {
                positional_args: vec![first, second],
                keyword_args: BTreeMap::new(),
            },
        )?;

        for token in iter {
            let lhs = self.ensure_output_field(node_id)?;
            node_id = self.lower_parsed_call(
                op_name,
                ParsedCall {
                    positional_args: vec![canonical_field_name(lhs.source_kind, &lhs.field), token],
                    keyword_args: BTreeMap::new(),
                },
            )?;
        }
        Ok(node_id)
    }

    fn lower_arg_expr_to_node_id(&mut self, ast: &ExprAst) -> Result<usize, CompileError> {
        match ast {
            ExprAst::Call { .. } => self.lower_call(ast),
            ExprAst::Binary { .. } => self.lower_binary(ast),
            ExprAst::Unary { .. } => self.lower_unary(ast),
            ExprAst::Identifier(_) | ExprAst::Number(_) => {
                let arg = self.lower_arg_token(ast)?;
                self.lower_parsed_call(
                    "elem_mul",
                    ParsedCall {
                        positional_args: vec![arg, normalize_number_token(1.0)?],
                        keyword_args: BTreeMap::new(),
                    },
                )
            }
        }
    }

    fn canonical_sort_key(&self, token: &str) -> Result<CanonicalSortKey, CompileError> {
        if let Some(value) = parse_scalar_literal(token) {
            return Ok(CanonicalSortKey::Scalar(value.to_bits()));
        }
        let key = parse_field_key(self.default_source_kind, token, self.expr)?;
        Ok(CanonicalSortKey::Field(key))
    }

    fn lower_parsed_call(
        &mut self,
        op_name: &str,
        parsed: ParsedCall,
    ) -> Result<usize, CompileError> {
        let meta = OperatorRegistry::get(op_name).ok_or_else(|| CompileError::UnknownOperator {
            name: op_name.to_string(),
        })?;
        let compiled_args = if is_cs_neutralize_op(op_name) {
            parse_cs_neutralize_call(op_name, parsed, self.expr)?
        } else {
            let normalized_args = normalize_call_args(meta.arg_spec, parsed, self.expr, op_name)?;
            meta.arg_spec.parse(&normalized_args, self.expr)?
        };
        self.lower_stats.lowered_op_count += 1;
        if compiled_args.fields.len() > MAX_NODE_INPUTS {
            return Err(CompileError::InvalidExpression {
                expr: self.expr.to_string(),
                reason: format!(
                    "operator `{}` has {} inputs, max supported is {}",
                    op_name,
                    compiled_args.fields.len(),
                    MAX_NODE_INPUTS
                ),
            });
        }

        let param = compiled_args.param;
        let mut input_fields = Vec::with_capacity(compiled_args.fields.len());
        for raw_field in compiled_args.fields {
            let key = if let Some(value) = parse_scalar_literal(&raw_field) {
                if !meta.allow_scalar_literals {
                    return Err(CompileError::InvalidExpression {
                        expr: self.expr.to_string(),
                        reason: format!(
                            "scalar literal `{}` is not allowed for operator `{}`",
                            raw_field, op_name
                        ),
                    });
                }
                make_const_field_key(value)
            } else {
                parse_field_key(self.default_source_kind, &raw_field, self.expr)?
            };
            input_fields.push(key);
        }
        canonicalize_first_two_inputs(meta.commutative_first_two, &mut input_fields);
        if let Some(node_id) = self.try_identity_passthrough(meta.op, &input_fields) {
            self.lower_stats.identity_fold_count += 1;
            return Ok(node_id);
        }

        let history_len = history_len_from_spec(meta.history_spec, param);
        for key in &input_fields {
            self.required_fields
                .entry(key.clone())
                .and_modify(|entry| *entry = (*entry).max(history_len))
                .or_insert(history_len);
        }

        let signature = NodeSignature {
            op: meta.op,
            input_fields: input_fields.clone(),
            param,
        };
        if let Some(existing_slot) = self.node_by_sig.get(&signature).copied() {
            self.lower_stats.cse_hit_count += 1;
            return Ok(existing_slot);
        }

        let node_id = self.nodes.len();
        let lineage = self.build_lineage(&input_fields, meta);
        self.nodes.push(LogicalNode {
            node_id,
            op: meta.op,
            lineage,
            input_fields,
            output_field: None,
            param,
            output_name: format!("f{node_id}_{}", meta.name),
        });
        self.node_by_sig.insert(signature, node_id);
        Ok(node_id)
    }

    fn try_identity_passthrough(&self, op: OpCode, input_fields: &[FieldKey]) -> Option<usize> {
        if input_fields.len() != 2 {
            return None;
        }
        let lhs = &input_fields[0];
        let rhs = &input_fields[1];
        match op {
            OpCode::ElemAdd => self
                .passthrough_if_identity(lhs, rhs, 0.0)
                .or_else(|| self.passthrough_if_identity(rhs, lhs, 0.0)),
            OpCode::ElemMul => self
                .passthrough_if_identity(lhs, rhs, 1.0)
                .or_else(|| self.passthrough_if_identity(rhs, lhs, 1.0)),
            OpCode::ElemSub => self.passthrough_if_identity(lhs, rhs, 0.0),
            OpCode::ElemDiv => self.passthrough_if_identity(lhs, rhs, 1.0),
            OpCode::ElemPow => self.passthrough_if_identity(lhs, rhs, 1.0),
            _ => None,
        }
    }

    fn passthrough_if_identity(
        &self,
        value_key: &FieldKey,
        maybe_identity: &FieldKey,
        expected: f64,
    ) -> Option<usize> {
        let value = const_value_from_field_key(maybe_identity)?;
        if value == expected {
            return self.node_id_from_derived_key(value_key);
        }
        None
    }

    fn node_id_from_derived_key(&self, key: &FieldKey) -> Option<usize> {
        let suffix = key
            .field
            .strip_prefix(DERIVED_FIELD_PREFIX)?
            .strip_prefix('n')?;
        let node_id = suffix.parse::<usize>().ok()?;
        if self
            .derived_field_by_node
            .get(&node_id)
            .is_some_and(|known| known == key)
        {
            Some(node_id)
        } else {
            None
        }
    }

    fn build_lineage(&self, input_fields: &[FieldKey], meta: &crate::ops::OpMeta) -> NodeLineage {
        let mut children = Vec::with_capacity(input_fields.len());
        for key in input_fields {
            if let Some(node_id) = self.node_id_from_derived_key(key) {
                children.push(self.nodes.get(node_id).map(|node| node.lineage).unwrap_or(
                    NodeLineage {
                        source_mask: 0,
                        has_multi_ancestor: false,
                        barrier_tainted: false,
                    },
                ));
                continue;
            }
            if is_const_field_name(&key.field) {
                children.push(NodeLineage {
                    source_mask: 0,
                    has_multi_ancestor: false,
                    barrier_tainted: false,
                });
                continue;
            }
            children.push(NodeLineage::from_source_slot(source_bit_slot(
                key.source_kind,
            )));
        }

        let op_barrier_semantic =
            meta.domain == Domain::Cs || matches!(meta.exec, ExecCapability::BarrierBatchExact);
        NodeLineage::merge(&children, op_barrier_semantic)
    }

    fn ensure_output_field(&mut self, node_id: usize) -> Result<FieldKey, CompileError> {
        if let Some(key) = self.derived_field_by_node.get(&node_id) {
            return Ok(key.clone());
        }
        let source_kind = self
            .nodes
            .get(node_id)
            .map(|n| source_kind_from_lineage(n.lineage, self.default_source_kind))
            .ok_or_else(|| CompileError::InvalidExpression {
                expr: self.expr.to_string(),
                reason: format!("internal lowering error: invalid node id {}", node_id),
            })?;
        let key = FieldKey {
            source_kind,
            field: format!("{DERIVED_FIELD_PREFIX}n{node_id}"),
        };
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.output_field = Some(key.clone());
        }
        self.required_fields.entry(key.clone()).or_insert(1);
        self.derived_field_by_node.insert(node_id, key.clone());
        Ok(key)
    }
}

fn collect_associative_terms<'a>(ast: &'a ExprAst, target: BinaryOp, out: &mut Vec<&'a ExprAst>) {
    if let ExprAst::Binary { op, lhs, rhs } = ast {
        if *op == target {
            collect_associative_terms(lhs, target, out);
            collect_associative_terms(rhs, target, out);
            return;
        }
    }
    out.push(ast);
}

#[inline]
fn canonicalize_first_two_inputs(commutative_first_two: bool, input_fields: &mut [FieldKey]) {
    if !commutative_first_two || input_fields.len() < 2 {
        return;
    }
    if input_fields[1] < input_fields[0] {
        input_fields.swap(0, 1);
    }
}

fn normalize_number_token(v: f64) -> Result<String, CompileError> {
    if !v.is_finite() {
        return Err(CompileError::InvalidExpression {
            expr: v.to_string(),
            reason: "argument number must be finite".to_string(),
        });
    }
    if v.fract() == 0.0 {
        Ok(format!("{v:.0}"))
    } else {
        Ok(v.to_string())
    }
}

fn evaluate_const_expr(ast: &ExprAst) -> Option<f64> {
    match ast {
        ExprAst::Number(v) => v.is_finite().then_some(*v),
        ExprAst::Unary { op, expr } => {
            let v = evaluate_const_expr(expr)?;
            let out = match op {
                UnaryOp::Plus => v,
                UnaryOp::Minus => -v,
                UnaryOp::Not => {
                    if v == 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            out.is_finite().then_some(out)
        }
        ExprAst::Binary { op, lhs, rhs } => {
            let lhs = evaluate_const_expr(lhs)?;
            let rhs = evaluate_const_expr(rhs)?;
            let out = match op {
                BinaryOp::Add => lhs + rhs,
                BinaryOp::Sub => lhs - rhs,
                BinaryOp::Mul => lhs * rhs,
                BinaryOp::Div => {
                    if rhs == 0.0 {
                        return None;
                    }
                    lhs / rhs
                }
                BinaryOp::Lt => {
                    if lhs < rhs {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Le => {
                    if lhs <= rhs {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Gt => {
                    if lhs > rhs {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Ge => {
                    if lhs >= rhs {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Eq => {
                    if lhs == rhs {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Ne => {
                    if lhs != rhs {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::And => {
                    if lhs != 0.0 && rhs != 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Or => {
                    if lhs != 0.0 || rhs != 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            out.is_finite().then_some(out)
        }
        ExprAst::Call { .. } | ExprAst::Identifier(_) => None,
    }
}

#[inline]
fn is_cs_neutralize_op(op_name: &str) -> bool {
    matches!(
        op_name,
        "cs_neutralize" | "cs_neutralize_ols" | "cs_neutralize_ols_multi"
    )
}

fn parse_standardize_token(token: &str, expr: &str, op_name: &str) -> Result<bool, CompileError> {
    let t = token.trim().to_ascii_lowercase();
    match t.as_str() {
        "1" | "1.0" | "true" => Ok(true),
        "0" | "0.0" | "false" => Ok(false),
        _ => Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: format!(
                "operator `{op_name}` keyword `standardize` expects true/false/1/0, got `{token}`"
            ),
        }),
    }
}

fn parse_cs_neutralize_call(
    op_name: &str,
    parsed: ParsedCall,
    expr: &str,
) -> Result<crate::ops::ParsedCompileArgs, CompileError> {
    let ParsedCall {
        mut positional_args,
        mut keyword_args,
    } = parsed;
    let regressor_count = match op_name {
        "cs_neutralize" => {
            if positional_args.len() != 1 {
                return Err(CompileError::InvalidArity {
                    name: op_name.to_string(),
                    expected: 1,
                    actual: positional_args.len(),
                });
            }
            0_u8
        }
        "cs_neutralize_ols" => {
            if positional_args.len() != 2 {
                return Err(CompileError::InvalidArity {
                    name: op_name.to_string(),
                    expected: 2,
                    actual: positional_args.len(),
                });
            }
            1_u8
        }
        "cs_neutralize_ols_multi" => {
            if positional_args.len() < 2 || positional_args.len() > 4 {
                return Err(CompileError::InvalidExpression {
                    expr: expr.to_string(),
                    reason: format!(
                        "operator `{op_name}` expects 2~4 positional args, got {}",
                        positional_args.len()
                    ),
                });
            }
            (positional_args.len() - 1) as u8
        }
        _ => unreachable!("non neutralize op should not call normalize_cs_neutralize_call"),
    };

    let standardize = if let Some(raw) = keyword_args.remove("standardize") {
        parse_standardize_token(&raw, expr, op_name)?
    } else {
        false
    };
    let group = keyword_args.remove("group");
    let weights = keyword_args.remove("weights");
    let has_group = group.is_some();
    let has_weights = weights.is_some();
    if let Some((key, _)) = keyword_args.into_iter().next() {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: format!("unexpected keyword arg `{key}` for `{op_name}`"),
        });
    }

    if let Some(group_field) = group {
        if parse_scalar_literal(&group_field).is_some() {
            return Err(CompileError::InvalidExpression {
                expr: expr.to_string(),
                reason: format!("operator `{op_name}` keyword `group` must be a field/expression"),
            });
        }
        positional_args.push(group_field);
    }
    if let Some(weight_field) = weights {
        if parse_scalar_literal(&weight_field).is_some() {
            return Err(CompileError::InvalidExpression {
                expr: expr.to_string(),
                reason: format!(
                    "operator `{op_name}` keyword `weights` must be a field/expression"
                ),
            });
        }
        positional_args.push(weight_field);
    }

    let expected_inputs =
        1 + regressor_count as usize + usize::from(has_group) + usize::from(has_weights);
    if expected_inputs != positional_args.len() {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: format!(
                "operator `{op_name}` has invalid neutralize argument layout (expected {expected_inputs}, got {})",
                positional_args.len()
            ),
        });
    }
    if expected_inputs > MAX_NODE_INPUTS {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: format!(
                "operator `{op_name}` exceeds max supported inputs {} when group/weights are used",
                MAX_NODE_INPUTS
            ),
        });
    }

    Ok(crate::ops::ParsedCompileArgs {
        fields: positional_args,
        param: LogicalParam::CsNeutralize {
            regressor_count,
            has_group,
            has_weights,
            standardize,
        },
    })
}

fn normalize_call_args(
    spec: CompileArgSpec,
    parsed: ParsedCall,
    expr: &str,
    op_name: &str,
) -> Result<Vec<String>, CompileError> {
    match spec {
        CompileArgSpec::FieldOnly => normalize_positional_only_args(parsed, expr, op_name, 1),
        CompileArgSpec::TwoFields => normalize_positional_only_args(parsed, expr, op_name, 2),
        CompileArgSpec::ThreeFields => normalize_positional_only_args(parsed, expr, op_name, 3),
        CompileArgSpec::Fields2To4 => normalize_positional_range_args(parsed, expr, op_name, 2, 4),
        CompileArgSpec::FieldWindow => {
            normalize_field_param_args(parsed, expr, op_name, "window", CompileArgSpec::FieldWindow)
        }
        CompileArgSpec::FieldLag => {
            normalize_field_param_args(parsed, expr, op_name, "lag", CompileArgSpec::FieldLag)
        }
        CompileArgSpec::TwoFieldsWindow => normalize_field_param_args(
            parsed,
            expr,
            op_name,
            "window",
            CompileArgSpec::TwoFieldsWindow,
        ),
        CompileArgSpec::FieldWindowQuantile => {
            normalize_field_window_quantile_args(parsed, expr, op_name)
        }
    }
}

fn normalize_positional_only_args(
    parsed: ParsedCall,
    expr: &str,
    op_name: &str,
    expected_arity: usize,
) -> Result<Vec<String>, CompileError> {
    let ParsedCall {
        positional_args,
        keyword_args,
    } = parsed;
    if !keyword_args.is_empty() {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: format!("operator `{op_name}` does not accept keyword args"),
        });
    }
    if positional_args.len() != expected_arity {
        return Err(CompileError::InvalidArity {
            name: op_name.to_string(),
            expected: expected_arity,
            actual: positional_args.len(),
        });
    }
    Ok(positional_args)
}

fn normalize_positional_range_args(
    parsed: ParsedCall,
    expr: &str,
    op_name: &str,
    min_arity: usize,
    max_arity: usize,
) -> Result<Vec<String>, CompileError> {
    let ParsedCall {
        positional_args,
        keyword_args,
    } = parsed;
    if !keyword_args.is_empty() {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: format!("operator `{op_name}` does not accept keyword args"),
        });
    }
    if positional_args.len() < min_arity || positional_args.len() > max_arity {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: format!(
                "operator `{op_name}` expects {min_arity}~{max_arity} positional args, got {}",
                positional_args.len()
            ),
        });
    }
    Ok(positional_args)
}

fn normalize_field_param_args(
    parsed: ParsedCall,
    expr: &str,
    op_name: &str,
    param_name: &str,
    spec: CompileArgSpec,
) -> Result<Vec<String>, CompileError> {
    let ParsedCall {
        mut positional_args,
        mut keyword_args,
    } = parsed;
    if let Some(value) = keyword_args.remove(param_name) {
        let expected_positional = spec.arity();
        if positional_args.len() == expected_positional {
            return Err(CompileError::InvalidExpression {
                expr: expr.to_string(),
                reason: format!("duplicate `{param_name}` in positional and keyword args"),
            });
        }
        if positional_args.len() == expected_positional - 1 {
            positional_args.push(value);
        } else {
            return Err(CompileError::InvalidArity {
                name: op_name.to_string(),
                expected: expected_positional,
                actual: positional_args.len(),
            });
        }
    }
    if let Some((key, _)) = keyword_args.into_iter().next() {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: format!("unexpected keyword arg `{key}` for `{op_name}`"),
        });
    }
    if positional_args.len() != spec.arity() {
        return Err(CompileError::InvalidArity {
            name: op_name.to_string(),
            expected: spec.arity(),
            actual: positional_args.len(),
        });
    }
    Ok(positional_args)
}

fn normalize_field_window_quantile_args(
    parsed: ParsedCall,
    expr: &str,
    op_name: &str,
) -> Result<Vec<String>, CompileError> {
    let ParsedCall {
        mut positional_args,
        mut keyword_args,
    } = parsed;
    if positional_args.len() > CompileArgSpec::FieldWindowQuantile.arity() {
        return Err(CompileError::InvalidArity {
            name: op_name.to_string(),
            expected: CompileArgSpec::FieldWindowQuantile.arity(),
            actual: positional_args.len(),
        });
    }

    let window_kw = keyword_args.remove("window");
    let q_kw = keyword_args.remove("q");
    if positional_args.len() >= 2 && window_kw.is_some() {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: "duplicate `window` in positional and keyword args".to_string(),
        });
    }
    if positional_args.len() >= 3 && q_kw.is_some() {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: "duplicate `q` in positional and keyword args".to_string(),
        });
    }

    if positional_args.len() == 1 {
        let Some(window) = window_kw else {
            return Err(CompileError::InvalidArity {
                name: op_name.to_string(),
                expected: CompileArgSpec::FieldWindowQuantile.arity(),
                actual: positional_args.len(),
            });
        };
        positional_args.push(window);
    }
    if positional_args.len() == 2 {
        let Some(q) = q_kw else {
            return Err(CompileError::InvalidArity {
                name: op_name.to_string(),
                expected: CompileArgSpec::FieldWindowQuantile.arity(),
                actual: positional_args.len(),
            });
        };
        positional_args.push(q);
    }
    if let Some((key, _)) = keyword_args.into_iter().next() {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: format!("unexpected keyword arg `{key}` for `{op_name}`"),
        });
    }
    if positional_args.len() != CompileArgSpec::FieldWindowQuantile.arity() {
        return Err(CompileError::InvalidArity {
            name: op_name.to_string(),
            expected: CompileArgSpec::FieldWindowQuantile.arity(),
            actual: positional_args.len(),
        });
    }
    Ok(positional_args)
}

fn build_ready_required_fields(
    logical: &LogicalPlan,
    field_slot_by_key: &HashMap<FieldKey, usize>,
    field_count: usize,
) -> Vec<usize> {
    let mut producer_by_output_field: HashMap<FieldKey, usize> = HashMap::new();
    for node in &logical.nodes {
        if let Some(key) = &node.output_field {
            producer_by_output_field.insert(key.clone(), node.node_id);
        }
    }
    let mut seen_nodes = vec![false; logical.nodes.len()];
    let mut seen = vec![false; field_count];
    let mut out = Vec::new();
    let mut stack = logical.outputs.clone();

    while let Some(node_idx) = stack.pop() {
        let Some(node) = logical.nodes.get(node_idx) else {
            continue;
        };
        if seen_nodes[node_idx] {
            continue;
        }
        seen_nodes[node_idx] = true;
        for key in &node.input_fields {
            if is_derived_field_name(&key.field) {
                if let Some(producer_idx) = producer_by_output_field.get(key).copied() {
                    stack.push(producer_idx);
                    continue;
                }
            }
            if is_const_field_name(&key.field) {
                continue;
            }
            let slot = *field_slot_by_key
                .get(key)
                .expect("field slots are built from required_fields");
            if !seen[slot] {
                seen[slot] = true;
                out.push(slot);
            }
        }
    }
    out
}

fn register_output_name(
    output_name_to_slot: &mut HashMap<String, usize>,
    output_aliases: &mut Vec<(String, usize)>,
    nodes: &mut [LogicalNode],
    output_name: String,
    output_slot: usize,
    expr: &str,
) -> Result<(), CompileError> {
    if let Some(prev_slot) = output_name_to_slot.get(&output_name).copied() {
        if prev_slot != output_slot {
            return Err(CompileError::InvalidExpression {
                expr: expr.to_string(),
                reason: format!(
                    "output name `{output_name}` conflicts across different expressions"
                ),
            });
        }
        return Ok(());
    }
    let has_name_for_slot = output_name_to_slot
        .values()
        .any(|slot| *slot == output_slot);
    output_name_to_slot.insert(output_name.clone(), output_slot);
    if has_name_for_slot {
        output_aliases.push((output_name, output_slot));
    } else if let Some(node) = nodes.get_mut(output_slot) {
        node.output_name = output_name;
    }
    Ok(())
}

#[inline]
fn param_value(param: LogicalParam) -> usize {
    match param {
        LogicalParam::None => 1,
        LogicalParam::Window(v) | LogicalParam::Lag(v) => v,
        LogicalParam::WindowQuantile { window, .. } => window,
        LogicalParam::CsNeutralize { .. } => 1,
    }
}

fn history_len_from_spec(spec: HistorySpec, param: LogicalParam) -> usize {
    let raw = param_value(param);
    match spec {
        HistorySpec::One => 1,
        HistorySpec::Param => raw.max(1),
        HistorySpec::ParamPlusOne => raw.max(1) + 1,
    }
}

fn parse_field_key(
    default_source_kind: SourceKind,
    raw: &str,
    expr: &str,
) -> Result<FieldKey, CompileError> {
    let token = raw.trim();
    if token.is_empty() {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: "empty field identifier".to_string(),
        });
    }

    let (source_kind, field) = if let Some((source, field)) = token.split_once('.') {
        if field.contains('.') {
            return Err(CompileError::InvalidExpression {
                expr: expr.to_string(),
                reason: format!("invalid field identifier `{token}`"),
            });
        }
        (parse_source_kind(source, expr)?, field)
    } else {
        (default_source_kind, token)
    };
    let field = normalize_field_name(field, expr)?;
    Ok(FieldKey { source_kind, field })
}

fn parse_source_kind(raw: &str, expr: &str) -> Result<SourceKind, CompileError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "bar" => Ok(SourceKind::Bar),
        "trade_tick" | "tradetick" => Ok(SourceKind::TradeTick),
        "quote_tick" | "quotetick" => Ok(SourceKind::QuoteTick),
        "orderbook_snapshot" | "orderbooksnapshot" | "order_book_snapshot" => {
            Ok(SourceKind::OrderBookSnapshot)
        }
        "data" => Ok(SourceKind::Data),
        _ => Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: format!("unknown source kind `{}`", raw.trim()),
        }),
    }
}

fn parse_scalar_literal(raw: &str) -> Option<f64> {
    let token = raw.trim();
    if token.is_empty() {
        return None;
    }
    let value = token.parse::<f64>().ok()?;
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

fn make_const_field_key(value: f64) -> FieldKey {
    FieldKey {
        source_kind: SourceKind::Data,
        field: format!("{CONST_FIELD_PREFIX}{:016x}", value.to_bits()),
    }
}

fn const_value_from_field_key(key: &FieldKey) -> Option<f64> {
    if key.source_kind != SourceKind::Data {
        return None;
    }
    let hex = key.field.strip_prefix(CONST_FIELD_PREFIX)?;
    let bits = u64::from_str_radix(hex, 16).ok()?;
    let value = f64::from_bits(bits);
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

#[inline]
const fn source_bit_slot(kind: SourceKind) -> u16 {
    match kind {
        SourceKind::Bar => 0,
        SourceKind::TradeTick => 1,
        SourceKind::QuoteTick => 2,
        SourceKind::OrderBookSnapshot => 3,
        SourceKind::Data => 4,
    }
}

fn source_kind_from_lineage(lineage: NodeLineage, fallback: SourceKind) -> SourceKind {
    if lineage.source_mask.count_ones() == 1 {
        let bit = lineage.source_mask.trailing_zeros() as u16;
        source_kind_from_lineage_bit(bit).unwrap_or(fallback)
    } else {
        SourceKind::Data
    }
}

fn source_kind_from_lineage_bit(bit: u16) -> Option<SourceKind> {
    match bit {
        0 => Some(SourceKind::Bar),
        1 => Some(SourceKind::TradeTick),
        2 => Some(SourceKind::QuoteTick),
        3 => Some(SourceKind::OrderBookSnapshot),
        4 => Some(SourceKind::Data),
        _ => None,
    }
}

fn normalize_field_name(raw: &str, expr: &str) -> Result<String, CompileError> {
    let field = raw.trim().to_ascii_lowercase();
    if field.is_empty() || !field.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: format!("invalid field identifier `{}`", raw.trim()),
        });
    }
    Ok(field)
}

fn canonical_field_name(source_kind: SourceKind, field: &str) -> String {
    let source = match source_kind {
        SourceKind::Bar => "bar",
        SourceKind::TradeTick => "trade_tick",
        SourceKind::QuoteTick => "quote_tick",
        SourceKind::OrderBookSnapshot => "orderbook_snapshot",
        SourceKind::Data => "data",
    };
    format!("{source}.{field}")
}

fn catalog_contains_field(catalog: &InputFieldCatalog, key: &FieldKey) -> bool {
    catalog.contains(&key.field)
        || catalog.contains(&canonical_field_name(key.source_kind, &key.field))
}
