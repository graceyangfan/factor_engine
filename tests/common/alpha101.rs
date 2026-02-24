use factor_engine::ops::{CompileArgSpec, OperatorRegistry};
use factor_engine::plan::{FieldKey, LogicalPlan};
use factor_engine::types::SourceKind;
use factor_engine::{BarLite, EventEnvelope, FeatureFrame, Payload};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::process::{Command, Stdio};

#[allow(dead_code)]
const TOL: f64 = 1e-8;

#[derive(Debug, Clone, Serialize)]
pub struct OfflineInputRow {
    pub ts: i64,
    pub instrument_slot: u32,
    pub fields: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LegacyNeutralizeSpec {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub regressors: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub standardize: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LegacyOfflineExpr {
    pub output: String,
    pub op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lhs: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lag: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neutralize: Option<LegacyNeutralizeSpec>,
}

#[derive(Debug, Serialize)]
pub struct LegacyOfflinePayload {
    pub rows: Vec<OfflineInputRow>,
    pub expressions: Vec<LegacyOfflineExpr>,
}

#[derive(Debug, Deserialize)]
struct OfflineRow {
    output: String,
    instrument_slot: u32,
    ts: i64,
    value: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct OfflineResponse {
    rows: Vec<OfflineRow>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompareKey {
    pub output: String,
    pub instrument_slot: u32,
    pub ts: i64,
}

pub fn python_polars_available() -> bool {
    Command::new("python3")
        .arg("-c")
        .arg("import polars")
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

pub fn run_polars_offline(payload: &LegacyOfflinePayload) -> HashMap<CompareKey, Option<f64>> {
    let script = format!(
        "{}/tests/data/polars_offline_baseline.py",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut child = Command::new("python3")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn python3");
    {
        let stdin = child.stdin.as_mut().expect("stdin unavailable");
        let bytes = serde_json::to_vec(payload).expect("serialize payload");
        stdin.write_all(&bytes).expect("write payload");
    }
    let output = child.wait_with_output().expect("python execution failed");
    if !output.status.success() {
        panic!(
            "python baseline failed: status={:?} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let parsed: OfflineResponse = serde_json::from_slice(&output.stdout).unwrap_or_else(|err| {
        panic!(
            "parse python output failed: {err}: {}",
            String::from_utf8_lossy(&output.stdout)
        )
    });
    parsed
        .rows
        .into_iter()
        .map(|row| {
            (
                CompareKey {
                    output: row.output,
                    instrument_slot: row.instrument_slot,
                    ts: row.ts,
                },
                row.value,
            )
        })
        .collect()
}

pub fn normalize(v: f64) -> Option<f64> {
    if v.is_finite() {
        Some(v)
    } else {
        None
    }
}

#[allow(dead_code)]
pub fn assert_close(lhs: Option<f64>, rhs: Option<f64>, key: &CompareKey) {
    match (lhs, rhs) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            let abs = (a - b).abs();
            let scale = a.abs().max(b.abs()).max(1.0);
            assert!(
                abs <= TOL || abs <= TOL * 10.0 * scale,
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

pub fn record_frame_results(
    out: &mut HashMap<CompareKey, Option<f64>>,
    frame: &FeatureFrame,
    outputs: &[String],
    universe: &[u32],
    ts: i64,
) {
    for output in outputs {
        let factor_idx = frame.factor_idx(output).expect("missing output");
        for (instrument_idx, &instrument_slot) in universe.iter().enumerate() {
            let value = frame
                .value_at(instrument_idx, factor_idx)
                .and_then(normalize);
            out.insert(
                CompareKey {
                    output: output.clone(),
                    instrument_slot,
                    ts,
                },
                value,
            );
        }
    }
}

pub fn retain_outputs(
    rows: &HashMap<CompareKey, Option<f64>>,
    outputs: &[String],
) -> HashMap<CompareKey, Option<f64>> {
    let wanted: BTreeSet<&str> = outputs.iter().map(|s| s.as_str()).collect();
    rows.iter()
        .filter(|(k, _)| wanted.contains(k.output.as_str()))
        .map(|(k, v)| (k.clone(), *v))
        .collect()
}

pub fn retain_from_ts(
    rows: &HashMap<CompareKey, Option<f64>>,
    min_ts: i64,
) -> HashMap<CompareKey, Option<f64>> {
    rows.iter()
        .filter(|(k, _)| k.ts >= min_ts)
        .map(|(k, v)| (k.clone(), *v))
        .collect()
}

pub fn bar_event_ohlcv(
    ts: i64,
    instrument_slot: u32,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
) -> EventEnvelope {
    EventEnvelope {
        ts_event_ns: ts,
        ts_init_ns: ts,
        seq: 1,
        source_slot: 0,
        instrument_slot,
        quality_flags: 0,
        payload: Payload::Bar(BarLite {
            open,
            high,
            low,
            close,
            volume,
        }),
    }
}

pub fn expr(
    output: &str,
    op: &str,
    field: Option<&str>,
    lhs: Option<&str>,
    rhs: Option<&str>,
    window: Option<usize>,
    lag: Option<usize>,
) -> LegacyOfflineExpr {
    LegacyOfflineExpr {
        output: output.to_string(),
        op: op.to_string(),
        field: field.map(ToString::to_string),
        lhs: lhs.map(ToString::to_string),
        rhs: rhs.map(ToString::to_string),
        window,
        lag,
        neutralize: None,
    }
}

#[allow(dead_code)]
const DERIVED_FIELD_PREFIX: &str = "__derived__";
#[allow(dead_code)]
const CONST_FIELD_PREFIX: &str = "__const__";

#[allow(dead_code)]
pub fn build_legacy_payload_from_logical(
    rows: Vec<OfflineInputRow>,
    logical: &LogicalPlan,
) -> LegacyOfflinePayload {
    let mut derived_output_name_by_field = HashMap::new();
    for node in &logical.nodes {
        if let Some(output_field) = &node.output_field {
            derived_output_name_by_field
                .insert(output_field.field.clone(), node.output_name.clone());
        }
    }

    let mut expressions = Vec::with_capacity(logical.nodes.len() + logical.output_aliases.len());
    for node in &logical.nodes {
        let meta = OperatorRegistry::get_by_op(node.op).unwrap_or_else(|| {
            panic!("missing op meta for {:?}", node.op);
        });
        let mut offline = LegacyOfflineExpr {
            output: node.output_name.clone(),
            op: meta.name.to_string(),
            field: None,
            lhs: None,
            rhs: None,
            window: None,
            lag: None,
            neutralize: None,
        };
        let operands: Vec<String> = node
            .input_fields
            .iter()
            .map(|key| offline_operand(key, &derived_output_name_by_field))
            .collect();

        if let Some((regressor_count, has_group, has_weights, standardize)) =
            node.param.cs_neutralize()
        {
            let mut cursor = 1usize;
            let regressors = (0..regressor_count as usize)
                .map(|_| {
                    let value = expect_operand(&operands, cursor, meta.name);
                    cursor += 1;
                    value
                })
                .collect();
            let group = if has_group {
                let value = expect_operand(&operands, cursor, meta.name);
                cursor += 1;
                Some(value)
            } else {
                None
            };
            let weights = if has_weights {
                Some(expect_operand(&operands, cursor, meta.name))
            } else {
                None
            };
            offline.field = Some(expect_operand(&operands, 0, meta.name));
            offline.neutralize = Some(LegacyNeutralizeSpec {
                regressors,
                group,
                weights,
                standardize: if standardize { Some(true) } else { None },
            });
            expressions.push(offline);
            continue;
        }

        match meta.arg_spec {
            CompileArgSpec::FieldOnly => {
                offline.field = Some(expect_operand(&operands, 0, meta.name));
            }
            CompileArgSpec::TwoFields => {
                if meta.name.starts_with("elem_") {
                    offline.lhs = Some(expect_operand(&operands, 0, meta.name));
                    offline.rhs = Some(expect_operand(&operands, 1, meta.name));
                } else {
                    offline.field = Some(expect_operand(&operands, 0, meta.name));
                    offline.lhs = Some(expect_operand(&operands, 1, meta.name));
                }
            }
            CompileArgSpec::ThreeFields => {
                offline.field = Some(expect_operand(&operands, 0, meta.name));
                offline.lhs = Some(expect_operand(&operands, 1, meta.name));
                offline.rhs = Some(expect_operand(&operands, 2, meta.name));
            }
            CompileArgSpec::FieldWindow => {
                offline.field = Some(expect_operand(&operands, 0, meta.name));
                offline.window = node.param.window();
            }
            CompileArgSpec::FieldLag => {
                offline.field = Some(expect_operand(&operands, 0, meta.name));
                offline.lag = node.param.lag();
            }
            CompileArgSpec::TwoFieldsWindow => {
                offline.lhs = Some(expect_operand(&operands, 0, meta.name));
                offline.rhs = Some(expect_operand(&operands, 1, meta.name));
                offline.window = node.param.window();
            }
            CompileArgSpec::FieldWindowQuantile => {
                offline.field = Some(expect_operand(&operands, 0, meta.name));
                offline.window = node.param.window();
                let q = node.param.quantile().unwrap_or_else(|| {
                    panic!("operator `{}` missing quantile param", meta.name);
                });
                offline.rhs = Some(normalize_scalar(q));
            }
            CompileArgSpec::Fields2To4 => {
                panic!(
                    "operator `{}` with variable arity is not supported by legacy offline payload",
                    meta.name
                );
            }
        }

        expressions.push(offline);
    }

    for (alias, slot) in &logical.output_aliases {
        let output_name = logical
            .nodes
            .get(*slot)
            .unwrap_or_else(|| panic!("invalid output alias slot {}", slot))
            .output_name
            .clone();
        expressions.push(expr(
            alias,
            "elem_mul",
            None,
            Some(&output_name),
            Some("1"),
            None,
            None,
        ));
    }

    LegacyOfflinePayload { rows, expressions }
}

#[allow(dead_code)]
fn expect_operand(operands: &[String], idx: usize, op_name: &str) -> String {
    operands
        .get(idx)
        .cloned()
        .unwrap_or_else(|| panic!("operator `{}` missing operand index {}", op_name, idx))
}

#[allow(dead_code)]
fn offline_operand(
    key: &FieldKey,
    derived_output_name_by_field: &HashMap<String, String>,
) -> String {
    if key.source_kind == SourceKind::Data {
        if let Some(value) = decode_const_from_field(&key.field) {
            return normalize_scalar(value);
        }
    }
    if key.field.starts_with(DERIVED_FIELD_PREFIX) {
        return derived_output_name_by_field
            .get(&key.field)
            .cloned()
            .unwrap_or_else(|| panic!("missing derived mapping for field `{}`", key.field));
    }
    key.field.clone()
}

#[allow(dead_code)]
fn decode_const_from_field(field: &str) -> Option<f64> {
    let hex = field.strip_prefix(CONST_FIELD_PREFIX)?;
    let bits = u64::from_str_radix(hex, 16).ok()?;
    let value = f64::from_bits(bits);
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

#[allow(dead_code)]
fn normalize_scalar(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{v:.0}")
    } else {
        v.to_string()
    }
}
