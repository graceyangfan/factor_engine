use crate::compile_expr::ExprAst;
use crate::error::CompileError;
use crate::ops::{Domain, ExecCapability, OperatorRegistry};
use crate::plan::NodeLineage;
use crate::types::SourceKind;

pub fn infer_lineage(
    ast: &ExprAst,
    default_source_kind: SourceKind,
) -> Result<NodeLineage, CompileError> {
    match ast {
        ExprAst::Identifier(name) => {
            let source_kind = parse_identifier_source(name, default_source_kind)?;
            Ok(NodeLineage::from_source_slot(source_bit_slot(source_kind)))
        }
        ExprAst::Number(_) => Ok(NodeLineage {
            source_mask: 0,
            has_multi_ancestor: false,
            barrier_tainted: false,
        }),
        ExprAst::Unary { expr, .. } => infer_lineage(expr, default_source_kind),
        ExprAst::Binary { lhs, rhs, .. } => {
            let lhs = infer_lineage(lhs, default_source_kind)?;
            let rhs = infer_lineage(rhs, default_source_kind)?;
            Ok(NodeLineage::merge(&[lhs, rhs], false))
        }
        ExprAst::Call { name, args, kwargs } => {
            let op_name = name.to_ascii_lowercase();
            let meta =
                OperatorRegistry::get(&op_name).ok_or_else(|| CompileError::UnknownOperator {
                    name: op_name.clone(),
                })?;
            let mut children = Vec::with_capacity(args.len() + kwargs.len());
            for arg in args {
                children.push(infer_lineage(arg, default_source_kind)?);
            }
            for value in kwargs.values() {
                children.push(infer_lineage(value, default_source_kind)?);
            }
            let op_barrier_semantic =
                meta.domain == Domain::Cs || matches!(meta.exec, ExecCapability::BarrierBatchExact);
            Ok(NodeLineage::merge(&children, op_barrier_semantic))
        }
    }
}

fn parse_identifier_source(
    identifier: &str,
    default_source_kind: SourceKind,
) -> Result<SourceKind, CompileError> {
    let token = identifier.trim();
    if token.is_empty() {
        return Err(CompileError::InvalidExpression {
            expr: identifier.to_string(),
            reason: "empty identifier".to_string(),
        });
    }
    if let Some((source, _field)) = token.split_once('.') {
        parse_source_kind_hint(source)
    } else {
        Ok(default_source_kind)
    }
}

fn parse_source_kind_hint(raw: &str) -> Result<SourceKind, CompileError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "bar" => Ok(SourceKind::Bar),
        "trade_tick" | "tradetick" => Ok(SourceKind::TradeTick),
        "quote_tick" | "quotetick" => Ok(SourceKind::QuoteTick),
        "orderbook_snapshot" | "orderbooksnapshot" | "order_book_snapshot" => {
            Ok(SourceKind::OrderBookSnapshot)
        }
        "data" => Ok(SourceKind::Data),
        _ => Err(CompileError::InvalidExpression {
            expr: raw.to_string(),
            reason: format!("unknown source kind `{}`", raw.trim()),
        }),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile_expr::parse_expression;

    #[test]
    fn lineage_is_tainted_when_any_ancestor_is_multi_source() {
        let ast =
            parse_expression("ts_corr(ts_mean(bar.close, 2), ts_mean(quote_tick.bid_price, 2), 3)")
                .expect("parse should succeed");
        let lineage =
            infer_lineage(&ast, SourceKind::Bar).expect("lineage inference should succeed");
        assert_eq!(lineage.source_cardinality(), 2);
        assert!(lineage.has_multi_ancestor);
        assert!(lineage.barrier_tainted);
    }

    #[test]
    fn lineage_stays_single_for_single_source_nested_tree() {
        let ast = parse_expression("ts_corr(ts_mean(bar.close, 2), ts_delta(bar.volume, 1), 3)")
            .expect("parse should succeed");
        let lineage =
            infer_lineage(&ast, SourceKind::Bar).expect("lineage inference should succeed");
        assert_eq!(lineage.source_cardinality(), 1);
        assert!(!lineage.has_multi_ancestor);
        assert!(!lineage.barrier_tainted);
    }

    #[test]
    fn cs_node_is_barrier_tainted_even_when_single_source() {
        let ast = parse_expression("cs_rank(ts_mean(bar.close, 2))").expect("parse should succeed");
        let lineage =
            infer_lineage(&ast, SourceKind::Bar).expect("lineage inference should succeed");
        assert_eq!(lineage.source_cardinality(), 1);
        assert!(!lineage.has_multi_ancestor);
        assert!(lineage.barrier_tainted);
    }
}
