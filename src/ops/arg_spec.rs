use crate::error::CompileError;
use crate::plan::LogicalParam;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedCompileArgs {
    pub fields: Vec<String>,
    pub param: LogicalParam,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompileArgSpec {
    FieldOnly,
    TwoFields,
    ThreeFields,
    Fields2To4,
    FieldWindow,
    FieldLag,
    TwoFieldsWindow,
    FieldWindowQuantile,
}

impl CompileArgSpec {
    #[inline]
    pub const fn arity(self) -> usize {
        match self {
            Self::FieldOnly => 1,
            Self::TwoFields => 2,
            Self::ThreeFields => 3,
            Self::Fields2To4 => 4,
            Self::FieldWindow => 2,
            Self::FieldLag => 2,
            Self::TwoFieldsWindow => 3,
            Self::FieldWindowQuantile => 3,
        }
    }

    #[inline]
    pub fn parse(self, args: &[String], expr: &str) -> Result<ParsedCompileArgs, CompileError> {
        match self {
            Self::FieldOnly => parse_field_only(args, expr),
            Self::TwoFields => parse_two_fields(args, expr),
            Self::ThreeFields => parse_three_fields(args, expr),
            Self::Fields2To4 => parse_fields_2_to_4(args, expr),
            Self::FieldWindow => parse_field_window(args, expr),
            Self::FieldLag => parse_field_lag(args, expr),
            Self::TwoFieldsWindow => parse_two_fields_window(args, expr),
            Self::FieldWindowQuantile => parse_field_window_quantile(args, expr),
        }
    }
}

fn parse_field_only(args: &[String], expr: &str) -> Result<ParsedCompileArgs, CompileError> {
    let field = parse_field_or_scalar(&args[0], expr)?;
    Ok(ParsedCompileArgs {
        fields: vec![field],
        param: LogicalParam::None,
    })
}

fn parse_two_fields(args: &[String], expr: &str) -> Result<ParsedCompileArgs, CompileError> {
    let lhs = parse_field_or_scalar(&args[0], expr)?;
    let rhs = parse_field_or_scalar(&args[1], expr)?;
    Ok(ParsedCompileArgs {
        fields: vec![lhs, rhs],
        param: LogicalParam::None,
    })
}

fn parse_three_fields(args: &[String], expr: &str) -> Result<ParsedCompileArgs, CompileError> {
    let a = parse_field_or_scalar(&args[0], expr)?;
    let b = parse_field_or_scalar(&args[1], expr)?;
    let c = parse_field_or_scalar(&args[2], expr)?;
    Ok(ParsedCompileArgs {
        fields: vec![a, b, c],
        param: LogicalParam::None,
    })
}

fn parse_fields_2_to_4(args: &[String], expr: &str) -> Result<ParsedCompileArgs, CompileError> {
    if !(2..=4).contains(&args.len()) {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: "operator expects 2~4 field arguments".to_string(),
        });
    }
    let mut fields = Vec::with_capacity(args.len());
    for arg in args {
        fields.push(parse_identifier(arg, expr)?);
    }
    Ok(ParsedCompileArgs {
        fields,
        param: LogicalParam::None,
    })
}

fn parse_field_window(args: &[String], expr: &str) -> Result<ParsedCompileArgs, CompileError> {
    let field = parse_identifier(&args[0], expr)?;
    let window = parse_positive_usize(&args[1], expr, "window")?;
    Ok(ParsedCompileArgs {
        fields: vec![field],
        param: LogicalParam::Window(window),
    })
}

fn parse_field_lag(args: &[String], expr: &str) -> Result<ParsedCompileArgs, CompileError> {
    let field = parse_identifier(&args[0], expr)?;
    let lag = parse_positive_usize(&args[1], expr, "lag")?;
    Ok(ParsedCompileArgs {
        fields: vec![field],
        param: LogicalParam::Lag(lag),
    })
}

fn parse_two_fields_window(args: &[String], expr: &str) -> Result<ParsedCompileArgs, CompileError> {
    let lhs = parse_identifier(&args[0], expr)?;
    let rhs = parse_identifier(&args[1], expr)?;
    let window = parse_positive_usize(&args[2], expr, "window")?;
    Ok(ParsedCompileArgs {
        fields: vec![lhs, rhs],
        param: LogicalParam::Window(window),
    })
}

fn parse_field_window_quantile(
    args: &[String],
    expr: &str,
) -> Result<ParsedCompileArgs, CompileError> {
    let field = parse_identifier(&args[0], expr)?;
    let window = parse_positive_usize(&args[1], expr, "window")?;
    let q = parse_bounded_f64(&args[2], expr, "q", 0.0, 1.0)?;
    Ok(ParsedCompileArgs {
        fields: vec![field],
        param: LogicalParam::WindowQuantile {
            window,
            q_bits: q.to_bits(),
        },
    })
}

fn parse_identifier(raw: &str, expr: &str) -> Result<String, CompileError> {
    let name = raw.trim();
    if name.is_empty() {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: "empty field identifier".to_string(),
        });
    }
    if name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
    {
        Ok(name.to_string())
    } else {
        Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: format!("invalid field identifier `{name}`"),
        })
    }
}

fn parse_field_or_scalar(raw: &str, expr: &str) -> Result<String, CompileError> {
    let token = raw.trim();
    if token.is_empty() {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: "empty field identifier".to_string(),
        });
    }
    if token
        .parse::<f64>()
        .ok()
        .is_some_and(|value| value.is_finite())
    {
        return Ok(token.to_string());
    }
    parse_identifier(raw, expr)
}

fn parse_usize(raw: &str, expr: &str) -> Result<usize, CompileError> {
    raw.trim()
        .parse::<usize>()
        .map_err(|_| CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: format!("invalid integer parameter `{}`", raw.trim()),
        })
}

fn parse_positive_usize(raw: &str, expr: &str, param_name: &str) -> Result<usize, CompileError> {
    let value = parse_usize(raw, expr)?;
    if value == 0 {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: format!("{param_name} must be >= 1"),
        });
    }
    Ok(value)
}

fn parse_bounded_f64(
    raw: &str,
    expr: &str,
    param_name: &str,
    min: f64,
    max: f64,
) -> Result<f64, CompileError> {
    let value = raw
        .trim()
        .parse::<f64>()
        .map_err(|_| CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: format!("invalid float parameter `{}`", raw.trim()),
        })?;
    if !value.is_finite() || value < min || value > max {
        return Err(CompileError::InvalidExpression {
            expr: expr.to_string(),
            reason: format!("{param_name} must be finite and in [{min}, {max}]"),
        });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|p| p.to_string()).collect()
    }

    #[test]
    fn field_window_requires_positive_window() {
        let err =
            parse_field_window(&to_args(&["close", "0"]), "ts_mean(close, 0)").expect_err("zero");
        assert!(matches!(err, CompileError::InvalidExpression { .. }));
    }

    #[test]
    fn field_lag_requires_positive_lag() {
        let err =
            parse_field_lag(&to_args(&["close", "0"]), "ts_delta(close, 0)").expect_err("zero");
        assert!(matches!(err, CompileError::InvalidExpression { .. }));
    }

    #[test]
    fn two_fields_window_requires_positive_window() {
        let err = parse_two_fields_window(
            &to_args(&["close", "volume", "0"]),
            "ts_corr(close, volume, 0)",
        )
        .expect_err("zero");
        assert!(matches!(err, CompileError::InvalidExpression { .. }));
    }

    #[test]
    fn field_window_quantile_requires_valid_q_range() {
        let err = parse_field_window_quantile(
            &to_args(&["close", "5", "1.1"]),
            "ts_quantile(close, 5, 1.1)",
        )
        .expect_err("q out of range");
        assert!(matches!(err, CompileError::InvalidExpression { .. }));
    }

    #[test]
    fn fields_2_to_4_accepts_only_range() {
        let ok = parse_fields_2_to_4(
            &to_args(&["close", "volume", "turnover"]),
            "cs_neutralize_ols_multi(close, volume, turnover)",
        )
        .expect("3 fields should pass");
        assert_eq!(ok.fields.len(), 3);

        assert!(
            parse_fields_2_to_4(&to_args(&["close"]), "cs_neutralize_ols_multi(close)").is_err()
        );
        assert!(parse_fields_2_to_4(
            &to_args(&["a", "b", "c", "d", "e"]),
            "cs_neutralize_ols_multi(a,b,c,d,e)"
        )
        .is_err());
    }
}
