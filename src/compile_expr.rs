use crate::error::CompileError;
use std::collections::BTreeMap;
use std::iter::Peekable;
use std::str::Chars;

#[derive(Debug, Clone, PartialEq)]
pub enum ExprAst {
    Call {
        name: String,
        args: Vec<ExprAst>,
        kwargs: BTreeMap<String, ExprAst>,
    },
    Identifier(String),
    Number(f64),
    Unary {
        op: UnaryOp,
        expr: Box<ExprAst>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<ExprAst>,
        rhs: Box<ExprAst>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Plus,
    Minus,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Number(f64),
    Plus,
    Minus,
    Star,
    Slash,
    Amp,
    Pipe,
    Tilde,
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    NotEq,
    Comma,
    Equal,
    LParen,
    RParen,
    Eof,
}

struct Lexer<'a> {
    chars: Peekable<Chars<'a>>,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            chars: input.chars().peekable(),
        }
    }

    fn next_token(&mut self) -> Result<Token, CompileError> {
        self.skip_ws();
        let Some(&ch) = self.chars.peek() else {
            return Ok(Token::Eof);
        };
        match ch {
            '+' => {
                self.chars.next();
                Ok(Token::Plus)
            }
            '-' => {
                self.chars.next();
                Ok(Token::Minus)
            }
            '*' => {
                self.chars.next();
                Ok(Token::Star)
            }
            '/' => {
                self.chars.next();
                Ok(Token::Slash)
            }
            '&' => {
                self.chars.next();
                Ok(Token::Amp)
            }
            '|' => {
                self.chars.next();
                Ok(Token::Pipe)
            }
            '~' => {
                self.chars.next();
                Ok(Token::Tilde)
            }
            '<' => {
                self.chars.next();
                if matches!(self.chars.peek(), Some('=')) {
                    self.chars.next();
                    Ok(Token::Le)
                } else {
                    Ok(Token::Lt)
                }
            }
            '>' => {
                self.chars.next();
                if matches!(self.chars.peek(), Some('=')) {
                    self.chars.next();
                    Ok(Token::Ge)
                } else {
                    Ok(Token::Gt)
                }
            }
            '!' => {
                self.chars.next();
                if matches!(self.chars.peek(), Some('=')) {
                    self.chars.next();
                    Ok(Token::NotEq)
                } else {
                    Err(CompileError::InvalidExpression {
                        expr: "!".to_string(),
                        reason: "unexpected character `!` (did you mean `!=`?)".to_string(),
                    })
                }
            }
            '(' => {
                self.chars.next();
                Ok(Token::LParen)
            }
            ')' => {
                self.chars.next();
                Ok(Token::RParen)
            }
            ',' => {
                self.chars.next();
                Ok(Token::Comma)
            }
            '=' => {
                self.chars.next();
                if matches!(self.chars.peek(), Some('=')) {
                    self.chars.next();
                    Ok(Token::EqEq)
                } else {
                    Ok(Token::Equal)
                }
            }
            c if is_ident_start(c) => Ok(Token::Ident(self.read_ident())),
            c if c.is_ascii_digit() || c == '.' => {
                let raw = self.read_number();
                let num = raw
                    .parse::<f64>()
                    .map_err(|_| CompileError::InvalidExpression {
                        expr: raw.clone(),
                        reason: format!("invalid number `{raw}`"),
                    })?;
                Ok(Token::Number(num))
            }
            other => Err(CompileError::InvalidExpression {
                expr: other.to_string(),
                reason: format!("unexpected character `{other}`"),
            }),
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.chars.peek(), Some(c) if c.is_ascii_whitespace()) {
            self.chars.next();
        }
    }

    fn read_ident(&mut self) -> String {
        let mut out = String::new();
        while let Some(&c) = self.chars.peek() {
            if is_ident_continue(c) {
                out.push(c);
                self.chars.next();
            } else {
                break;
            }
        }
        out
    }

    fn read_number(&mut self) -> String {
        let mut out = String::new();
        let mut seen_dot = false;
        let mut seen_exp = false;

        while let Some(&c) = self.chars.peek() {
            if c.is_ascii_digit() {
                out.push(c);
                self.chars.next();
                continue;
            }
            if c == '.' && !seen_dot && !seen_exp {
                seen_dot = true;
                out.push(c);
                self.chars.next();
                continue;
            }
            if (c == 'e' || c == 'E') && !seen_exp {
                seen_exp = true;
                out.push(c);
                self.chars.next();
                if let Some(&sign) = self.chars.peek() {
                    if sign == '+' || sign == '-' {
                        out.push(sign);
                        self.chars.next();
                    }
                }
                continue;
            }
            break;
        }
        out
    }
}

#[inline]
fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

#[inline]
fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '.'
}

pub fn parse_expression(source: &str) -> Result<ExprAst, CompileError> {
    let mut parser = Parser::new(source);
    let expr = parser.parse_expr()?;
    match parser.next_token()? {
        Token::Eof => Ok(expr),
        other => Err(CompileError::InvalidExpression {
            expr: source.to_string(),
            reason: format!("unexpected trailing token: {other:?}"),
        }),
    }
}

struct Parser<'a> {
    source: &'a str,
    lexer: Lexer<'a>,
    lookahead: Option<Token>,
}

impl<'a> Parser<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            lexer: Lexer::new(source),
            lookahead: None,
        }
    }

    fn next_token(&mut self) -> Result<Token, CompileError> {
        if let Some(tok) = self.lookahead.take() {
            return Ok(tok);
        }
        self.lexer.next_token()
    }

    fn peek_token(&mut self) -> Result<Token, CompileError> {
        if self.lookahead.is_none() {
            self.lookahead = Some(self.lexer.next_token()?);
        }
        Ok(self.lookahead.clone().expect("lookahead just initialized"))
    }

    fn parse_expr(&mut self) -> Result<ExprAst, CompileError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<ExprAst, CompileError> {
        let mut lhs = self.parse_and()?;
        while let Token::Pipe = self.peek_token()? {
            let op = BinaryOp::Or;
            self.next_token()?;
            let rhs = self.parse_and()?;
            lhs = ExprAst::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<ExprAst, CompileError> {
        let mut lhs = self.parse_compare()?;
        while let Token::Amp = self.peek_token()? {
            let op = BinaryOp::And;
            self.next_token()?;
            let rhs = self.parse_compare()?;
            lhs = ExprAst::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_compare(&mut self) -> Result<ExprAst, CompileError> {
        let mut lhs = self.parse_add_sub()?;
        loop {
            let op = match self.peek_token()? {
                Token::Lt => BinaryOp::Lt,
                Token::Le => BinaryOp::Le,
                Token::Gt => BinaryOp::Gt,
                Token::Ge => BinaryOp::Ge,
                Token::EqEq => BinaryOp::Eq,
                Token::NotEq => BinaryOp::Ne,
                _ => break,
            };
            self.next_token()?;
            let rhs = self.parse_add_sub()?;
            lhs = ExprAst::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_add_sub(&mut self) -> Result<ExprAst, CompileError> {
        let mut lhs = self.parse_mul_div()?;
        loop {
            let op = match self.peek_token()? {
                Token::Plus => BinaryOp::Add,
                Token::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.next_token()?;
            let rhs = self.parse_mul_div()?;
            lhs = ExprAst::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_mul_div(&mut self) -> Result<ExprAst, CompileError> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek_token()? {
                Token::Star => BinaryOp::Mul,
                Token::Slash => BinaryOp::Div,
                _ => break,
            };
            self.next_token()?;
            let rhs = self.parse_unary()?;
            lhs = ExprAst::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<ExprAst, CompileError> {
        match self.peek_token()? {
            Token::Plus => {
                self.next_token()?;
                let expr = self.parse_unary()?;
                Ok(ExprAst::Unary {
                    op: UnaryOp::Plus,
                    expr: Box::new(expr),
                })
            }
            Token::Minus => {
                self.next_token()?;
                let expr = self.parse_unary()?;
                Ok(ExprAst::Unary {
                    op: UnaryOp::Minus,
                    expr: Box::new(expr),
                })
            }
            Token::Tilde => {
                self.next_token()?;
                let expr = self.parse_unary()?;
                Ok(ExprAst::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(expr),
                })
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<ExprAst, CompileError> {
        match self.next_token()? {
            Token::Ident(name) => {
                if matches!(self.peek_token()?, Token::LParen) {
                    self.next_token()?; // consume '('
                    let (args, kwargs) = self.parse_arg_list()?;
                    self.expect_token(Token::RParen)?;
                    Ok(ExprAst::Call { name, args, kwargs })
                } else {
                    Ok(ExprAst::Identifier(name))
                }
            }
            Token::Number(value) => Ok(ExprAst::Number(value)),
            Token::LParen => {
                let expr = self.parse_expr()?;
                self.expect_token(Token::RParen)?;
                Ok(expr)
            }
            other => Err(CompileError::InvalidExpression {
                expr: self.source.to_string(),
                reason: format!("unexpected token: {other:?}"),
            }),
        }
    }

    fn parse_arg_list(
        &mut self,
    ) -> Result<(Vec<ExprAst>, BTreeMap<String, ExprAst>), CompileError> {
        let mut args = Vec::new();
        let mut kwargs = BTreeMap::new();
        loop {
            match self.peek_token()? {
                Token::RParen => break,
                Token::Eof => {
                    return Err(CompileError::InvalidExpression {
                        expr: self.source.to_string(),
                        reason: "unexpected EOF in argument list".to_string(),
                    });
                }
                _ => {}
            }

            let expr = self.parse_expr()?;
            if let ExprAst::Identifier(name) = &expr {
                if matches!(self.peek_token()?, Token::Equal) {
                    self.next_token()?;
                    let value = self.parse_expr()?;
                    kwargs.insert(name.clone(), value);
                } else {
                    args.push(expr);
                }
            } else {
                args.push(expr);
            }

            match self.peek_token()? {
                Token::Comma => {
                    self.next_token()?;
                }
                Token::RParen => break,
                other => {
                    return Err(CompileError::InvalidExpression {
                        expr: self.source.to_string(),
                        reason: format!("invalid token in argument list: {other:?}"),
                    });
                }
            }
        }
        Ok((args, kwargs))
    }

    fn expect_token(&mut self, expected: Token) -> Result<(), CompileError> {
        let got = self.next_token()?;
        if got == expected {
            Ok(())
        } else {
            Err(CompileError::InvalidExpression {
                expr: self.source.to_string(),
                reason: format!("expected {expected:?}, got {got:?}"),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_arithmetic_precedence() {
        let ast = parse_expression("a + b * c").expect("parse should succeed");
        match ast {
            ExprAst::Binary {
                op: BinaryOp::Add, ..
            } => {}
            other => panic!("unexpected ast: {other:?}"),
        }
    }

    #[test]
    fn parses_nested_calls_with_kwargs() {
        let ast = parse_expression("cs_rank(ts_mean(bar.close, window=5))")
            .expect("parse should succeed");
        match ast {
            ExprAst::Call { name, args, .. } => {
                assert_eq!(name, "cs_rank");
                assert_eq!(args.len(), 1);
            }
            other => panic!("unexpected ast: {other:?}"),
        }
    }

    #[test]
    fn parses_comparison_and_logical_precedence() {
        let ast = parse_expression("a < b & c >= d | ~e").expect("parse should succeed");
        match ast {
            ExprAst::Binary {
                op: BinaryOp::Or, ..
            } => {}
            other => panic!("unexpected ast: {other:?}"),
        }
    }
}
