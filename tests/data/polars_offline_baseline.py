#!/usr/bin/env python3
"""Offline Polars baseline for online parity tests.

Reads JSON payload from stdin and writes JSON rows:
{
  "rows": [{"output": "...", "instrument_slot": 101, "ts": 1, "value": 1.23 | null}, ...]
}
"""

from __future__ import annotations

import json
import math
import os
import inspect
import struct
import sys
from typing import Any

import polars as pl


_ROLLING_MIN_KWARG = (
    "min_samples"
    if "min_samples" in inspect.signature(pl.Expr.rolling_mean).parameters
    else "min_periods"
)


def _rolling_window_kwargs(window: int) -> dict[str, int]:
    return {"window_size": window, _ROLLING_MIN_KWARG: window}


def _finite_or_null_expr(column: str) -> pl.Expr:
    col = pl.col(column).cast(pl.Float64, strict=False)
    return pl.when(col.is_finite()).then(col).otherwise(None)


def _is_finite_expr(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.Float64, strict=False).is_finite()


def _rolling_mean_expr(column: str, window: int) -> pl.Expr:
    return pl.col(column).rolling_mean(**_rolling_window_kwargs(window))


def _rolling_std_expr(column: str, window: int) -> pl.Expr:
    return pl.col(column).rolling_std(**_rolling_window_kwargs(window), ddof=1)


def _rolling_corr_expr(lhs: str, rhs: str, window: int) -> pl.Expr:
    x = _finite_or_null_expr(lhs)
    y = _finite_or_null_expr(rhs)
    corr = pl.rolling_corr(pl.col(lhs), pl.col(rhs), **_rolling_window_kwargs(window))
    mean_x = x.rolling_mean(**_rolling_window_kwargs(window))
    mean_y = y.rolling_mean(**_rolling_window_kwargs(window))
    mean_xx = (x * x).rolling_mean(**_rolling_window_kwargs(window))
    mean_yy = (y * y).rolling_mean(**_rolling_window_kwargs(window))
    var_x = mean_xx - (mean_x * mean_x)
    var_y = mean_yy - (mean_y * mean_y)
    eps = 1.0e-12
    return (
        pl.when(var_x > eps)
        .then(
            pl.when(var_y > eps)
            .then(corr)
            .otherwise(None)
        )
        .otherwise(None)
    )


def _rolling_quantile_expr(column: str, window: int, q: float) -> pl.Expr:
    return _finite_or_null_expr(column).rolling_quantile(
        quantile=q,
        interpolation="linear",
        **_rolling_window_kwargs(window),
    )


def _rolling_min_expr(column: str, window: int) -> pl.Expr:
    return _finite_or_null_expr(column).rolling_min(**_rolling_window_kwargs(window))


def _rolling_max_expr(column: str, window: int) -> pl.Expr:
    return _finite_or_null_expr(column).rolling_max(**_rolling_window_kwargs(window))


def _operand_expr(operand: str) -> pl.Expr:
    try:
        literal = float(operand)
        return pl.lit(literal)
    except (TypeError, ValueError):
        return pl.col(operand)


def _elem_binary_expr(op: str, lhs: str, rhs: str) -> pl.Expr:
    lhs_expr = _operand_expr(lhs)
    rhs_expr = _operand_expr(rhs)
    if op == "elem_add":
        raw = lhs_expr + rhs_expr
    elif op == "elem_sub":
        raw = lhs_expr - rhs_expr
    elif op == "elem_mul":
        raw = lhs_expr * rhs_expr
    elif op == "elem_div":
        raw = lhs_expr / rhs_expr
    elif op == "elem_pow":
        # Keep dtype stable when one side is temporarily null-only.
        raw = lhs_expr.cast(pl.Float64).pow(rhs_expr.cast(pl.Float64))
    elif op == "elem_min":
        raw = pl.min_horizontal(lhs_expr, rhs_expr)
    elif op == "elem_max":
        raw = pl.max_horizontal(lhs_expr, rhs_expr)
    elif op == "elem_signed_power":
        sign_expr = pl.when(lhs_expr == 0.0).then(pl.lit(1.0)).otherwise(lhs_expr.sign())
        raw = sign_expr * lhs_expr.abs().pow(rhs_expr)
    elif op == "elem_lt":
        raw = (lhs_expr < rhs_expr).cast(pl.Float64)
    elif op == "elem_le":
        raw = (lhs_expr <= rhs_expr).cast(pl.Float64)
    elif op == "elem_gt":
        raw = (lhs_expr > rhs_expr).cast(pl.Float64)
    elif op == "elem_ge":
        raw = (lhs_expr >= rhs_expr).cast(pl.Float64)
    elif op == "elem_eq":
        raw = (lhs_expr == rhs_expr).cast(pl.Float64)
    elif op == "elem_ne":
        raw = (lhs_expr != rhs_expr).cast(pl.Float64)
    elif op == "elem_and":
        raw = ((lhs_expr != 0.0) & (rhs_expr != 0.0)).cast(pl.Float64)
    elif op == "elem_or":
        raw = ((lhs_expr != 0.0) | (rhs_expr != 0.0)).cast(pl.Float64)
    else:
        raise ValueError(f"unsupported elem op: {op}")
    raw = raw.cast(pl.Float64, strict=False)
    finite_guard = lhs_expr.is_not_null() & rhs_expr.is_not_null()
    if op in (
        "elem_lt",
        "elem_le",
        "elem_gt",
        "elem_ge",
        "elem_eq",
        "elem_ne",
        "elem_and",
        "elem_or",
    ):
        finite_guard = finite_guard & _is_finite_expr(lhs_expr) & _is_finite_expr(rhs_expr)
        return pl.when(finite_guard).then(raw).otherwise(None)
    return (
        pl.when(finite_guard & _is_finite_expr(raw))
        .then(raw)
        .otherwise(None)
    )


def _elem_unary_expr(op: str, field: str) -> pl.Expr:
    x = _operand_expr(field)
    if op == "elem_abs":
        raw = x.abs()
    elif op == "elem_exp":
        raw = x.exp()
    elif op == "elem_log":
        raw = x.log()
    elif op == "elem_sign":
        # Align with Rust f64::signum semantics used by online engine:
        # signum(+0.0) == 1.0.
        raw = pl.when(x == 0.0).then(pl.lit(1.0)).otherwise(x.sign())
    elif op == "elem_sqrt":
        raw = x.sqrt()
    elif op == "elem_to_int":
        raw = (x != 0.0).cast(pl.Float64)
    elif op == "elem_not":
        raw = (x == 0.0).cast(pl.Float64)
    else:
        raise ValueError(f"unsupported unary elem op: {op}")
    if op in ("elem_to_int", "elem_not"):
        return (
            pl.when(x.is_not_null() & _is_finite_expr(x))
            .then(raw)
            .otherwise(None)
        )
    return pl.when(x.is_not_null() & _is_finite_expr(raw)).then(raw).otherwise(None)


def _elem_clip_expr(field: str, lower: str, upper: str) -> pl.Expr:
    x = _operand_expr(field)
    lo = _operand_expr(lower)
    hi = _operand_expr(upper)
    raw = x.clip(lower_bound=lo, upper_bound=hi)
    return (
        pl.when(
            x.is_not_null()
            & lo.is_not_null()
            & hi.is_not_null()
            & (lo <= hi)
            & _is_finite_expr(raw)
        )
        .then(raw)
        .otherwise(None)
    )


def _elem_where_expr(cond: str, then_v: str, else_v: str) -> pl.Expr:
    c = _operand_expr(cond)
    t = _operand_expr(then_v)
    e = _operand_expr(else_v)
    return (
        pl.when(c.is_null())
        .then(None)
        .otherwise(pl.when(c != 0.0).then(t).otherwise(e))
    )


def _elem_fillna_expr(field: str, fill: str) -> pl.Expr:
    x = _operand_expr(field)
    f = _operand_expr(fill)
    return pl.when(x.is_not_null()).then(x).otherwise(f)


def _rolling_moments_expr(
    column: str, window: int
) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    x = _finite_or_null_expr(column)
    mean_x = x.rolling_mean(**_rolling_window_kwargs(window)).over("instrument_slot")
    mean_x2 = (x * x).rolling_mean(**_rolling_window_kwargs(window)).over(
        "instrument_slot"
    )
    mean_x3 = (x * x * x).rolling_mean(**_rolling_window_kwargs(window)).over(
        "instrument_slot"
    )
    mean_x4 = (x * x * x * x).rolling_mean(**_rolling_window_kwargs(window)).over(
        "instrument_slot"
    )
    return mean_x, mean_x2, mean_x3, mean_x4


def _ts_std_from_moments(sum_x: pl.Expr, sum_x2: pl.Expr, n: float) -> pl.Expr:
    m2 = sum_x2 - (sum_x * sum_x) / n
    variance = pl.when((m2 / (n - 1.0)) >= 0.0).then(m2 / (n - 1.0)).otherwise(0.0)
    return variance.sqrt()


def _ts_skew_expr(column: str, window: int) -> pl.Expr:
    if window <= 2:
        return pl.lit(None)
    n = float(window)
    eps = float.fromhex("0x1.0000000000000p-52")
    mean_x, mean_x2, mean_x3, _ = _rolling_moments_expr(column, window)
    sum_x = mean_x * n
    sum_x2 = mean_x2 * n
    sum_x3 = mean_x3 * n
    std = _ts_std_from_moments(sum_x, sum_x2, n)
    mean = sum_x / n
    mean_sq = mean * mean
    m3 = sum_x3 - 3.0 * mean * sum_x2 + 3.0 * mean_sq * sum_x - n * mean_sq * mean
    denom = (n - 1.0) * (n - 2.0) * std.pow(3)
    return (
        pl.when(_is_finite_expr(std) & (std > 0.0) & (denom.abs() > eps))
        .then((n * m3) / denom)
        .otherwise(None)
    )


def _ts_kurt_expr(column: str, window: int) -> pl.Expr:
    if window <= 3:
        return pl.lit(None)
    n = float(window)
    eps = float.fromhex("0x1.0000000000000p-52")
    mean_x, mean_x2, mean_x3, mean_x4 = _rolling_moments_expr(column, window)
    sum_x = mean_x * n
    sum_x2 = mean_x2 * n
    sum_x3 = mean_x3 * n
    sum_x4 = mean_x4 * n
    std = _ts_std_from_moments(sum_x, sum_x2, n)
    mean = sum_x / n
    mean_sq = mean * mean
    mean_cu = mean_sq * mean
    mean_qu = mean_sq * mean_sq
    m4 = (
        sum_x4
        - 4.0 * mean * sum_x3
        + 6.0 * mean_sq * sum_x2
        - 4.0 * mean_cu * sum_x
        + n * mean_qu
    )
    denom = (n - 1.0) * (n - 2.0) * (n - 3.0) * std.pow(4)
    term1 = (n * (n + 1.0) * m4) / denom
    term2 = 3.0 * ((n - 1.0) ** 2) / ((n - 2.0) * (n - 3.0))
    return (
        pl.when(_is_finite_expr(std) & (std > 0.0) & (denom.abs() > eps))
        .then(term1 - term2)
        .otherwise(None)
    )


def _cs_rank_expr(column: str) -> pl.Expr:
    masked = _finite_or_null_expr(column)
    rank = masked.rank(method="average").over("ts")
    count = masked.is_not_null().cast(pl.Int64).sum().over("ts")
    return (
        pl.when(masked.is_not_null())
        .then(pl.when(count > 1).then((rank - 1.0) / (count - 1.0)).otherwise(0.0))
        .otherwise(None)
    )


def _cs_rank_values_loop(df: pl.DataFrame, column: str) -> list[float | None]:
    values = df[column].to_list()
    out: list[float | None] = [None] * len(values)
    for idxs in _iter_ts_groups(df).values():
        finite = [(idx, float(values[idx])) for idx in idxs if _is_finite_number(values[idx])]
        if not finite:
            continue
        sorted_vals = sorted(v for _, v in finite)
        n = len(sorted_vals)
        for idx, value in finite:
            lower = sum(1 for v in sorted_vals if v < value)
            upper = sum(1 for v in sorted_vals if v <= value)
            avg_rank = (lower + 1 + upper) * 0.5
            out[idx] = (avg_rank - 1.0) / (n - 1.0) if n > 1 else 0.0
    return out


def _cs_center_expr(column: str) -> pl.Expr:
    masked = _finite_or_null_expr(column)
    mean = masked.mean().over("ts")
    return pl.when(masked.is_not_null()).then(masked - mean).otherwise(None)


def _cs_norm_expr(column: str) -> pl.Expr:
    masked = _finite_or_null_expr(column)
    lower = masked.min().over("ts")
    upper = masked.max().over("ts")
    range_expr = upper - lower
    return (
        pl.when(masked.is_not_null())
        .then(pl.when(range_expr > 0.0).then((masked - lower) / range_expr).otherwise(0.0))
        .otherwise(None)
    )


def _cs_fillna_expr(column: str) -> pl.Expr:
    masked = _finite_or_null_expr(column)
    mean = masked.mean().over("ts")
    return pl.when(masked.is_not_null()).then(masked).otherwise(mean)


def _cs_scale_expr(column: str) -> pl.Expr:
    masked = _finite_or_null_expr(column)
    denom = masked.abs().sum().over("ts")
    return (
        pl.when(masked.is_not_null() & denom.is_not_null() & (denom > 0.0))
        .then(masked / denom)
        .otherwise(None)
    )


def _cs_winsorize_expr(column: str, percentile: float) -> pl.Expr:
    masked = _finite_or_null_expr(column)
    lower = masked.quantile(percentile, interpolation="linear").over("ts")
    upper = masked.quantile(1.0 - percentile, interpolation="linear").over("ts")
    raw = masked.clip(lower_bound=lower, upper_bound=upper)
    return (
        pl.when(masked.is_not_null() & lower.is_not_null() & upper.is_not_null())
        .then(raw)
        .otherwise(None)
    )


def _cs_percentiles_expr(column: str, lower_pct: float, upper_pct: float) -> pl.Expr:
    masked = _finite_or_null_expr(column)
    lower = masked.quantile(lower_pct, interpolation="linear").over("ts")
    upper = masked.quantile(upper_pct, interpolation="linear").over("ts")
    fill = 0.0
    return (
        pl.when(masked.is_not_null())
        .then(
            pl.when((masked <= lower) | (masked >= upper))
            .then(masked)
            .otherwise(fill)
        )
        .otherwise(None)
    )


def _float_to_bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", float(value)))[0]


def _iter_ts_groups(df: pl.DataFrame) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for idx, ts in enumerate(df["ts"].to_list()):
        groups.setdefault(int(ts), []).append(idx)
    return groups


def _parse_standardize(raw: Any) -> bool:
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    token = str(raw).strip().lower()
    if token in ("1", "1.0", "true"):
        return True
    if token in ("0", "0.0", "false"):
        return False
    raise ValueError(f"invalid standardize token: {raw}")


def _parse_regressors(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str) and raw.strip():
        return [token.strip() for token in raw.split("|") if token.strip()]
    return []


def _parse_neutralize_layout(
    expr: dict[str, Any], op: str
) -> tuple[str, list[str], str | None, str | None, bool]:
    y_col = str(expr["field"])
    neutralize_cfg = expr.get("neutralize")
    if not isinstance(neutralize_cfg, dict):
        raise ValueError(
            f"{op} offline payload requires `neutralize` object "
            "(e.g. regressors/group/weights/standardize)"
        )

    reg_cols = _parse_regressors(neutralize_cfg.get("regressors"))
    group_raw = neutralize_cfg.get("group")
    group_col = group_raw.strip() if isinstance(group_raw, str) and group_raw.strip() else None
    weight_raw = neutralize_cfg.get("weights")
    weight_col = weight_raw.strip() if isinstance(weight_raw, str) and weight_raw.strip() else None
    standardize = _parse_standardize(neutralize_cfg.get("standardize"))

    if op == "cs_neutralize":
        if reg_cols:
            raise ValueError("cs_neutralize does not accept regressors")
        return y_col, reg_cols, group_col, weight_col, standardize

    if op == "cs_neutralize_ols":
        if len(reg_cols) != 1:
            raise ValueError("cs_neutralize_ols expects exactly one regressor")
        return y_col, reg_cols, group_col, weight_col, standardize

    if op == "cs_neutralize_ols_multi":
        if not (1 <= len(reg_cols) <= 3):
            raise ValueError(
                f"cs_neutralize_ols_multi expects 1~3 regressors, got {len(reg_cols)}"
            )
        return y_col, reg_cols, group_col, weight_col, standardize

    raise ValueError(f"unsupported neutralize op: {op}")


def _solve_linear_system(a: list[list[float]], b: list[float], eps: float) -> bool:
    dim = len(b)
    for pivot_col in range(dim):
        pivot_row = pivot_col
        pivot_abs = abs(a[pivot_col][pivot_col])
        for row in range(pivot_col + 1, dim):
            cand = abs(a[row][pivot_col])
            if cand > pivot_abs:
                pivot_abs = cand
                pivot_row = row
        if pivot_abs <= eps:
            return False
        if pivot_row != pivot_col:
            a[pivot_row], a[pivot_col] = a[pivot_col], a[pivot_row]
            b[pivot_row], b[pivot_col] = b[pivot_col], b[pivot_row]
        diag = a[pivot_col][pivot_col]
        for col in range(pivot_col, dim):
            a[pivot_col][col] /= diag
        b[pivot_col] /= diag
        for row in range(dim):
            if row == pivot_col:
                continue
            factor = a[row][pivot_col]
            if abs(factor) <= eps:
                continue
            for col in range(pivot_col, dim):
                a[row][col] -= factor * a[pivot_col][col]
            b[row] -= factor * b[pivot_col]
    return True


def _standardize_group_values(
    out: list[float | None], valid_indices: list[int], weights: list[Any] | None, eps: float
) -> None:
    if not valid_indices:
        return
    if weights is not None:
        sum_w = 0.0
        mean = 0.0
        for idx in valid_indices:
            raw_w = weights[idx]
            if not _is_finite_number(raw_w) or float(raw_w) <= 0.0:
                out[idx] = None
                continue
            cur = out[idx]
            if not _is_finite_number(cur):
                continue
            w = float(raw_w)
            sum_w += w
            mean += w * float(cur)
        if sum_w <= eps:
            for idx in valid_indices:
                out[idx] = None
            return
        mean /= sum_w
        var_num = 0.0
        for idx in valid_indices:
            raw_w = weights[idx]
            if not _is_finite_number(raw_w) or float(raw_w) <= 0.0:
                continue
            cur = out[idx]
            if not _is_finite_number(cur):
                continue
            centered = float(cur) - mean
            out[idx] = centered
            var_num += float(raw_w) * centered * centered
        if var_num <= eps:
            for idx in valid_indices:
                out[idx] = None
            return
        std = math.sqrt(var_num / sum_w)
        if not math.isfinite(std) or std <= eps:
            for idx in valid_indices:
                out[idx] = None
            return
        for idx in valid_indices:
            if _is_finite_number(out[idx]):
                out[idx] = float(out[idx]) / std
        return

    if len(valid_indices) < 2:
        for idx in valid_indices:
            out[idx] = None
        return
    mean = sum(float(out[idx]) for idx in valid_indices if _is_finite_number(out[idx])) / float(
        len(valid_indices)
    )
    var_num = 0.0
    for idx in valid_indices:
        centered = float(out[idx]) - mean
        out[idx] = centered
        var_num += centered * centered
    variance = var_num / float(len(valid_indices) - 1)
    if not math.isfinite(variance) or variance <= eps:
        for idx in valid_indices:
            out[idx] = None
        return
    std = math.sqrt(variance)
    for idx in valid_indices:
        out[idx] = float(out[idx]) / std


def _cs_neutralize_values(
    df: pl.DataFrame,
    y_col: str,
    reg_cols: list[str],
    group_col: str | None,
    weight_col: str | None,
    standardize: bool,
) -> list[float | None]:
    eps = 1e-12
    out: list[float | None] = [None] * df.height
    y_vals = df[y_col].to_list()
    reg_vals = [df[col].to_list() for col in reg_cols]
    group_vals = df[group_col].to_list() if group_col else None
    weight_vals = df[weight_col].to_list() if weight_col else None

    for ts_indices in _iter_ts_groups(df).values():
        grouped: dict[int, list[int]] = {}
        for idx in ts_indices:
            y = y_vals[idx]
            if not _is_finite_number(y):
                continue
            if group_vals is not None:
                g = group_vals[idx]
                if not _is_finite_number(g):
                    continue
                key = _float_to_bits(float(g))
            else:
                key = 0
            grouped.setdefault(key, []).append(idx)

        for indices in grouped.values():
            if not reg_cols:
                valid: list[int] = []
                sum_w = 0.0
                sum_y = 0.0
                for idx in indices:
                    y = float(y_vals[idx])
                    if weight_vals is not None:
                        w_raw = weight_vals[idx]
                        if not _is_finite_number(w_raw) or float(w_raw) <= 0.0:
                            continue
                        w = float(w_raw)
                    else:
                        w = 1.0
                    valid.append(idx)
                    sum_w += w
                    sum_y += y * w
                if not valid or sum_w <= eps:
                    continue
                mean = sum_y / sum_w
                for idx in valid:
                    out[idx] = float(y_vals[idx]) - mean
                if standardize:
                    _standardize_group_values(out, valid, weight_vals, eps)
                continue

            reg_count = len(reg_cols)
            rows: list[tuple[int, float, list[float]]] = []
            sum_w = 0.0
            sum_y = 0.0
            sum_x = [0.0] * reg_count
            sum_xy = [0.0] * reg_count
            sum_xx = [[0.0] * reg_count for _ in range(reg_count)]

            for idx in indices:
                y = y_vals[idx]
                if not _is_finite_number(y):
                    continue
                if weight_vals is not None:
                    w_raw = weight_vals[idx]
                    if not _is_finite_number(w_raw) or float(w_raw) <= 0.0:
                        continue
                    w = float(w_raw)
                else:
                    w = 1.0
                x_row: list[float] = []
                valid_row = True
                for reg in reg_vals:
                    x = reg[idx]
                    if not _is_finite_number(x):
                        valid_row = False
                        break
                    x_row.append(float(x))
                if not valid_row:
                    continue
                y_f = float(y)
                rows.append((idx, y_f, x_row))
                sum_w += w
                sum_y += w * y_f
                for i in range(reg_count):
                    xi = x_row[i]
                    sum_x[i] += w * xi
                    sum_xy[i] += w * xi * y_f
                    for j in range(i, reg_count):
                        sum_xx[i][j] += w * xi * x_row[j]

            if not rows or sum_w <= eps:
                continue
            for i in range(reg_count):
                for j in range(i):
                    sum_xx[i][j] = sum_xx[j][i]

            mean_y = sum_y / sum_w
            mean_x = [sx / sum_w for sx in sum_x]
            cov_xx = [[0.0] * reg_count for _ in range(reg_count)]
            cov_xy = [0.0] * reg_count
            for i in range(reg_count):
                cov_xy[i] = (sum_xy[i] / sum_w) - mean_x[i] * mean_y
                for j in range(reg_count):
                    cov_xx[i][j] = (sum_xx[i][j] / sum_w) - mean_x[i] * mean_x[j]

            beta = [0.0] * reg_count
            if reg_count == 1:
                var_x = cov_xx[0][0]
                if var_x > eps:
                    beta[0] = cov_xy[0] / var_x
            else:
                rhs = cov_xy[:]
                if _solve_linear_system(cov_xx, rhs, eps):
                    beta = rhs

            alpha = mean_y
            for i in range(reg_count):
                alpha -= beta[i] * mean_x[i]

            valid_indices: list[int] = []
            for idx, y_f, x_row in rows:
                fitted = alpha
                for i in range(reg_count):
                    fitted += beta[i] * x_row[i]
                residual = y_f - fitted
                if _is_finite_number(residual):
                    out[idx] = residual
                    valid_indices.append(idx)
            if standardize:
                _standardize_group_values(out, valid_indices, weight_vals, eps)

    return out


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _ts_rank_from_window(window_values: pl.Series) -> float | None:
    values = window_values.to_list()
    if not values or not all(_is_finite_number(v) for v in values):
        return None
    latest = float(values[-1])
    sorted_vals = sorted(float(v) for v in values)
    lower = sum(1 for v in sorted_vals if v < latest)
    upper = sum(1 for v in sorted_vals if v <= latest)
    avg_rank = (lower + 1 + upper) * 0.5
    return (avg_rank - 1.0) / (len(values) - 1.0) if len(values) > 1 else 0.0


def _ts_rank_expr(column: str, window: int) -> pl.Expr:
    return _finite_or_null_expr(column).rolling_map(
        _ts_rank_from_window,
        **_rolling_window_kwargs(window),
    )


def _ts_argext_from_window(window_values: pl.Series, kind: str) -> float | None:
    values = window_values.to_list()
    if not values or not all(_is_finite_number(v) for v in values):
        return None
    latest_first = [float(values[-1 - lag]) for lag in range(len(values))]
    best_idx = 0
    best_val = latest_first[0]
    for lag, current in enumerate(latest_first[1:], start=1):
        if kind == "max":
            if current > best_val:
                best_val = current
                best_idx = lag
        elif kind == "min":
            if current < best_val:
                best_val = current
                best_idx = lag
        else:
            raise ValueError(f"unsupported argext kind: {kind}")
    return float(best_idx)


def _ts_argmax_from_window(window_values: pl.Series) -> float | None:
    return _ts_argext_from_window(window_values, "max")


def _ts_argmin_from_window(window_values: pl.Series) -> float | None:
    return _ts_argext_from_window(window_values, "min")


def _ts_argmax_expr(column: str, window: int) -> pl.Expr:
    return _finite_or_null_expr(column).rolling_map(
        _ts_argmax_from_window,
        **_rolling_window_kwargs(window),
    )


def _ts_argmin_expr(column: str, window: int) -> pl.Expr:
    return _finite_or_null_expr(column).rolling_map(
        _ts_argmin_from_window,
        **_rolling_window_kwargs(window),
    )


def _iter_instrument_groups(df: pl.DataFrame) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for idx, instrument_slot in enumerate(df["instrument_slot"].to_list()):
        groups.setdefault(int(instrument_slot), []).append(idx)
    return groups


def _ts_rank_values_loop(df: pl.DataFrame, column: str, window: int) -> list[float | None]:
    values = df[column].to_list()
    out: list[float | None] = [None] * df.height
    for idxs in _iter_instrument_groups(df).values():
        for pos, row_idx in enumerate(idxs):
            if pos + 1 < window:
                continue
            sample = [values[idxs[pos - lag]] for lag in range(window)]
            if not all(_is_finite_number(v) for v in sample):
                continue
            latest = float(sample[0])
            sorted_vals = sorted(float(v) for v in sample)
            lower = sum(1 for v in sorted_vals if v < latest)
            upper = sum(1 for v in sorted_vals if v <= latest)
            avg_rank = (lower + 1 + upper) * 0.5
            out[row_idx] = (avg_rank - 1.0) / (window - 1.0) if window > 1 else 0.0
    return out


def _ts_argext_values_loop(
    df: pl.DataFrame, column: str, window: int, kind: str
) -> list[float | None]:
    values = df[column].to_list()
    out: list[float | None] = [None] * df.height
    for idxs in _iter_instrument_groups(df).values():
        for pos, row_idx in enumerate(idxs):
            if pos + 1 < window:
                continue
            sample = [values[idxs[pos - lag]] for lag in range(window)]
            if not all(_is_finite_number(v) for v in sample):
                continue
            best_idx = 0
            best_val = float(sample[0])
            for lag in range(1, window):
                cur = float(sample[lag])
                if kind == "max":
                    if cur > best_val:
                        best_val = cur
                        best_idx = lag
                elif kind == "min":
                    if cur < best_val:
                        best_val = cur
                        best_idx = lag
                else:
                    raise ValueError(f"unsupported argext kind: {kind}")
            out[row_idx] = float(best_idx)
    return out


def _window_std(sum_x: float, sum_x2: float, n: float) -> float | None:
    if n <= 1.0:
        return None
    m2 = sum_x2 - (sum_x * sum_x) / n
    variance = max(m2 / (n - 1.0), 0.0)
    std = math.sqrt(variance)
    return std if math.isfinite(std) and std > 0.0 else None


def _naive_power_sums(sample: list[float]) -> tuple[float, float, float, float]:
    sum_x = 0.0
    sum_x2 = 0.0
    sum_x3 = 0.0
    sum_x4 = 0.0
    for v in sample:
        sum_x += v
        sq = v * v
        sum_x2 += sq
        cu = sq * v
        sum_x3 += cu
        sum_x4 += sq * sq
    return sum_x, sum_x2, sum_x3, sum_x4


def _ts_skew_from_sample(sample: list[float]) -> float | None:
    n = float(len(sample))
    if n <= 2.0:
        return None
    sum_x, sum_x2, sum_x3, _ = _naive_power_sums(sample)
    std = _window_std(sum_x, sum_x2, n)
    if std is None:
        return None
    mean = sum_x / n
    mean_sq = mean * mean
    m3 = sum_x3 - 3.0 * mean * sum_x2 + 3.0 * mean_sq * sum_x - n * mean_sq * mean
    denom = (n - 1.0) * (n - 2.0) * (std**3)
    if abs(denom) <= float.fromhex("0x1.0000000000000p-52"):
        return None
    out = (n * m3) / denom
    return out if math.isfinite(out) else None


def _ts_kurt_from_sample(sample: list[float]) -> float | None:
    n = float(len(sample))
    if n <= 3.0:
        return None
    sum_x, sum_x2, sum_x3, sum_x4 = _naive_power_sums(sample)
    std = _window_std(sum_x, sum_x2, n)
    if std is None:
        return None
    mean = sum_x / n
    mean_sq = mean * mean
    mean_cu = mean_sq * mean
    mean_qu = mean_sq * mean_sq
    m4 = (
        sum_x4
        - 4.0 * mean * sum_x3
        + 6.0 * mean_sq * sum_x2
        - 4.0 * mean_cu * sum_x
        + n * mean_qu
    )
    denom = (n - 1.0) * (n - 2.0) * (n - 3.0) * (std**4)
    if abs(denom) <= float.fromhex("0x1.0000000000000p-52"):
        return None
    term1 = (n * (n + 1.0) * m4) / denom
    term2 = 3.0 * ((n - 1.0) ** 2) / ((n - 2.0) * (n - 3.0))
    out = term1 - term2
    return out if math.isfinite(out) else None


def _ts_higher_moment_values_loop(
    df: pl.DataFrame, column: str, window: int, kind: str
) -> list[float | None]:
    values = df[column].to_list()
    out: list[float | None] = [None] * df.height
    for idxs in _iter_instrument_groups(df).values():
        for pos, row_idx in enumerate(idxs):
            if pos + 1 < window:
                continue
            sample_raw = [values[idxs[pos - lag]] for lag in range(window)]
            if not all(_is_finite_number(v) for v in sample_raw):
                continue
            sample = [float(v) for v in sample_raw]
            if kind == "skew":
                out[row_idx] = _ts_skew_from_sample(sample)
            elif kind == "kurt":
                out[row_idx] = _ts_kurt_from_sample(sample)
            else:
                raise ValueError(f"unsupported higher moment kind: {kind}")
    return out


def _ts_higher_moment_values_incremental(
    df: pl.DataFrame, column: str, window: int, kind: str
) -> list[float | None]:
    values = df[column].to_list()
    out: list[float | None] = [None] * df.height
    eps = float.fromhex("0x1.0000000000000p-52")

    for idxs in _iter_instrument_groups(df).values():
        ring = [math.nan] * window
        head = 0
        length = 0
        nan_count = 0
        sum_x = 0.0
        sum_x2 = 0.0
        sum_x3 = 0.0
        sum_x4 = 0.0

        for row_idx in idxs:
            raw = values[row_idx]
            value = float(raw) if _is_finite_number(raw) else math.nan

            if length == window:
                evicted = ring[head]
                if math.isfinite(evicted):
                    sum_x -= evicted
                    sq = evicted * evicted
                    sum_x2 -= sq
                    cu = sq * evicted
                    sum_x3 -= cu
                    sum_x4 -= sq * sq
                else:
                    nan_count -= 1
            else:
                length += 1

            ring[head] = value
            if math.isfinite(value):
                sum_x += value
                sq = value * value
                sum_x2 += sq
                cu = sq * value
                sum_x3 += cu
                sum_x4 += sq * sq
            else:
                nan_count += 1
            head = (head + 1) % window

            if length < window or nan_count > 0:
                continue

            n = float(window)
            mean = sum_x / n
            m2 = sum_x2 - (sum_x * sum_x) / n
            if n <= 1.0:
                continue
            variance = max(m2 / (n - 1.0), 0.0)
            std = math.sqrt(variance)
            if not math.isfinite(std) or std <= 0.0:
                continue

            if kind == "skew":
                if n <= 2.0:
                    continue
                mean_sq = mean * mean
                m3 = sum_x3 - 3.0 * mean * sum_x2 + 3.0 * mean_sq * sum_x - n * mean_sq * mean
                denom = (n - 1.0) * (n - 2.0) * (std * std * std)
                if abs(denom) <= eps:
                    continue
                value_out = (n * m3) / denom
            elif kind == "kurt":
                if n <= 3.0:
                    continue
                mean_sq = mean * mean
                mean_cu = mean_sq * mean
                mean_qu = mean_sq * mean_sq
                m4 = (
                    sum_x4
                    - 4.0 * mean * sum_x3
                    + 6.0 * mean_sq * sum_x2
                    - 4.0 * mean_cu * sum_x
                    + n * mean_qu
                )
                denom = (n - 1.0) * (n - 2.0) * (n - 3.0) * (std * std * std * std)
                if abs(denom) <= eps:
                    continue
                term1 = (n * (n + 1.0) * m4) / denom
                term2 = 3.0 * ((n - 1.0) ** 2) / ((n - 2.0) * (n - 3.0))
                value_out = term1 - term2
            else:
                raise ValueError(f"unsupported higher moment kind: {kind}")

            out[row_idx] = value_out if math.isfinite(value_out) else None

    return out


def _ewm_alpha(window: int) -> float:
    return 2.0 / (float(window) + 1.0)


def _ewm_mean_var_values_loop(
    df: pl.DataFrame, column: str, window: int
) -> tuple[list[float | None], list[float | None]]:
    values = df[column].to_list()
    out_mean: list[float | None] = [None] * df.height
    out_var: list[float | None] = [None] * df.height
    alpha = _ewm_alpha(window)
    decay = 1.0 - alpha
    for idxs in _iter_instrument_groups(df).values():
        for pos, row_idx in enumerate(idxs):
            if pos + 1 < window:
                continue
            sample_raw = [values[idxs[pos - lag]] for lag in range(window)]
            if not all(_is_finite_number(v) for v in sample_raw):
                continue
            weight = 1.0
            weight_sum = 0.0
            wx = 0.0
            wx2 = 0.0
            for raw in sample_raw:
                x = float(raw)
                weight_sum += weight
                wx += weight * x
                wx2 += weight * x * x
                weight *= decay
            if weight_sum <= 0.0 or not math.isfinite(weight_sum):
                continue
            mean = wx / weight_sum
            var = max(wx2 / weight_sum - mean * mean, 0.0)
            out_mean[row_idx] = mean if math.isfinite(mean) else None
            out_var[row_idx] = var if math.isfinite(var) else None
    return out_mean, out_var


def _ewm_cov_values_loop(
    df: pl.DataFrame, lhs: str, rhs: str, window: int
) -> list[float | None]:
    xs = df[lhs].to_list()
    ys = df[rhs].to_list()
    out: list[float | None] = [None] * df.height
    alpha = _ewm_alpha(window)
    decay = 1.0 - alpha
    for idxs in _iter_instrument_groups(df).values():
        for pos, row_idx in enumerate(idxs):
            if pos + 1 < window:
                continue
            sx_raw = [xs[idxs[pos - lag]] for lag in range(window)]
            sy_raw = [ys[idxs[pos - lag]] for lag in range(window)]
            if not all(_is_finite_number(v) for v in sx_raw):
                continue
            if not all(_is_finite_number(v) for v in sy_raw):
                continue
            weight = 1.0
            weight_sum = 0.0
            wx = 0.0
            wy = 0.0
            wxy = 0.0
            for raw_x, raw_y in zip(sx_raw, sy_raw):
                x = float(raw_x)
                y = float(raw_y)
                weight_sum += weight
                wx += weight * x
                wy += weight * y
                wxy += weight * x * y
                weight *= decay
            if weight_sum <= 0.0 or not math.isfinite(weight_sum):
                continue
            mean_x = wx / weight_sum
            mean_y = wy / weight_sum
            cov = wxy / weight_sum - mean_x * mean_y
            out[row_idx] = cov if math.isfinite(cov) else None
    return out


def _ts_decay_linear_expr(column: str, window: int) -> pl.Expr:
    if window <= 0:
        return pl.lit(None)

    denom = float(window * (window + 1)) / 2.0
    if not math.isfinite(denom) or denom <= 0.0:
        return pl.lit(None)

    x = _finite_or_null_expr(column)
    shifted = [x.shift(lag).over("instrument_slot") for lag in range(window)]
    weighted_terms = [shifted[lag] * float(window - lag) for lag in range(window)]

    weighted_sum = weighted_terms[0]
    for term in weighted_terms[1:]:
        weighted_sum = weighted_sum + term

    return (
        pl.when(
            pl.all_horizontal([s.is_not_null() for s in shifted]) & _is_finite_expr(weighted_sum)
        )
        .then(weighted_sum / denom)
        .otherwise(None)
    )


def _ts_product_expr(column: str, window: int) -> pl.Expr:
    if window <= 0:
        return pl.lit(None)
    x = _finite_or_null_expr(column)
    shifted = [x.shift(lag).over("instrument_slot") for lag in range(window)]
    product = shifted[0]
    for s in shifted[1:]:
        product = product * s
    return (
        pl.when(pl.all_horizontal([s.is_not_null() for s in shifted]) & _is_finite_expr(product))
        .then(product)
        .otherwise(None)
    )


def _ts_mad_expr(column: str, window: int) -> pl.Expr:
    if window <= 0:
        return pl.lit(None)
    x = _finite_or_null_expr(column)
    shifted = [x.shift(lag).over("instrument_slot") for lag in range(window)]
    sum_expr = shifted[0]
    for s in shifted[1:]:
        sum_expr = sum_expr + s
    mean_expr = sum_expr / float(window)
    abs_sum = (shifted[0] - mean_expr).abs()
    for s in shifted[1:]:
        abs_sum = abs_sum + (s - mean_expr).abs()
    mad = abs_sum / float(window)
    return (
        pl.when(pl.all_horizontal([s.is_not_null() for s in shifted]) & _is_finite_expr(mad))
        .then(mad)
        .otherwise(None)
    )


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item: dict[str, Any] = {
            "ts": int(row["ts"]),
            "instrument_slot": int(row["instrument_slot"]),
        }
        for key, value in row.get("fields", {}).items():
            item[key] = float("nan") if value is None else float(value)
        out.append(item)
    return out


def _evaluate(df: pl.DataFrame, expr: dict[str, Any]) -> pl.DataFrame:
    output = expr["output"]
    op = expr["op"]

    if op in (
        "elem_add",
        "elem_sub",
        "elem_mul",
        "elem_div",
        "elem_pow",
        "elem_min",
        "elem_max",
        "elem_signed_power",
        "elem_lt",
        "elem_le",
        "elem_gt",
        "elem_ge",
        "elem_eq",
        "elem_ne",
        "elem_and",
        "elem_or",
    ):
        lhs = expr["lhs"]
        rhs = expr["rhs"]
        value_expr = _elem_binary_expr(op, lhs, rhs)
    elif op in (
        "elem_abs",
        "elem_exp",
        "elem_log",
        "elem_sign",
        "elem_sqrt",
        "elem_to_int",
        "elem_not",
    ):
        field = expr["field"]
        value_expr = _elem_unary_expr(op, field)
    elif op == "elem_clip":
        field = expr["field"]
        lower = expr["lhs"]
        upper = expr["rhs"]
        value_expr = _elem_clip_expr(field, lower, upper)
    elif op == "elem_where":
        cond = expr["field"]
        then_v = expr["lhs"]
        else_v = expr["rhs"]
        value_expr = _elem_where_expr(cond, then_v, else_v)
    elif op == "elem_fillna":
        field = expr["field"]
        fill = expr["lhs"]
        value_expr = _elem_fillna_expr(field, fill)
    elif op == "ts_mean":
        field = expr["field"]
        window = int(expr["window"])
        value_expr = _rolling_mean_expr(field, window).over("instrument_slot")
    elif op == "ts_sum":
        field = expr["field"]
        window = int(expr["window"])
        value_expr = (
            _finite_or_null_expr(field)
            .rolling_sum(**_rolling_window_kwargs(window))
            .over("instrument_slot")
        )
    elif op == "ts_product":
        field = expr["field"]
        window = int(expr["window"])
        value_expr = _ts_product_expr(field, window)
    elif op == "ts_min":
        field = expr["field"]
        window = int(expr["window"])
        value_expr = _rolling_min_expr(field, window).over("instrument_slot")
    elif op == "ts_max":
        field = expr["field"]
        window = int(expr["window"])
        value_expr = _rolling_max_expr(field, window).over("instrument_slot")
    elif op == "ts_mad":
        field = expr["field"]
        window = int(expr["window"])
        value_expr = _ts_mad_expr(field, window)
    elif op == "ts_std":
        field = expr["field"]
        window = int(expr["window"])
        value_expr = _rolling_std_expr(field, window).over("instrument_slot")
    elif op == "ts_var":
        field = expr["field"]
        window = int(expr["window"])
        value_expr = (
            _finite_or_null_expr(field)
            .rolling_var(**_rolling_window_kwargs(window), ddof=1)
            .over("instrument_slot")
        )
    elif op == "ts_zscore":
        field = expr["field"]
        window = int(expr["window"])
        x = _finite_or_null_expr(field)
        mean = x.rolling_mean(**_rolling_window_kwargs(window)).over("instrument_slot")
        std = x.rolling_std(**_rolling_window_kwargs(window), ddof=1).over(
            "instrument_slot"
        )
        value_expr = (
            pl.when(x.is_not_null() & mean.is_not_null() & std.is_not_null())
            .then(
                pl.when(std > 0.0)
                .then((x - mean) / std)
                .otherwise(0.0)
            )
            .otherwise(None)
        )
    elif op == "ts_delta":
        field = expr["field"]
        lag = int(expr["lag"])
        value_expr = (pl.col(field) - pl.col(field).shift(lag)).over("instrument_slot")
    elif op == "ts_lag":
        field = expr["field"]
        lag = int(expr["lag"])
        value_expr = pl.col(field).shift(lag).over("instrument_slot")
    elif op == "ts_cov":
        lhs = expr["lhs"]
        rhs = expr["rhs"]
        window = int(expr["window"])
        if window <= 1:
            value_expr = pl.lit(None)
            return df.with_columns(value_expr.alias(output))
        x = _finite_or_null_expr(lhs)
        y = _finite_or_null_expr(rhs)
        mean_x = x.rolling_mean(**_rolling_window_kwargs(window)).over(
            "instrument_slot"
        )
        mean_y = y.rolling_mean(**_rolling_window_kwargs(window)).over(
            "instrument_slot"
        )
        mean_xy = (x * y).rolling_mean(**_rolling_window_kwargs(window)).over(
            "instrument_slot"
        )
        cov_num = mean_xy - (mean_x * mean_y)
        value_expr = cov_num * (float(window) / float(window - 1))
    elif op == "ts_beta":
        lhs = expr["lhs"]
        rhs = expr["rhs"]
        window = int(expr["window"])
        x = _finite_or_null_expr(lhs)
        y = _finite_or_null_expr(rhs)
        mean_x = x.rolling_mean(**_rolling_window_kwargs(window)).over(
            "instrument_slot"
        )
        mean_y = y.rolling_mean(**_rolling_window_kwargs(window)).over(
            "instrument_slot"
        )
        mean_xy = (x * y).rolling_mean(**_rolling_window_kwargs(window)).over(
            "instrument_slot"
        )
        mean_yy = (y * y).rolling_mean(**_rolling_window_kwargs(window)).over(
            "instrument_slot"
        )
        cov_num = mean_xy - (mean_x * mean_y)
        var_y_num = mean_yy - (mean_y * mean_y)
        value_expr = (
            pl.when(var_y_num > 0.0)
            .then(cov_num / var_y_num)
            .otherwise(None)
        )
    elif op == "ts_ewm_mean":
        field = expr["field"]
        window = int(expr["window"])
        means, _ = _ewm_mean_var_values_loop(df, field, window)
        return df.with_columns(pl.Series(output, means))
    elif op == "ts_ewm_var":
        field = expr["field"]
        window = int(expr["window"])
        _, vars_ = _ewm_mean_var_values_loop(df, field, window)
        return df.with_columns(pl.Series(output, vars_))
    elif op == "ts_ewm_cov":
        lhs = expr["lhs"]
        rhs = expr["rhs"]
        window = int(expr["window"])
        return df.with_columns(pl.Series(output, _ewm_cov_values_loop(df, lhs, rhs, window)))
    elif op == "ts_decay_linear":
        field = expr["field"]
        window = int(expr["window"])
        value_expr = _ts_decay_linear_expr(field, window)
    elif op == "ts_corr":
        lhs = expr["lhs"]
        rhs = expr["rhs"]
        window = int(expr["window"])
        value_expr = _rolling_corr_expr(lhs, rhs, window).over("instrument_slot")
    elif op == "cs_rank":
        field = expr["field"]
        value_expr = _cs_rank_expr(field)
    elif op == "cs_center":
        field = expr["field"]
        value_expr = _cs_center_expr(field)
    elif op == "cs_neutralize":
        y_col, reg_cols, group_col, weight_col, standardize = _parse_neutralize_layout(expr, op)
        values = _cs_neutralize_values(df, y_col, reg_cols, group_col, weight_col, standardize)
        return df.with_columns(pl.Series(output, values))
    elif op == "cs_norm":
        field = expr["field"]
        value_expr = _cs_norm_expr(field)
    elif op == "cs_fillna":
        field = expr["field"]
        value_expr = _cs_fillna_expr(field)
    elif op == "cs_scale":
        field = expr["field"]
        value_expr = _cs_scale_expr(field)
    elif op == "cs_winsorize":
        field = expr["field"]
        percentile = float(expr["lhs"])
        value_expr = _cs_winsorize_expr(field, percentile)
    elif op == "cs_percentiles":
        field = expr["field"]
        lower_pct = float(expr["lhs"])
        upper_pct = float(expr["rhs"])
        value_expr = _cs_percentiles_expr(field, lower_pct, upper_pct)
    elif op == "cs_neutralize_ols":
        y_col, reg_cols, group_col, weight_col, standardize = _parse_neutralize_layout(expr, op)
        values = _cs_neutralize_values(df, y_col, reg_cols, group_col, weight_col, standardize)
        return df.with_columns(pl.Series(output, values))
    elif op == "cs_neutralize_ols_multi":
        y_col, reg_cols, group_col, weight_col, standardize = _parse_neutralize_layout(expr, op)
        values = _cs_neutralize_values(df, y_col, reg_cols, group_col, weight_col, standardize)
        return df.with_columns(pl.Series(output, values))
    elif op == "ts_rank":
        field = expr["field"]
        window = int(expr["window"])
        mode = os.environ.get("FACTOR_ENGINE_TS_RANK_MODE", "expr")
        if mode == "loop":
            return df.with_columns(pl.Series(output, _ts_rank_values_loop(df, field, window)))
        if mode != "expr":
            raise ValueError(f"unsupported FACTOR_ENGINE_TS_RANK_MODE: {mode}")
        value_expr = _ts_rank_expr(field, window).over("instrument_slot")
    elif op == "ts_argmax":
        field = expr["field"]
        window = int(expr["window"])
        mode = os.environ.get("FACTOR_ENGINE_TS_ARGEXT_MODE", "expr")
        if mode == "loop":
            return df.with_columns(
                pl.Series(output, _ts_argext_values_loop(df, field, window, "max"))
            )
        if mode != "expr":
            raise ValueError(f"unsupported FACTOR_ENGINE_TS_ARGEXT_MODE: {mode}")
        value_expr = _ts_argmax_expr(field, window).over("instrument_slot")
    elif op == "ts_argmin":
        field = expr["field"]
        window = int(expr["window"])
        mode = os.environ.get("FACTOR_ENGINE_TS_ARGEXT_MODE", "expr")
        if mode == "loop":
            return df.with_columns(
                pl.Series(output, _ts_argext_values_loop(df, field, window, "min"))
            )
        if mode != "expr":
            raise ValueError(f"unsupported FACTOR_ENGINE_TS_ARGEXT_MODE: {mode}")
        value_expr = _ts_argmin_expr(field, window).over("instrument_slot")
    elif op == "ts_quantile":
        field = expr["field"]
        window = int(expr["window"])
        q_raw = expr.get("q", None)
        if q_raw is None:
            q_raw = expr.get("rhs", None)
        if q_raw is None:
            raise ValueError("ts_quantile requires `q` (or `rhs`) parameter")
        q = float(q_raw)
        if not math.isfinite(q) or q < 0.0 or q > 1.0:
            raise ValueError(f"invalid q for ts_quantile: {q_raw}")
        value_expr = _rolling_quantile_expr(field, window, q).over("instrument_slot")
    elif op == "ts_skew":
        field = expr["field"]
        window = int(expr["window"])
        mode = os.environ.get("FACTOR_ENGINE_TS_HIGHER_MOMENTS_MODE", "expr")
        if mode == "incremental":
            return df.with_columns(
                pl.Series(
                    output,
                    _ts_higher_moment_values_incremental(df, field, window, "skew"),
                )
            )
        if mode == "loop":
            return df.with_columns(
                pl.Series(output, _ts_higher_moment_values_loop(df, field, window, "skew"))
            )
        if mode != "expr":
            raise ValueError(f"unsupported FACTOR_ENGINE_TS_HIGHER_MOMENTS_MODE: {mode}")
        value_expr = _ts_skew_expr(field, window)
    elif op == "ts_kurt":
        field = expr["field"]
        window = int(expr["window"])
        mode = os.environ.get("FACTOR_ENGINE_TS_HIGHER_MOMENTS_MODE", "expr")
        if mode == "incremental":
            return df.with_columns(
                pl.Series(
                    output,
                    _ts_higher_moment_values_incremental(df, field, window, "kurt"),
                )
            )
        if mode == "loop":
            return df.with_columns(
                pl.Series(output, _ts_higher_moment_values_loop(df, field, window, "kurt"))
            )
        if mode != "expr":
            raise ValueError(f"unsupported FACTOR_ENGINE_TS_HIGHER_MOMENTS_MODE: {mode}")
        value_expr = _ts_kurt_expr(field, window)
    elif op == "ts_linear_regression":
        lhs = expr["lhs"]
        rhs = expr["rhs"]
        window = int(expr["window"])
        x = _finite_or_null_expr(lhs)
        y = _finite_or_null_expr(rhs)
        mean_x = x.rolling_mean(**_rolling_window_kwargs(window)).over(
            "instrument_slot"
        )
        mean_y = y.rolling_mean(**_rolling_window_kwargs(window)).over(
            "instrument_slot"
        )
        mean_xy = (x * y).rolling_mean(**_rolling_window_kwargs(window)).over(
            "instrument_slot"
        )
        mean_xx = (x * x).rolling_mean(**_rolling_window_kwargs(window)).over(
            "instrument_slot"
        )
        cov_num = mean_xy - (mean_x * mean_y)
        var_x_num = mean_xx - (mean_x * mean_x)
        value_expr = (
            pl.when(var_x_num > 0.0)
            .then(cov_num / var_x_num)
            .otherwise(None)
        )
    elif op == "cs_zscore":
        field = expr["field"]
        masked_col = f"__{output}_masked"
        mean_col = f"__{output}_mean"
        std_col = f"__{output}_std"
        df = df.with_columns(_finite_or_null_expr(field).alias(masked_col))
        df = df.with_columns(pl.col(masked_col).mean().over("ts").alias(mean_col))
        df = df.with_columns(
            (
                ((pl.col(masked_col) - pl.col(mean_col)) * (pl.col(masked_col) - pl.col(mean_col)))
                .mean()
                .over("ts")
                .sqrt()
            ).alias(std_col)
        )
        df = df.with_columns(
            (
                pl.when(pl.col(masked_col).is_not_null())
                .then(
                    pl.when(pl.col(std_col) > 0.0)
                    .then((pl.col(masked_col) - pl.col(mean_col)) / pl.col(std_col))
                    .otherwise(0.0)
                )
                .otherwise(None)
            ).alias(output)
        )
        return df.drop([masked_col, mean_col, std_col])
    else:
        raise ValueError(f"unsupported op: {op}")

    return df.with_columns(value_expr.alias(output))


def _normalize_value(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def main() -> None:
    payload = json.load(sys.stdin)
    rows = _normalize_rows(payload["rows"])
    expressions = payload["expressions"]

    df = pl.DataFrame(rows).sort(["instrument_slot", "ts"])
    for expr in expressions:
        df = _evaluate(df, expr)

    output_rows = []
    for expr in expressions:
        output = expr["output"]
        for item in df.select(["instrument_slot", "ts", output]).iter_rows(named=True):
            output_rows.append(
                {
                    "output": output,
                    "instrument_slot": int(item["instrument_slot"]),
                    "ts": int(item["ts"]),
                    "value": _normalize_value(item[output]),
                }
            )

    json.dump({"rows": output_rows}, sys.stdout)


if __name__ == "__main__":
    main()
