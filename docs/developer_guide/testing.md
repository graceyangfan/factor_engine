# Testing Guide

This project follows a strict testing discipline: tests are executable specs, compact, and deterministic.

## Test layers

- Unit tests: `src/tests.rs`
- Integration/parity tests: `tests/*.rs`
- Offline baseline: `tests/data/polars_offline_baseline.py`

## Style rules

- Name tests by behavior under test, not assertion details.
- Arrange all setup first, run act step once, then group assertions.
- Prefer table-style/parametric patterns over copy-paste tests.
- In parity tests, prefer Polars `Expr` paths; loop paths are fallback only.
- Keep random data deterministic (fixed formula/seed).

## Running tests

```bash
cargo test -p factor_engine --lib
cargo test -p factor_engine --test offline_online_parity
cargo test --workspace --features offline-all
```

## Offline baseline modes

- `FACTOR_ENGINE_TS_RANK_MODE=expr|loop` (default `expr`)
- `FACTOR_ENGINE_TS_ARGEXT_MODE=expr|loop` (default `expr`)
- `FACTOR_ENGINE_TS_HIGHER_MOMENTS_MODE=expr|incremental|loop` (default `expr`)
