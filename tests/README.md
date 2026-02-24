# Test Notes

`tests/data/polars_offline_baseline.py` is the offline reference for online parity tests.

Default policy is to prefer Polars expression paths for reproducibility and readability.
Fallback modes remain available for debugging/performance comparisons via env vars:

- `FACTOR_ENGINE_TS_RANK_MODE=loop|expr` (default: `expr`)
- `FACTOR_ENGINE_TS_ARGEXT_MODE=loop|expr` (default: `expr`)
- `FACTOR_ENGINE_TS_HIGHER_MOMENTS_MODE=incremental|loop|expr` (default: `expr`)

Run a focused suite first, then widen scope:

```bash
cargo test -p factor_engine --test offline_online_alpha101_batch_j_parity
cargo test --workspace --features offline-all
```
