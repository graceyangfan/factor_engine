# factor_engine

`factor_engine` is a low-latency factor expression engine with an offline/online parity workflow.

It uses a two-stage architecture:

- `compile`: expression DSL -> canonical logical DAG (CSE, identity fold, const fold)
- `bind/runtime`: logical DAG -> slot-based physical plan -> incremental online execution

## Features

- Expression compiler with deterministic canonicalization
- Streaming runtime for low-latency updates
- Registry-driven operator extension model
- Polars offline baseline for parity verification
- Alpha101 parity coverage in integration tests

## Project Layout

- `src/compile.rs`: parser lowering and logical plan construction
- `src/plan.rs`: logical/physical plan data model
- `src/runtime.rs`: execution engine and scheduling
- `src/state.rs`: ring buffers, scratch pools, shared rolling state
- `src/ops/`: operator layer
  - `spec.rs`: `OpCode`, domain, execution capability
  - `catalog.rs`: registry and compile/runtime compatibility checks
  - `elem.rs`, `ts.rs`, `cs.rs`: kernel implementations
- `tests/`: offline/online parity suites
- `tests/data/polars_offline_baseline.py`: offline reference runner

## Documentation

- Getting started: `docs/getting_started/quickstart.md`
- Testing guide: `docs/developer_guide/testing.md`
- Operator extension: `docs/developer_guide/operators.md`

## Quick Start

```bash
cargo test -p factor_engine --lib
cargo test -p factor_engine --test offline_online_parity
```

## Add a New Operator

1. Implement kernel in `src/ops/elem.rs`, `src/ops/ts.rs`, or `src/ops/cs.rs`
2. Add opcode in `src/ops/spec.rs`
3. Register one `OpMeta` in `src/ops/catalog.rs` (`OP_METAS`)
4. Add unit/parity tests and run quality gates

`catalog` invariants fail fast on missing/duplicate opcodes and invalid arg/param wiring.

## Quality Gates

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --locked -- -D warnings -D clippy::dbg_macro
cargo test -p cjmm_engine
cargo test --workspace --features offline-all
```

## References

- [alpha_examples](https://github.com/wukan1986/alpha_examples)

## Thanks

Thanks to the `alpha_examples` maintainers and contributors for publishing open Alpha expression examples.
