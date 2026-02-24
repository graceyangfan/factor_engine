# Operator Extension Guide

Goal: add one operator with minimal touch points and no hidden coupling.

## Steps

1. Add kernel implementation in one of:
   - `src/ops/elem.rs`
   - `src/ops/ts.rs`
   - `src/ops/cs.rs`
2. Add opcode in `src/ops/spec.rs` (`OpCode`).
3. Register metadata in `src/ops/catalog.rs` (`OP_METAS`):
   - `arg_spec`
   - `history_spec`
   - `single_kernel` / `multi_kernel`
   - param specs + scratch profiles
4. Add compile/runtime tests:
   - unit test in `src/tests.rs`
   - parity/integration test in `tests/`

## Invariants

- `catalog` validates missing/duplicate opcode registration at startup.
- `KernelParamSpec` must match `CompileArgSpec`.
- `history_spec` must match arg contract.

If any contract is violated, tests should fail immediately.
