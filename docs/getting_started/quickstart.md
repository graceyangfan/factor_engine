# Quickstart

## 1) Compile expressions

```rust
use factor_engine::{CompileOptions, FactorRequest, Planner, SimplePlanner};

let planner = SimplePlanner;
let req = FactorRequest {
    exprs: vec!["ts_mean(close, 5)".to_string()],
    outputs: vec!["mean5".to_string()],
    opts: CompileOptions::default(),
};
let (logical, manifest) = planner.compile(&req)?;
assert_eq!(manifest.expr_count, 1);
```

## 2) Bind to universe + input catalog

```rust
use factor_engine::{AdvancePolicy, InputFieldCatalog, Universe};

let catalog = InputFieldCatalog::new(vec!["bar.close".to_string()]);
let physical = planner.bind(
    &logical,
    &Universe::new(vec![1001, 1002]),
    &catalog,
    AdvancePolicy::StrictAllReady,
)?;
```

## 3) Feed events and advance

```rust
use factor_engine::{BarLite, Engine, EventEnvelope, OnlineFactorEngine, Payload};

let mut engine = OnlineFactorEngine::default();
engine.load(physical)?;
engine.on_event(&EventEnvelope {
    ts_event_ns: 1,
    ts_init_ns: 1,
    seq: 1,
    source_slot: 0,
    instrument_slot: 1001,
    quality_flags: 0,
    payload: Payload::Bar(BarLite {
        open: 10.0,
        high: 10.0,
        low: 10.0,
        close: 10.0,
        volume: 100.0,
    }),
})?;
let frame = engine.advance(1)?;
let _ = frame.factor_value(0, "mean5");
```
