use crate::ops::KernelIo;
use crate::plan::LogicalParam;
use crate::state::EngineState;

#[inline]
fn set_valid_or_invalid(
    state: &mut EngineState,
    instrument_idx: usize,
    output_slot: usize,
    value: f64,
) {
    if value.is_finite() {
        state.set_node_output(instrument_idx, output_slot, value, true);
    } else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
    }
}

#[inline]
fn set_bool_output(state: &mut EngineState, instrument_idx: usize, output_slot: usize, v: bool) {
    state.set_node_output(instrument_idx, output_slot, if v { 1.0 } else { 0.0 }, true);
}

pub fn elem_add(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let lhs = state.field_store.get(instrument_idx, io.input(0));
    let rhs = state.field_store.get(instrument_idx, io.input(1));
    if !lhs.has_latest || !rhs.has_latest || !lhs.latest.is_finite() || !rhs.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_valid_or_invalid(state, instrument_idx, io.output, lhs.latest + rhs.latest);
}

pub fn elem_abs(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let input = state.field_store.get(instrument_idx, io.input(0));
    if !input.has_latest {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_valid_or_invalid(state, instrument_idx, io.output, input.latest.abs());
}

pub fn elem_exp(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let input = state.field_store.get(instrument_idx, io.input(0));
    if !input.has_latest {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_valid_or_invalid(state, instrument_idx, io.output, input.latest.exp());
}

pub fn elem_log(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let input = state.field_store.get(instrument_idx, io.input(0));
    if !input.has_latest {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_valid_or_invalid(state, instrument_idx, io.output, input.latest.ln());
}

pub fn elem_sign(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let input = state.field_store.get(instrument_idx, io.input(0));
    if !input.has_latest {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_valid_or_invalid(state, instrument_idx, io.output, input.latest.signum());
}

pub fn elem_sqrt(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let input = state.field_store.get(instrument_idx, io.input(0));
    if !input.has_latest {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_valid_or_invalid(state, instrument_idx, io.output, input.latest.sqrt());
}

pub fn elem_clip(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let x = state.field_store.get(instrument_idx, io.input(0));
    let lower = state.field_store.get(instrument_idx, io.input(1));
    let upper = state.field_store.get(instrument_idx, io.input(2));
    if !x.has_latest || !lower.has_latest || !upper.has_latest {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    if lower.latest > upper.latest {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    let clipped = x.latest.max(lower.latest).min(upper.latest);
    set_valid_or_invalid(state, instrument_idx, io.output, clipped);
}

pub fn elem_where(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let cond = state.field_store.get(instrument_idx, io.input(0));
    let then_v = state.field_store.get(instrument_idx, io.input(1));
    let else_v = state.field_store.get(instrument_idx, io.input(2));
    if !cond.has_latest || !then_v.has_latest || !else_v.has_latest {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    if !cond.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    let out = if cond.latest != 0.0 {
        then_v.latest
    } else {
        else_v.latest
    };
    set_valid_or_invalid(state, instrument_idx, io.output, out);
}

pub fn elem_fillna(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let x = state.field_store.get(instrument_idx, io.input(0));
    let fill = state.field_store.get(instrument_idx, io.input(1));
    if !x.has_latest || !fill.has_latest {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    let out = if x.latest.is_finite() {
        x.latest
    } else {
        fill.latest
    };
    set_valid_or_invalid(state, instrument_idx, io.output, out);
}

pub fn elem_sub(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let lhs = state.field_store.get(instrument_idx, io.input(0));
    let rhs = state.field_store.get(instrument_idx, io.input(1));
    if !lhs.has_latest || !rhs.has_latest || !lhs.latest.is_finite() || !rhs.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_valid_or_invalid(state, instrument_idx, io.output, lhs.latest - rhs.latest);
}

pub fn elem_mul(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let lhs = state.field_store.get(instrument_idx, io.input(0));
    let rhs = state.field_store.get(instrument_idx, io.input(1));
    if !lhs.has_latest || !rhs.has_latest || !lhs.latest.is_finite() || !rhs.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_valid_or_invalid(state, instrument_idx, io.output, lhs.latest * rhs.latest);
}

pub fn elem_div(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let lhs = state.field_store.get(instrument_idx, io.input(0));
    let rhs = state.field_store.get(instrument_idx, io.input(1));
    if !lhs.has_latest || !rhs.has_latest || !lhs.latest.is_finite() || !rhs.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_valid_or_invalid(state, instrument_idx, io.output, lhs.latest / rhs.latest);
}

pub fn elem_pow(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let lhs = state.field_store.get(instrument_idx, io.input(0));
    let rhs = state.field_store.get(instrument_idx, io.input(1));
    if !lhs.has_latest || !rhs.has_latest || !lhs.latest.is_finite() || !rhs.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_valid_or_invalid(
        state,
        instrument_idx,
        io.output,
        lhs.latest.powf(rhs.latest),
    );
}

pub fn elem_min(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let lhs = state.field_store.get(instrument_idx, io.input(0));
    let rhs = state.field_store.get(instrument_idx, io.input(1));
    if !lhs.has_latest || !rhs.has_latest || !lhs.latest.is_finite() || !rhs.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_valid_or_invalid(state, instrument_idx, io.output, lhs.latest.min(rhs.latest));
}

pub fn elem_max(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let lhs = state.field_store.get(instrument_idx, io.input(0));
    let rhs = state.field_store.get(instrument_idx, io.input(1));
    if !lhs.has_latest || !rhs.has_latest || !lhs.latest.is_finite() || !rhs.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_valid_or_invalid(state, instrument_idx, io.output, lhs.latest.max(rhs.latest));
}

pub fn elem_signed_power(
    state: &mut EngineState,
    instrument_idx: usize,
    io: KernelIo,
    _p: LogicalParam,
) {
    let lhs = state.field_store.get(instrument_idx, io.input(0));
    let rhs = state.field_store.get(instrument_idx, io.input(1));
    if !lhs.has_latest || !rhs.has_latest || !lhs.latest.is_finite() || !rhs.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    let out = lhs.latest.signum() * lhs.latest.abs().powf(rhs.latest);
    set_valid_or_invalid(state, instrument_idx, io.output, out);
}

pub fn elem_to_int(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let input = state.field_store.get(instrument_idx, io.input(0));
    if !input.has_latest || !input.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_bool_output(state, instrument_idx, io.output, input.latest != 0.0);
}

pub fn elem_not(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let input = state.field_store.get(instrument_idx, io.input(0));
    if !input.has_latest || !input.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_bool_output(state, instrument_idx, io.output, input.latest == 0.0);
}

pub fn elem_lt(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let lhs = state.field_store.get(instrument_idx, io.input(0));
    let rhs = state.field_store.get(instrument_idx, io.input(1));
    if !lhs.has_latest || !rhs.has_latest || !lhs.latest.is_finite() || !rhs.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_bool_output(state, instrument_idx, io.output, lhs.latest < rhs.latest);
}

pub fn elem_le(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let lhs = state.field_store.get(instrument_idx, io.input(0));
    let rhs = state.field_store.get(instrument_idx, io.input(1));
    if !lhs.has_latest || !rhs.has_latest || !lhs.latest.is_finite() || !rhs.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_bool_output(state, instrument_idx, io.output, lhs.latest <= rhs.latest);
}

pub fn elem_gt(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let lhs = state.field_store.get(instrument_idx, io.input(0));
    let rhs = state.field_store.get(instrument_idx, io.input(1));
    if !lhs.has_latest || !rhs.has_latest || !lhs.latest.is_finite() || !rhs.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_bool_output(state, instrument_idx, io.output, lhs.latest > rhs.latest);
}

pub fn elem_ge(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let lhs = state.field_store.get(instrument_idx, io.input(0));
    let rhs = state.field_store.get(instrument_idx, io.input(1));
    if !lhs.has_latest || !rhs.has_latest || !lhs.latest.is_finite() || !rhs.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_bool_output(state, instrument_idx, io.output, lhs.latest >= rhs.latest);
}

pub fn elem_eq(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let lhs = state.field_store.get(instrument_idx, io.input(0));
    let rhs = state.field_store.get(instrument_idx, io.input(1));
    if !lhs.has_latest || !rhs.has_latest || !lhs.latest.is_finite() || !rhs.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_bool_output(state, instrument_idx, io.output, lhs.latest == rhs.latest);
}

pub fn elem_ne(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let lhs = state.field_store.get(instrument_idx, io.input(0));
    let rhs = state.field_store.get(instrument_idx, io.input(1));
    if !lhs.has_latest || !rhs.has_latest || !lhs.latest.is_finite() || !rhs.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_bool_output(state, instrument_idx, io.output, lhs.latest != rhs.latest);
}

pub fn elem_and(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let lhs = state.field_store.get(instrument_idx, io.input(0));
    let rhs = state.field_store.get(instrument_idx, io.input(1));
    if !lhs.has_latest || !rhs.has_latest || !lhs.latest.is_finite() || !rhs.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_bool_output(
        state,
        instrument_idx,
        io.output,
        lhs.latest != 0.0 && rhs.latest != 0.0,
    );
}

pub fn elem_or(state: &mut EngineState, instrument_idx: usize, io: KernelIo, _p: LogicalParam) {
    let lhs = state.field_store.get(instrument_idx, io.input(0));
    let rhs = state.field_store.get(instrument_idx, io.input(1));
    if !lhs.has_latest || !rhs.has_latest || !lhs.latest.is_finite() || !rhs.latest.is_finite() {
        state.set_node_output(instrument_idx, io.output, f64::NAN, false);
        return;
    }
    set_bool_output(
        state,
        instrument_idx,
        io.output,
        lhs.latest != 0.0 || rhs.latest != 0.0,
    );
}
