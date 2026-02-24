use crate::ops::stats::{collect_window_bivariate_moments, collect_window_moments};
use crate::ops::KernelIo;
use crate::plan::LogicalParam;
use crate::state::EngineState;

const VAR_NUM_EPS: f64 = 1e-12;

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

pub fn ts_mean(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_mean requires window > 0");
    let input = state.field_store.get(instrument_idx, input_field_slot);
    if let Some(moments) = collect_window_moments(&input.ring, window, 1) {
        set_valid_or_invalid(state, instrument_idx, output_slot, moments.mean());
    } else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
    }
}

pub fn ts_sum(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_sum requires window > 0");
    let input = state.field_store.get(instrument_idx, input_field_slot);
    if let Some(moments) = collect_window_moments(&input.ring, window, 1) {
        set_valid_or_invalid(state, instrument_idx, output_slot, moments.sum);
    } else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
    }
}

pub fn ts_product(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_product requires window > 0");
    let input = state.field_store.get(instrument_idx, input_field_slot);
    if input.ring.len() < window {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    }

    let mut out = 1.0_f64;
    for lag in 0..window {
        let Some(v) = input.ring.get_lag(lag) else {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
            return;
        };
        if !v.is_finite() {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
            return;
        }
        out *= v;
    }
    set_valid_or_invalid(state, instrument_idx, output_slot, out);
}

pub fn ts_min(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_min requires window > 0");
    let input = state.field_store.get(instrument_idx, input_field_slot);
    if input.ring.len() < window {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    }
    let mut out = f64::INFINITY;
    for lag in 0..window {
        let Some(v) = input.ring.get_lag(lag) else {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
            return;
        };
        if !v.is_finite() {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
            return;
        }
        out = out.min(v);
    }
    set_valid_or_invalid(state, instrument_idx, output_slot, out);
}

pub fn ts_max(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_max requires window > 0");
    let input = state.field_store.get(instrument_idx, input_field_slot);
    if input.ring.len() < window {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    }
    let mut out = f64::NEG_INFINITY;
    for lag in 0..window {
        let Some(v) = input.ring.get_lag(lag) else {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
            return;
        };
        if !v.is_finite() {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
            return;
        }
        out = out.max(v);
    }
    set_valid_or_invalid(state, instrument_idx, output_slot, out);
}

pub fn ts_mad(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_mad requires window > 0");
    let input = state.field_store.get(instrument_idx, input_field_slot);
    if input.ring.len() < window {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    }

    let mut sum = 0.0_f64;
    for lag in 0..window {
        let Some(v) = input.ring.get_lag(lag) else {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
            return;
        };
        if !v.is_finite() {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
            return;
        }
        sum += v;
    }
    let mean = sum / window as f64;

    let mut abs_sum = 0.0_f64;
    for lag in 0..window {
        let Some(v) = input.ring.get_lag(lag) else {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
            return;
        };
        abs_sum += (v - mean).abs();
    }
    set_valid_or_invalid(state, instrument_idx, output_slot, abs_sum / window as f64);
}

pub fn ts_std(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_std requires window > 0");
    let input = state.field_store.get(instrument_idx, input_field_slot);
    if let Some(moments) = collect_window_moments(&input.ring, window, 2) {
        set_valid_or_invalid(state, instrument_idx, output_slot, moments.std());
    } else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
    }
}

pub fn ts_var(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_var requires window > 0");
    let input = state.field_store.get(instrument_idx, input_field_slot);
    if let Some(moments) = collect_window_moments(&input.ring, window, 2) {
        if moments.n <= 1.0 {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
            return;
        }
        let m2 = moments.sum_sq - (moments.sum * moments.sum) / moments.n;
        let variance = (m2 / (moments.n - 1.0)).max(0.0);
        set_valid_or_invalid(state, instrument_idx, output_slot, variance);
    } else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
    }
}

pub fn ts_skew(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_skew requires window > 0");
    let input = state.field_store.get(instrument_idx, input_field_slot);
    if let Some(moments) = collect_window_moments(&input.ring, window, 3) {
        set_valid_or_invalid(state, instrument_idx, output_slot, moments.skew());
    } else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
    }
}

pub fn ts_kurt(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_kurt requires window > 0");
    let input = state.field_store.get(instrument_idx, input_field_slot);
    if let Some(moments) = collect_window_moments(&input.ring, window, 4) {
        set_valid_or_invalid(state, instrument_idx, output_slot, moments.kurt());
    } else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
    }
}

pub fn delta(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let lag = p.lag().unwrap_or(0);
    debug_assert!(lag > 0, "delta requires lag > 0");
    let input = state.field_store.get(instrument_idx, input_field_slot);
    let cur = input.ring.get_lag(0);
    let old = input.ring.get_lag(lag);
    if let (Some(cur), Some(old)) = (cur, old) {
        set_valid_or_invalid(state, instrument_idx, output_slot, cur - old);
    } else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
    }
}

pub fn ts_lag(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let lag = p.lag().unwrap_or(0);
    debug_assert!(lag > 0, "ts_lag requires lag > 0");
    let input = state.field_store.get(instrument_idx, input_field_slot);
    if let Some(v) = input.ring.get_lag(lag) {
        set_valid_or_invalid(state, instrument_idx, output_slot, v);
    } else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
    }
}

pub fn ts_zscore(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_zscore requires window > 0");

    let input = state.field_store.get(instrument_idx, input_field_slot);
    let Some(latest) = input.ring.get_lag(0) else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    };
    let Some(moments) = collect_window_moments(&input.ring, window, 2) else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    };
    let std = moments.std();
    if !std.is_finite() || std <= 0.0 {
        state.set_node_output(instrument_idx, output_slot, 0.0, true);
        return;
    }
    let z = (latest - moments.mean()) / std;
    set_valid_or_invalid(state, instrument_idx, output_slot, z);
}

pub fn ts_cov(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let lhs_slot = io.input(0);
    let rhs_slot = io.input(1);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_cov requires window > 0");

    let lhs = state.field_store.get(instrument_idx, lhs_slot);
    let rhs = state.field_store.get(instrument_idx, rhs_slot);
    let Some(m) = collect_window_bivariate_moments(&lhs.ring, &rhs.ring, window) else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    };
    if m.n <= 1.0 {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    }
    let cov = (m.sum_xy - (m.sum_x * m.sum_y) / m.n) / (m.n - 1.0);
    set_valid_or_invalid(state, instrument_idx, output_slot, cov);
}

pub fn ts_beta(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let lhs_slot = io.input(0);
    let rhs_slot = io.input(1);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_beta requires window > 0");

    let lhs = state.field_store.get(instrument_idx, lhs_slot);
    let rhs = state.field_store.get(instrument_idx, rhs_slot);
    let Some(m) = collect_window_bivariate_moments(&lhs.ring, &rhs.ring, window) else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    };
    let var_y_num = m.sum_yy - (m.sum_y * m.sum_y) / m.n;
    if var_y_num <= VAR_NUM_EPS {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    }
    let cov_num = m.sum_xy - (m.sum_x * m.sum_y) / m.n;
    let beta = cov_num / var_y_num;
    set_valid_or_invalid(state, instrument_idx, output_slot, beta);
}

#[inline]
fn ewm_alpha(window: usize) -> f64 {
    2.0 / (window as f64 + 1.0)
}

fn ewm_weighted_mean_var(ring: &crate::state::RingBuffer, window: usize) -> Option<(f64, f64)> {
    if window == 0 || ring.len() < window {
        return None;
    }
    let alpha = ewm_alpha(window);
    let decay = 1.0 - alpha;
    let mut weight = 1.0;
    let mut weight_sum = 0.0;
    let mut wx_sum = 0.0;
    let mut wx2_sum = 0.0;
    for lag in 0..window {
        let v = ring.get_lag(lag)?;
        if !v.is_finite() {
            return None;
        }
        weight_sum += weight;
        wx_sum += weight * v;
        wx2_sum += weight * v * v;
        weight *= decay;
    }
    if weight_sum <= 0.0 || !weight_sum.is_finite() {
        return None;
    }
    let mean = wx_sum / weight_sum;
    let var = (wx2_sum / weight_sum - mean * mean).max(0.0);
    Some((mean, var))
}

fn ewm_weighted_cov(
    lhs: &crate::state::RingBuffer,
    rhs: &crate::state::RingBuffer,
    window: usize,
) -> Option<f64> {
    if window == 0 || lhs.len() < window || rhs.len() < window {
        return None;
    }
    let alpha = ewm_alpha(window);
    let decay = 1.0 - alpha;
    let mut weight = 1.0;
    let mut weight_sum = 0.0;
    let mut wx_sum = 0.0;
    let mut wy_sum = 0.0;
    let mut wxy_sum = 0.0;
    for lag in 0..window {
        let x = lhs.get_lag(lag)?;
        let y = rhs.get_lag(lag)?;
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        weight_sum += weight;
        wx_sum += weight * x;
        wy_sum += weight * y;
        wxy_sum += weight * x * y;
        weight *= decay;
    }
    if weight_sum <= 0.0 || !weight_sum.is_finite() {
        return None;
    }
    let mean_x = wx_sum / weight_sum;
    let mean_y = wy_sum / weight_sum;
    Some(wxy_sum / weight_sum - mean_x * mean_y)
}

pub fn ts_ewm_mean(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_ewm_mean requires window > 0");
    let input = state.field_store.get(instrument_idx, input_field_slot);
    if let Some((mean, _)) = ewm_weighted_mean_var(&input.ring, window) {
        set_valid_or_invalid(state, instrument_idx, output_slot, mean);
    } else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
    }
}

pub fn ts_ewm_var(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_ewm_var requires window > 0");
    let input = state.field_store.get(instrument_idx, input_field_slot);
    if let Some((_, var)) = ewm_weighted_mean_var(&input.ring, window) {
        set_valid_or_invalid(state, instrument_idx, output_slot, var);
    } else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
    }
}

pub fn ts_ewm_cov(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let lhs_slot = io.input(0);
    let rhs_slot = io.input(1);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_ewm_cov requires window > 0");
    let lhs = state.field_store.get(instrument_idx, lhs_slot);
    let rhs = state.field_store.get(instrument_idx, rhs_slot);
    if let Some(cov) = ewm_weighted_cov(&lhs.ring, &rhs.ring, window) {
        set_valid_or_invalid(state, instrument_idx, output_slot, cov);
    } else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
    }
}

pub fn ts_decay_linear(
    state: &mut EngineState,
    instrument_idx: usize,
    io: KernelIo,
    p: LogicalParam,
) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_decay_linear requires window > 0");
    if window == 0 {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    }

    let input = state.field_store.get(instrument_idx, input_field_slot);
    if input.ring.len() < window {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    }

    let mut weighted_sum = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    for lag in 0..window {
        let Some(v) = input.ring.get_lag(lag) else {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
            return;
        };
        if !v.is_finite() {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
            return;
        }
        let weight = (window - lag) as f64;
        weighted_sum += weight * v;
        weight_sum += weight;
    }

    if weight_sum <= 0.0 || !weight_sum.is_finite() {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    }
    set_valid_or_invalid(
        state,
        instrument_idx,
        output_slot,
        weighted_sum / weight_sum,
    );
}

pub fn ts_corr(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let lhs_slot = io.input(0);
    let rhs_slot = io.input(1);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_corr requires window > 0");

    let lhs = state.field_store.get(instrument_idx, lhs_slot);
    let rhs = state.field_store.get(instrument_idx, rhs_slot);
    let Some(m) = collect_window_bivariate_moments(&lhs.ring, &rhs.ring, window) else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    };

    let cov_num = m.sum_xy - (m.sum_x * m.sum_y) / m.n;
    let var_x_num = m.sum_xx - (m.sum_x * m.sum_x) / m.n;
    let var_y_num = m.sum_yy - (m.sum_y * m.sum_y) / m.n;
    if var_x_num <= VAR_NUM_EPS || var_y_num <= VAR_NUM_EPS {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    }
    let corr = cov_num / (var_x_num.sqrt() * var_y_num.sqrt());
    set_valid_or_invalid(state, instrument_idx, output_slot, corr);
}

pub fn ts_linear_regression(
    state: &mut EngineState,
    instrument_idx: usize,
    io: KernelIo,
    p: LogicalParam,
) {
    let x_slot = io.input(0);
    let y_slot = io.input(1);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_linear_regression requires window > 0");

    let x = state.field_store.get(instrument_idx, x_slot);
    let y = state.field_store.get(instrument_idx, y_slot);
    let Some(m) = collect_window_bivariate_moments(&x.ring, &y.ring, window) else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    };

    let var_x_num = m.sum_xx - (m.sum_x * m.sum_x) / m.n;
    if var_x_num <= VAR_NUM_EPS {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    }
    let cov_num = m.sum_xy - (m.sum_x * m.sum_y) / m.n;
    let slope = cov_num / var_x_num;
    set_valid_or_invalid(state, instrument_idx, output_slot, slope);
}

pub fn ts_argmax(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_argmax requires window > 0");
    let input = state.field_store.get(instrument_idx, input_field_slot);
    if window == 0 || input.ring.len() < window {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    }

    let Some(mut best) = input.ring.get_lag(0) else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    };
    if !best.is_finite() {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    }
    let mut best_lag = 0usize;
    for lag in 1..window {
        let Some(v) = input.ring.get_lag(lag) else {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
            return;
        };
        if !v.is_finite() {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
            return;
        }
        if v > best {
            best = v;
            best_lag = lag;
        }
    }
    state.set_node_output(instrument_idx, output_slot, best_lag as f64, true);
}

pub fn ts_argmin(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_argmin requires window > 0");
    let input = state.field_store.get(instrument_idx, input_field_slot);
    if window == 0 || input.ring.len() < window {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    }

    let Some(mut best) = input.ring.get_lag(0) else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    };
    if !best.is_finite() {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    }
    let mut best_lag = 0usize;
    for lag in 1..window {
        let Some(v) = input.ring.get_lag(lag) else {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
            return;
        };
        if !v.is_finite() {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
            return;
        }
        if v < best {
            best = v;
            best_lag = lag;
        }
    }
    state.set_node_output(instrument_idx, output_slot, best_lag as f64, true);
}

pub fn ts_quantile(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    let q = p.quantile().unwrap_or(f64::NAN);
    debug_assert!(window > 0, "ts_quantile requires window > 0");
    debug_assert!(q.is_finite() && (0.0..=1.0).contains(&q));
    if window == 0 || !q.is_finite() || !(0.0..=1.0).contains(&q) {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        return;
    }

    let mut values = state.scratch.take_tmp_f64(window);
    let quantile = (|| {
        let input = state.field_store.get(instrument_idx, input_field_slot);
        if input.ring.len() < window {
            return None;
        }
        for lag in 0..window {
            let value = input.ring.get_lag(lag)?;
            if !value.is_finite() {
                return None;
            }
            values.push(value);
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if values.len() == 1 {
            return Some(values[0]);
        }
        let idx = q * ((values.len() - 1) as f64);
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        let frac = idx - (lo as f64);
        Some(values[lo] + (values[hi] - values[lo]) * frac)
    })();
    if let Some(value) = quantile {
        set_valid_or_invalid(state, instrument_idx, output_slot, value);
    } else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
    }
    state.scratch.put_tmp_f64(values);
}

pub fn ts_rank(state: &mut EngineState, instrument_idx: usize, io: KernelIo, p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let window = p.window().unwrap_or(0);
    debug_assert!(window > 0, "ts_rank requires window > 0");
    let mut values = state.scratch.take_tmp_f64(window);
    let rank = (|| {
        let input = state.field_store.get(instrument_idx, input_field_slot);
        if input.ring.len() < window {
            return None;
        }
        let latest = input.ring.get_lag(0)?;
        if !latest.is_finite() {
            return None;
        }
        for lag in 0..window {
            let value = input.ring.get_lag(lag)?;
            if !value.is_finite() {
                return None;
            }
            values.push(value);
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lower = values.partition_point(|v| *v < latest);
        let upper = values.partition_point(|v| *v <= latest);
        let avg_rank = ((lower + 1 + upper) as f64) * 0.5;
        Some(if window > 1 {
            (avg_rank - 1.0) / (window as f64 - 1.0)
        } else {
            0.0
        })
    })();
    if let Some(value) = rank {
        set_valid_or_invalid(state, instrument_idx, output_slot, value);
    } else {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
    }
    state.scratch.put_tmp_f64(values);
}
