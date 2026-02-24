use crate::ops::KernelIo;
use crate::plan::LogicalParam;
use crate::plan::PhysicalPlan;
use crate::plan::MAX_NODE_INPUTS;
use crate::state::EngineState;

const MAX_REGRESSORS: usize = MAX_NODE_INPUTS - 1;
const OLS_EPS: f64 = 1e-12;

#[inline]
fn set_all_invalid(plan: &PhysicalPlan, state: &mut EngineState, output_slot: usize) {
    for instrument_idx in 0..plan.universe_slots.len() {
        state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
    }
}

#[inline]
fn read_constant_scalar(
    plan: &PhysicalPlan,
    state: &EngineState,
    field_slot: usize,
) -> Option<f64> {
    let mut scalar: Option<f64> = None;
    for instrument_idx in 0..plan.universe_slots.len() {
        let field = state.field_store.get(instrument_idx, field_slot);
        if !field.has_latest || !field.latest.is_finite() {
            continue;
        }
        if let Some(v) = scalar {
            if field.latest.to_bits() != v.to_bits() {
                return None;
            }
        } else {
            scalar = Some(field.latest);
        }
    }
    scalar
}

#[inline]
fn quantile_from_sorted(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() || !q.is_finite() || !(0.0..=1.0).contains(&q) {
        return None;
    }
    if sorted.len() == 1 {
        return Some(sorted[0]);
    }
    let idx = q * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    Some(sorted[lo] + (sorted[hi] - sorted[lo]) * frac)
}

#[inline]
fn solve_linear_system(
    dim: usize,
    a: &mut [[f64; MAX_REGRESSORS]; MAX_REGRESSORS],
    b: &mut [f64; MAX_REGRESSORS],
) -> bool {
    debug_assert!(dim > 0 && dim <= MAX_REGRESSORS);
    for pivot_col in 0..dim {
        let mut pivot_row = pivot_col;
        let mut pivot_abs = a[pivot_col][pivot_col].abs();
        let mut row = pivot_col + 1;
        while row < dim {
            let cand = a[row][pivot_col].abs();
            if cand > pivot_abs {
                pivot_abs = cand;
                pivot_row = row;
            }
            row += 1;
        }
        if pivot_abs <= OLS_EPS {
            return false;
        }
        if pivot_row != pivot_col {
            a.swap(pivot_row, pivot_col);
            b.swap(pivot_row, pivot_col);
        }

        let diag = a[pivot_col][pivot_col];
        for col in pivot_col..dim {
            a[pivot_col][col] /= diag;
        }
        b[pivot_col] /= diag;

        for row in 0..dim {
            if row == pivot_col {
                continue;
            }
            let factor = a[row][pivot_col];
            if factor.abs() <= OLS_EPS {
                continue;
            }
            for col in pivot_col..dim {
                a[row][col] -= factor * a[pivot_col][col];
            }
            b[row] -= factor * b[pivot_col];
        }
    }
    true
}

#[derive(Clone, Copy)]
struct CsOlsRow {
    idx: usize,
    y: f64,
    x: [f64; MAX_REGRESSORS],
}

fn standardize_group(
    state: &mut EngineState,
    output_slot: usize,
    valid_indices: &[usize],
    weight_slot: Option<usize>,
) {
    if valid_indices.is_empty() {
        return;
    }
    if let Some(weight_slot) = weight_slot {
        let mut sum_w = 0.0_f64;
        let mut mean = 0.0_f64;
        for &idx in valid_indices {
            let w_state = state.field_store.get(idx, weight_slot);
            if !(w_state.has_latest && w_state.latest.is_finite() && w_state.latest > 0.0) {
                state.set_node_output(idx, output_slot, f64::NAN, false);
                continue;
            }
            let out_idx = state.cell_idx(idx, output_slot);
            if !state.node_valid[out_idx] {
                continue;
            }
            let w = w_state.latest;
            sum_w += w;
            mean += w * state.node_outputs[out_idx];
        }
        if sum_w <= OLS_EPS {
            for &idx in valid_indices {
                state.set_node_output(idx, output_slot, f64::NAN, false);
            }
            return;
        }
        mean /= sum_w;
        let mut var_num = 0.0_f64;
        for &idx in valid_indices {
            let w_state = state.field_store.get(idx, weight_slot);
            if !(w_state.has_latest && w_state.latest.is_finite() && w_state.latest > 0.0) {
                continue;
            }
            let out_idx = state.cell_idx(idx, output_slot);
            if !state.node_valid[out_idx] {
                continue;
            }
            let centered = state.node_outputs[out_idx] - mean;
            state.node_outputs[out_idx] = centered;
            var_num += w_state.latest * centered * centered;
        }
        if var_num <= OLS_EPS {
            for &idx in valid_indices {
                state.set_node_output(idx, output_slot, f64::NAN, false);
            }
            return;
        }
        let std = (var_num / sum_w).sqrt();
        if !std.is_finite() || std <= OLS_EPS {
            for &idx in valid_indices {
                state.set_node_output(idx, output_slot, f64::NAN, false);
            }
            return;
        }
        for &idx in valid_indices {
            let out_idx = state.cell_idx(idx, output_slot);
            if state.node_valid[out_idx] {
                state.node_outputs[out_idx] /= std;
            }
        }
        return;
    }

    if valid_indices.len() < 2 {
        for &idx in valid_indices {
            state.set_node_output(idx, output_slot, f64::NAN, false);
        }
        return;
    }
    let mut mean = 0.0_f64;
    for &idx in valid_indices {
        let out_idx = state.cell_idx(idx, output_slot);
        mean += state.node_outputs[out_idx];
    }
    mean /= valid_indices.len() as f64;
    let mut var_num = 0.0_f64;
    for &idx in valid_indices {
        let out_idx = state.cell_idx(idx, output_slot);
        let centered = state.node_outputs[out_idx] - mean;
        state.node_outputs[out_idx] = centered;
        var_num += centered * centered;
    }
    let variance = var_num / (valid_indices.len() as f64 - 1.0);
    if !variance.is_finite() || variance <= OLS_EPS {
        for &idx in valid_indices {
            state.set_node_output(idx, output_slot, f64::NAN, false);
        }
        return;
    }
    let std = variance.sqrt();
    for &idx in valid_indices {
        let out_idx = state.cell_idx(idx, output_slot);
        state.node_outputs[out_idx] /= std;
    }
}

fn cs_center_group(
    state: &mut EngineState,
    output_slot: usize,
    y_slot: usize,
    indices: &[usize],
    weight_slot: Option<usize>,
    standardize: bool,
) {
    let mut valid = Vec::with_capacity(indices.len());
    let mut sum = 0.0_f64;
    let mut sum_w = 0.0_f64;
    for &idx in indices {
        let y_state = state.field_store.get(idx, y_slot);
        if !(y_state.has_latest && y_state.latest.is_finite()) {
            continue;
        }
        let w = if let Some(weight_slot) = weight_slot {
            let w_state = state.field_store.get(idx, weight_slot);
            if !(w_state.has_latest && w_state.latest.is_finite() && w_state.latest > 0.0) {
                continue;
            }
            w_state.latest
        } else {
            1.0
        };
        valid.push(idx);
        sum += y_state.latest * w;
        sum_w += w;
    }
    if valid.is_empty() || sum_w <= OLS_EPS {
        return;
    }
    let mean = sum / sum_w;
    for &idx in &valid {
        let y_state = state.field_store.get(idx, y_slot);
        state.set_node_output(idx, output_slot, y_state.latest - mean, true);
    }
    if standardize {
        standardize_group(state, output_slot, &valid, weight_slot);
    }
}

fn cs_ols_group(
    state: &mut EngineState,
    output_slot: usize,
    y_slot: usize,
    reg_slots: &[usize],
    indices: &[usize],
    weight_slot: Option<usize>,
    standardize: bool,
) {
    let reg_count = reg_slots.len();
    if reg_count == 0 || reg_count > MAX_REGRESSORS {
        return;
    }
    let mut rows = Vec::with_capacity(indices.len());
    let mut sum_w = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_x = [0.0_f64; MAX_REGRESSORS];
    let mut sum_xy = [0.0_f64; MAX_REGRESSORS];
    let mut sum_xx = [[0.0_f64; MAX_REGRESSORS]; MAX_REGRESSORS];

    for &idx in indices {
        let y_state = state.field_store.get(idx, y_slot);
        if !(y_state.has_latest && y_state.latest.is_finite()) {
            continue;
        }
        let w = if let Some(weight_slot) = weight_slot {
            let w_state = state.field_store.get(idx, weight_slot);
            if !(w_state.has_latest && w_state.latest.is_finite() && w_state.latest > 0.0) {
                continue;
            }
            w_state.latest
        } else {
            1.0
        };
        let mut row = CsOlsRow {
            idx,
            y: y_state.latest,
            x: [0.0_f64; MAX_REGRESSORS],
        };
        let mut valid_row = true;
        for (j, &slot) in reg_slots.iter().enumerate() {
            let x_state = state.field_store.get(idx, slot);
            if !(x_state.has_latest && x_state.latest.is_finite()) {
                valid_row = false;
                break;
            }
            row.x[j] = x_state.latest;
        }
        if !valid_row {
            continue;
        }
        rows.push(row);
        sum_w += w;
        sum_y += w * y_state.latest;
        for i in 0..reg_count {
            let xi = row.x[i];
            sum_x[i] += w * xi;
            sum_xy[i] += w * xi * y_state.latest;
            for j in i..reg_count {
                sum_xx[i][j] += w * xi * row.x[j];
            }
        }
    }

    if rows.is_empty() || sum_w <= OLS_EPS {
        return;
    }
    for i in 0..reg_count {
        for j in 0..i {
            sum_xx[i][j] = sum_xx[j][i];
        }
    }

    let mean_y = sum_y / sum_w;
    let mut mean_x = [0.0_f64; MAX_REGRESSORS];
    let mut cov_xx = [[0.0_f64; MAX_REGRESSORS]; MAX_REGRESSORS];
    let mut cov_xy = [0.0_f64; MAX_REGRESSORS];
    for i in 0..reg_count {
        mean_x[i] = sum_x[i] / sum_w;
    }
    for i in 0..reg_count {
        cov_xy[i] = (sum_xy[i] / sum_w) - mean_x[i] * mean_y;
        for j in 0..reg_count {
            cov_xx[i][j] = (sum_xx[i][j] / sum_w) - mean_x[i] * mean_x[j];
        }
    }

    let mut beta = [0.0_f64; MAX_REGRESSORS];
    if reg_count == 1 {
        let var_x = cov_xx[0][0];
        if var_x > OLS_EPS {
            beta[0] = cov_xy[0] / var_x;
        }
    } else {
        let mut rhs = cov_xy;
        if solve_linear_system(reg_count, &mut cov_xx, &mut rhs) {
            beta[..reg_count].copy_from_slice(&rhs[..reg_count]);
        }
    }
    let mut alpha = mean_y;
    for i in 0..reg_count {
        alpha -= beta[i] * mean_x[i];
    }

    let mut valid_indices = Vec::with_capacity(rows.len());
    for row in rows {
        let mut fitted = alpha;
        let mut i = 0usize;
        while i < reg_count {
            fitted += beta[i] * row.x[i];
            i += 1;
        }
        let residual = row.y - fitted;
        if residual.is_finite() {
            state.set_node_output(row.idx, output_slot, residual, true);
            valid_indices.push(row.idx);
        }
    }

    if standardize {
        standardize_group(state, output_slot, &valid_indices, weight_slot);
    }
}

fn cs_neutralize_core(plan: &PhysicalPlan, state: &mut EngineState, io: KernelIo, p: LogicalParam) {
    let output_slot = io.output;
    let Some((reg_count_u8, has_group, has_weights, standardize)) = p.cs_neutralize() else {
        set_all_invalid(plan, state, output_slot);
        return;
    };
    let reg_count = reg_count_u8 as usize;
    if reg_count > MAX_REGRESSORS {
        set_all_invalid(plan, state, output_slot);
        return;
    }
    let expected_inputs = 1 + reg_count + usize::from(has_group) + usize::from(has_weights);
    if io.input_count as usize != expected_inputs {
        set_all_invalid(plan, state, output_slot);
        return;
    }
    set_all_invalid(plan, state, output_slot);
    let y_slot = io.input(0);
    let reg_slots = &io.inputs[1..(1 + reg_count)];
    let group_slot = has_group.then(|| io.input(1 + reg_count));
    let weight_slot = has_weights.then(|| io.input(1 + reg_count + usize::from(has_group)));

    let mut groups = std::collections::BTreeMap::<u64, Vec<usize>>::new();
    for idx in 0..plan.universe_slots.len() {
        let y_state = state.field_store.get(idx, y_slot);
        if !(y_state.has_latest && y_state.latest.is_finite()) {
            continue;
        }
        let key = if let Some(group_slot) = group_slot {
            let g_state = state.field_store.get(idx, group_slot);
            if !(g_state.has_latest && g_state.latest.is_finite()) {
                continue;
            }
            g_state.latest.to_bits()
        } else {
            0_u64
        };
        groups.entry(key).or_default().push(idx);
    }

    for indices in groups.values() {
        if reg_count == 0 {
            cs_center_group(
                state,
                output_slot,
                y_slot,
                indices,
                weight_slot,
                standardize,
            );
        } else {
            cs_ols_group(
                state,
                output_slot,
                y_slot,
                reg_slots,
                indices,
                weight_slot,
                standardize,
            );
        }
    }
}

pub fn cs_rank(plan: &PhysicalPlan, state: &mut EngineState, io: KernelIo, _p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let mut rank_pairs = state.scratch.take_rank_pairs(plan.universe_slots.len());
    for instrument_idx in 0..plan.universe_slots.len() {
        let input = state.field_store.get(instrument_idx, input_field_slot);
        if input.has_latest && input.latest.is_finite() {
            rank_pairs.push((instrument_idx, input.latest));
        } else {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        }
    }

    rank_pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let n = rank_pairs.len() as f64;
    let mut i = 0usize;
    while i < rank_pairs.len() {
        let mut j = i + 1;
        while j < rank_pairs.len() && rank_pairs[j].1 == rank_pairs[i].1 {
            j += 1;
        }
        let avg_rank = ((i + 1 + j) as f64) * 0.5;
        let rank = if n > 1.0 {
            (avg_rank - 1.0) / (n - 1.0)
        } else {
            0.0
        };
        for (instrument_idx, _) in rank_pairs.iter().take(j).skip(i) {
            state.set_node_output(*instrument_idx, output_slot, rank, true);
        }
        i = j;
    }
    state.scratch.put_rank_pairs(rank_pairs);
}

pub fn cs_zscore(plan: &PhysicalPlan, state: &mut EngineState, io: KernelIo, _p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let instrument_count = plan.universe_slots.len();
    let mut idx_buf = state.scratch.take_tmp_usize(instrument_count);
    let mut values = state.scratch.take_tmp_f64(instrument_count);

    for instrument_idx in 0..instrument_count {
        let (has_latest, latest) = {
            let input = state.field_store.get(instrument_idx, input_field_slot);
            (input.has_latest, input.latest)
        };
        if has_latest && latest.is_finite() {
            idx_buf.push(instrument_idx);
            values.push(latest);
        } else {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        }
    }

    if values.len() >= 2 {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let var = values
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let std = var.sqrt();
        if std.is_finite() && std > 0.0 {
            for (&instrument_idx, &value) in idx_buf.iter().zip(values.iter()) {
                state.set_node_output(instrument_idx, output_slot, (value - mean) / std, true);
            }
        } else {
            for &instrument_idx in &idx_buf {
                state.set_node_output(instrument_idx, output_slot, 0.0, true);
            }
        }
    } else if values.len() == 1 {
        state.set_node_output(idx_buf[0], output_slot, 0.0, true);
    }

    state.scratch.put_tmp_usize(idx_buf);
    state.scratch.put_tmp_f64(values);
}

pub fn cs_center(plan: &PhysicalPlan, state: &mut EngineState, io: KernelIo, _p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let instrument_count = plan.universe_slots.len();

    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for instrument_idx in 0..instrument_count {
        let input = state.field_store.get(instrument_idx, input_field_slot);
        if input.has_latest && input.latest.is_finite() {
            sum += input.latest;
            count += 1;
        }
    }
    if count == 0 {
        for instrument_idx in 0..instrument_count {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        }
        return;
    }

    let mean = sum / count as f64;
    for instrument_idx in 0..instrument_count {
        let input = state.field_store.get(instrument_idx, input_field_slot);
        if input.has_latest && input.latest.is_finite() {
            state.set_node_output(instrument_idx, output_slot, input.latest - mean, true);
        } else {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        }
    }
}

pub fn cs_norm(plan: &PhysicalPlan, state: &mut EngineState, io: KernelIo, _p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let instrument_count = plan.universe_slots.len();

    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut count = 0usize;
    for instrument_idx in 0..instrument_count {
        let input = state.field_store.get(instrument_idx, input_field_slot);
        if input.has_latest && input.latest.is_finite() {
            min_v = min_v.min(input.latest);
            max_v = max_v.max(input.latest);
            count += 1;
        }
    }
    if count == 0 {
        for instrument_idx in 0..instrument_count {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        }
        return;
    }

    let range = max_v - min_v;
    let flat = !range.is_finite() || range <= 0.0;
    for instrument_idx in 0..instrument_count {
        let input = state.field_store.get(instrument_idx, input_field_slot);
        if input.has_latest && input.latest.is_finite() {
            let out = if flat {
                0.0
            } else {
                (input.latest - min_v) / range
            };
            state.set_node_output(instrument_idx, output_slot, out, true);
        } else {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        }
    }
}

pub fn cs_scale(plan: &PhysicalPlan, state: &mut EngineState, io: KernelIo, _p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let instrument_count = plan.universe_slots.len();

    let mut sum_abs = 0.0_f64;
    let mut count = 0usize;
    for instrument_idx in 0..instrument_count {
        let input = state.field_store.get(instrument_idx, input_field_slot);
        if input.has_latest && input.latest.is_finite() {
            sum_abs += input.latest.abs();
            count += 1;
        }
    }
    if count == 0 || !sum_abs.is_finite() || sum_abs <= 0.0 {
        for instrument_idx in 0..instrument_count {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        }
        return;
    }

    for instrument_idx in 0..instrument_count {
        let input = state.field_store.get(instrument_idx, input_field_slot);
        if input.has_latest && input.latest.is_finite() {
            state.set_node_output(instrument_idx, output_slot, input.latest / sum_abs, true);
        } else {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        }
    }
}

pub fn cs_fillna(plan: &PhysicalPlan, state: &mut EngineState, io: KernelIo, _p: LogicalParam) {
    let input_field_slot = io.input(0);
    let output_slot = io.output;
    let instrument_count = plan.universe_slots.len();

    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for instrument_idx in 0..instrument_count {
        let input = state.field_store.get(instrument_idx, input_field_slot);
        if input.has_latest && input.latest.is_finite() {
            sum += input.latest;
            count += 1;
        }
    }

    if count == 0 {
        for instrument_idx in 0..instrument_count {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        }
        return;
    }

    let mean = sum / count as f64;
    for instrument_idx in 0..instrument_count {
        let input = state.field_store.get(instrument_idx, input_field_slot);
        if input.has_latest && input.latest.is_finite() {
            state.set_node_output(instrument_idx, output_slot, input.latest, true);
        } else {
            state.set_node_output(instrument_idx, output_slot, mean, true);
        }
    }
}

pub fn cs_winsorize(plan: &PhysicalPlan, state: &mut EngineState, io: KernelIo, _p: LogicalParam) {
    let input_field_slot = io.input(0);
    let percentile_slot = io.input(1);
    let output_slot = io.output;
    let Some(percentile) = read_constant_scalar(plan, state, percentile_slot) else {
        set_all_invalid(plan, state, output_slot);
        return;
    };
    if !(0.0..=1.0).contains(&percentile) {
        set_all_invalid(plan, state, output_slot);
        return;
    }

    let mut values = state.scratch.take_tmp_f64(plan.universe_slots.len());
    for instrument_idx in 0..plan.universe_slots.len() {
        let input = state.field_store.get(instrument_idx, input_field_slot);
        if input.has_latest && input.latest.is_finite() {
            values.push(input.latest);
        }
    }
    if values.is_empty() {
        set_all_invalid(plan, state, output_slot);
        state.scratch.put_tmp_f64(values);
        return;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let Some(lower_bound) = quantile_from_sorted(&values, percentile) else {
        set_all_invalid(plan, state, output_slot);
        state.scratch.put_tmp_f64(values);
        return;
    };
    let Some(upper_bound) = quantile_from_sorted(&values, 1.0 - percentile) else {
        set_all_invalid(plan, state, output_slot);
        state.scratch.put_tmp_f64(values);
        return;
    };

    for instrument_idx in 0..plan.universe_slots.len() {
        let input = state.field_store.get(instrument_idx, input_field_slot);
        if input.has_latest && input.latest.is_finite() {
            state.set_node_output(
                instrument_idx,
                output_slot,
                input.latest.clamp(lower_bound, upper_bound),
                true,
            );
        } else {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        }
    }
    state.scratch.put_tmp_f64(values);
}

pub fn cs_percentiles(
    plan: &PhysicalPlan,
    state: &mut EngineState,
    io: KernelIo,
    _p: LogicalParam,
) {
    let input_field_slot = io.input(0);
    let lower_pct_slot = io.input(1);
    let upper_pct_slot = io.input(2);
    let output_slot = io.output;

    let Some(lower_pct) = read_constant_scalar(plan, state, lower_pct_slot) else {
        set_all_invalid(plan, state, output_slot);
        return;
    };
    let Some(upper_pct) = read_constant_scalar(plan, state, upper_pct_slot) else {
        set_all_invalid(plan, state, output_slot);
        return;
    };
    if !(0.0..=1.0).contains(&lower_pct)
        || !(0.0..=1.0).contains(&upper_pct)
        || lower_pct > upper_pct
    {
        set_all_invalid(plan, state, output_slot);
        return;
    }
    let fill = 0.0_f64;

    let mut values = state.scratch.take_tmp_f64(plan.universe_slots.len());
    for instrument_idx in 0..plan.universe_slots.len() {
        let input = state.field_store.get(instrument_idx, input_field_slot);
        if input.has_latest && input.latest.is_finite() {
            values.push(input.latest);
        }
    }
    if values.is_empty() {
        for instrument_idx in 0..plan.universe_slots.len() {
            state.set_node_output(instrument_idx, output_slot, fill, true);
        }
        state.scratch.put_tmp_f64(values);
        return;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let Some(lower_bound) = quantile_from_sorted(&values, lower_pct) else {
        set_all_invalid(plan, state, output_slot);
        state.scratch.put_tmp_f64(values);
        return;
    };
    let Some(upper_bound) = quantile_from_sorted(&values, upper_pct) else {
        set_all_invalid(plan, state, output_slot);
        state.scratch.put_tmp_f64(values);
        return;
    };

    for instrument_idx in 0..plan.universe_slots.len() {
        let input = state.field_store.get(instrument_idx, input_field_slot);
        if input.has_latest && input.latest.is_finite() {
            let value = input.latest;
            if value <= lower_bound || value >= upper_bound {
                state.set_node_output(instrument_idx, output_slot, value, true);
            } else {
                state.set_node_output(instrument_idx, output_slot, fill, true);
            }
        } else {
            state.set_node_output(instrument_idx, output_slot, f64::NAN, false);
        }
    }
    state.scratch.put_tmp_f64(values);
}

pub fn cs_neutralize(plan: &PhysicalPlan, state: &mut EngineState, io: KernelIo, p: LogicalParam) {
    cs_neutralize_core(plan, state, io, p);
}

pub fn cs_neutralize_ols(
    plan: &PhysicalPlan,
    state: &mut EngineState,
    io: KernelIo,
    p: LogicalParam,
) {
    cs_neutralize_core(plan, state, io, p);
}

pub fn cs_neutralize_ols_multi(
    plan: &PhysicalPlan,
    state: &mut EngineState,
    io: KernelIo,
    p: LogicalParam,
) {
    cs_neutralize_core(plan, state, io, p);
}
