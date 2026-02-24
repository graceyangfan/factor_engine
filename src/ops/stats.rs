#[derive(Debug, Clone, Copy)]
pub(super) struct WindowMoments {
    pub(super) n: f64,
    pub(super) sum: f64,
    pub(super) sum_sq: f64,
    pub(super) sum_cu: f64,
    pub(super) sum_qu: f64,
}

impl WindowMoments {
    #[inline]
    pub(super) fn mean(self) -> f64 {
        self.sum / self.n
    }

    #[inline]
    pub(super) fn std(self) -> f64 {
        if self.n <= 1.0 {
            return f64::NAN;
        }
        let m2 = self.sum_sq - (self.sum * self.sum) / self.n;
        let variance = (m2 / (self.n - 1.0)).max(0.0);
        variance.sqrt()
    }

    #[inline]
    pub(super) fn skew(self) -> f64 {
        if self.n <= 2.0 {
            return f64::NAN;
        }
        let std = self.std();
        if !std.is_finite() || std <= 0.0 {
            return f64::NAN;
        }
        let mean = self.mean();
        let mean_sq = mean * mean;
        let m3 = self.sum_cu - 3.0 * mean * self.sum_sq + 3.0 * mean_sq * self.sum
            - self.n * mean_sq * mean;
        let denom = (self.n - 1.0) * (self.n - 2.0) * std.powi(3);
        if denom.abs() <= f64::EPSILON {
            f64::NAN
        } else {
            (self.n * m3) / denom
        }
    }

    #[inline]
    pub(super) fn kurt(self) -> f64 {
        if self.n <= 3.0 {
            return f64::NAN;
        }
        let std = self.std();
        if !std.is_finite() || std <= 0.0 {
            return f64::NAN;
        }
        let mean = self.mean();
        let mean_sq = mean * mean;
        let mean_cu = mean_sq * mean;
        let mean_qu = mean_sq * mean_sq;
        let m4 = self.sum_qu - 4.0 * mean * self.sum_cu + 6.0 * mean_sq * self.sum_sq
            - 4.0 * mean_cu * self.sum
            + self.n * mean_qu;
        let denom = (self.n - 1.0) * (self.n - 2.0) * (self.n - 3.0) * std.powi(4);
        if denom.abs() <= f64::EPSILON {
            f64::NAN
        } else {
            let term1 = (self.n * (self.n + 1.0) * m4) / denom;
            let term2 = 3.0 * (self.n - 1.0).powi(2) / ((self.n - 2.0) * (self.n - 3.0));
            term1 - term2
        }
    }
}

pub(super) fn collect_window_moments(
    ring: &crate::state::RingBuffer,
    window: usize,
    order: u8,
) -> Option<WindowMoments> {
    if window == 0 || ring.len() < window {
        return None;
    }
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut sum_cu = 0.0;
    let mut sum_qu = 0.0;
    for lag in 0..window {
        let value = ring.get_lag(lag)?;
        if !value.is_finite() {
            return None;
        }
        sum += value;
        if order >= 2 {
            let sq = value * value;
            sum_sq += sq;
            if order >= 3 {
                sum_cu += sq * value;
                if order >= 4 {
                    sum_qu += sq * sq;
                }
            }
        }
    }
    Some(WindowMoments {
        n: window as f64,
        sum,
        sum_sq,
        sum_cu,
        sum_qu,
    })
}

#[derive(Debug, Clone, Copy)]
pub(super) struct WindowBivariateMoments {
    pub(super) n: f64,
    pub(super) sum_x: f64,
    pub(super) sum_y: f64,
    pub(super) sum_xx: f64,
    pub(super) sum_yy: f64,
    pub(super) sum_xy: f64,
}

pub(super) fn collect_window_bivariate_moments(
    lhs: &crate::state::RingBuffer,
    rhs: &crate::state::RingBuffer,
    window: usize,
) -> Option<WindowBivariateMoments> {
    if window == 0 || lhs.len() < window || rhs.len() < window {
        return None;
    }
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_xy = 0.0;
    for lag in 0..window {
        let x = lhs.get_lag(lag)?;
        let y = rhs.get_lag(lag)?;
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_yy += y * y;
        sum_xy += x * y;
    }
    Some(WindowBivariateMoments {
        n: window as f64,
        sum_x,
        sum_y,
        sum_xx,
        sum_yy,
        sum_xy,
    })
}
