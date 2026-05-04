use serde::{Deserialize, Serialize};

/// Sequence-valid empirical-Bernstein bound for macro admission.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmpiricalBernsteinReport {
    /// Number of replay gains.
    pub samples: usize,
    /// Number of adaptive candidates tested so far.
    pub candidates_tested: usize,
    /// Mean bounded replay gain.
    pub mean_gain: f64,
    /// Unbiased sample variance when at least two samples exist.
    pub variance: f64,
    /// Multiplicity-corrected empirical-Bernstein radius.
    pub radius: f64,
    /// Adaptivity loss charged inside the radius.
    pub adapt_loss: f64,
    /// Drift penalty.
    pub drift_loss: f64,
    /// Split leakage penalty.
    pub leak_loss: f64,
    /// Adaptive-selection penalty.
    pub selection_loss: f64,
    /// Lower confidence bound from eq. 127.
    pub lcb: f64,
    /// Upper confidence bound from eq. 128.
    pub ucb: f64,
}

/// Compute eqs. 125-128 over bounded replay gains.
pub fn empirical_bernstein_bounds(
    gains: &[f64],
    candidates_tested: usize,
    delta: f64,
    gain_bound: f64,
    adapt_loss: f64,
    drift_loss: f64,
    leak_loss: f64,
    selection_loss: f64,
) -> EmpiricalBernsteinReport {
    let samples = gains.len();
    let mean_gain = mean(gains);
    let variance = sample_variance(gains, mean_gain);
    let candidates = candidates_tested.max(1);
    let delta = unit_open(delta);
    let gain_bound = non_negative(gain_bound);
    let log_term = (3.0 * candidates as f64 / delta).ln().max(0.0);
    let radius = if samples == 0 {
        f64::INFINITY
    } else {
        let n = samples as f64;
        ((2.0 * variance * log_term) / n).sqrt()
            + ((3.0 * gain_bound * log_term) / n)
            + non_negative(adapt_loss)
    };
    let drift = non_negative(drift_loss);
    let leak = non_negative(leak_loss);
    let selection = non_negative(selection_loss);
    EmpiricalBernsteinReport {
        samples,
        candidates_tested: candidates,
        mean_gain,
        variance,
        radius,
        adapt_loss: non_negative(adapt_loss),
        drift_loss: drift,
        leak_loss: leak,
        selection_loss: selection,
        lcb: mean_gain - radius - drift - leak - selection,
        ucb: mean_gain + radius + drift + leak + selection,
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().copied().sum::<f64>() / values.len() as f64
}

fn sample_variance(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let total = values
        .iter()
        .copied()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>();
    total / (values.len() - 1) as f64
}

fn non_negative(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

const fn unit_open(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(1.0e-12, 1.0)
    } else {
        0.05
    }
}

#[cfg(test)]
mod tests {
    use super::empirical_bernstein_bounds;

    #[test]
    fn exact_replay_with_zero_gain_bound_has_tight_bounds() {
        let report =
            empirical_bernstein_bounds(&[0.75, 0.75, 0.75], 1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(close(report.mean_gain, 0.75));
        assert!(close(report.radius, 0.0));
        assert!(close(report.lcb, 0.75));
        assert!(close(report.ucb, 0.75));
    }

    #[test]
    fn uncertain_replay_charges_radius_and_shift_losses() {
        let report = empirical_bernstein_bounds(&[1.0, -1.0], 4, 0.05, 1.0, 0.1, 0.2, 0.3, 0.4);
        assert_eq!(report.samples, 2);
        assert!(report.radius > 0.1);
        assert!(report.lcb < report.mean_gain);
        assert!(report.ucb > report.mean_gain);
    }

    fn close(left: f64, right: f64) -> bool {
        (left - right).abs() <= f64::EPSILON
    }
}
