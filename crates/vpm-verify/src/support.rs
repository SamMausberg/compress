use serde::{Deserialize, Serialize};

/// Follow-up action required by the support guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportAction {
    /// Current support pruning is within the contract budget.
    Accept,
    /// Omitted alternatives must be restored or checked exactly.
    Rehydrate,
}

/// Calibrated support-pruning report from §5 eqs. 92-95.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupportGuardReport {
    /// Candidate count before pruning.
    pub candidates_before: usize,
    /// Candidate count after pruning.
    pub candidates_after: usize,
    /// Posterior mass retained by the current candidate set.
    pub retained_mass: f64,
    /// Eq. 92 mass omitted by pruning.
    pub omitted_mass: f64,
    /// Eq. 93 calibrated recall upper bound.
    pub recall_upper: f64,
    /// Shift residual for the current task.
    pub shift_loss: f64,
    /// Selection residual for adaptive pruning.
    pub selection_loss: f64,
    /// Eq. 94 total support-prune loss.
    pub epsilon_prune: f64,
    /// Eq. 95 non-evidence support certificate cap.
    pub certificate: f64,
    /// Contract maximum tolerated support loss.
    pub epsilon_max: f64,
    /// True when `epsilon_prune <= epsilon_max`.
    pub passed: bool,
    /// Required follow-up action.
    pub action: SupportAction,
}

/// Evaluate the §5 support guard for one pruning decision.
pub fn support_guard(
    candidates_before: usize,
    candidates_after: usize,
    retained_mass: f64,
    recall_upper: f64,
    shift_loss: f64,
    selection_loss: f64,
    epsilon_max: f64,
) -> SupportGuardReport {
    let retained_mass = unit_interval(retained_mass);
    let omitted_mass = unit_interval(1.0 - retained_mass);
    let recall_upper = unit_interval(recall_upper);
    let shift_loss = unit_interval(shift_loss);
    let selection_loss = unit_interval(selection_loss);
    let epsilon_prune = unit_interval(omitted_mass + recall_upper + shift_loss + selection_loss);
    let certificate = (-(epsilon_prune + f64::EPSILON).ln()).max(0.0);
    let epsilon_max = unit_interval(epsilon_max);
    let passed = epsilon_prune <= epsilon_max;
    let action = if passed {
        SupportAction::Accept
    } else {
        SupportAction::Rehydrate
    };
    SupportGuardReport {
        candidates_before,
        candidates_after,
        retained_mass,
        omitted_mass,
        recall_upper,
        shift_loss,
        selection_loss,
        epsilon_prune,
        certificate,
        epsilon_max,
        passed,
        action,
    }
}

const fn unit_interval(value: f64) -> f64 {
    if !value.is_finite() {
        1.0
    } else if value < 0.0 {
        0.0
    } else if value > 1.0 {
        1.0
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::{support_guard, SupportAction};

    #[test]
    fn exact_pruning_passes_support_guard() {
        let report = support_guard(4, 1, 1.0, 0.0, 0.0, 0.0, 0.2);
        assert!(report.passed);
        assert_eq!(report.action, SupportAction::Accept);
        assert!(report.epsilon_prune.abs() <= f64::EPSILON);
        assert!(report.certificate > 1.0);
    }

    #[test]
    fn lossy_topk_pruning_requires_rehydration() {
        let report = support_guard(4, 1, 0.25, 0.0, 0.0, 0.0, 0.2);
        assert!(!report.passed);
        assert_eq!(report.action, SupportAction::Rehydrate);
        assert!((report.omitted_mass - 0.75).abs() <= f64::EPSILON);
        assert!((report.epsilon_prune - 0.75).abs() <= f64::EPSILON);
    }
}
