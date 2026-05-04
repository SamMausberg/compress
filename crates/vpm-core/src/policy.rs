use serde::{Deserialize, Serialize};

/// Componentwise residual risk vector.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RiskVector {
    /// External impact risk.
    pub impact: f64,
    /// Privacy/declassification risk.
    pub privacy: f64,
    /// Capability/tool-use risk.
    pub capability: f64,
    /// Influence/manipulation risk.
    pub influence: f64,
    /// Dependence/correlation risk.
    pub dependence: f64,
}

impl RiskVector {
    /// Zero risk in every channel.
    pub const fn zero() -> Self {
        Self {
            impact: 0.0,
            privacy: 0.0,
            capability: 0.0,
            influence: 0.0,
            dependence: 0.0,
        }
    }

    /// Add two componentwise risk vectors.
    #[must_use]
    pub fn plus(self, other: Self) -> Self {
        Self {
            impact: self.impact + other.impact,
            privacy: self.privacy + other.privacy,
            capability: self.capability + other.capability,
            influence: self.influence + other.influence,
            dependence: self.dependence + other.dependence,
        }
    }

    /// True when every channel is within its corresponding budget.
    pub fn within(self, budget: Self) -> bool {
        self.impact <= budget.impact
            && self.privacy <= budget.privacy
            && self.capability <= budget.capability
            && self.influence <= budget.influence
            && self.dependence <= budget.dependence
    }
}

impl Default for RiskVector {
    fn default() -> Self {
        Self::zero()
    }
}

/// Certificate threshold and false-pass budget for a contract.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EvidenceThreshold {
    /// Required excess log evidence after family spending.
    pub min_cert: f64,
    /// Maximum calibrated false-pass probability.
    pub max_false_pass: f64,
    /// Minimum independent verifier weight.
    pub min_evidence_weight: f64,
}

/// Minimal executable contract Γ for the MVP kernel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contract {
    /// Contract name.
    pub name: String,
    /// Output type expected by the success predicate.
    pub output_type: String,
    /// Human-readable success predicate.
    pub success_predicate: String,
    /// Evidence policy.
    pub threshold: EvidenceThreshold,
    /// Componentwise risk budget.
    pub risk_budget: RiskVector,
    /// Maximum verification/execution cost.
    pub max_cost: f64,
    /// Allowed authority labels for gated side effects or rendering.
    pub allowed_authorities: Vec<String>,
    /// Maximum tolerated support/context/semantic/etc. loss.
    pub epsilon: f64,
}

impl Contract {
    /// A conservative local contract for C0 executable tasks.
    pub fn c0() -> Self {
        Self {
            name: "c0-executable-kernel".to_owned(),
            output_type: "value".to_owned(),
            success_predicate: "exact primitive verifier passes".to_owned(),
            threshold: EvidenceThreshold {
                min_cert: 1.0,
                max_false_pass: 0.05,
                min_evidence_weight: 0.5,
            },
            risk_budget: RiskVector {
                impact: 1.0,
                privacy: 0.0,
                capability: 0.5,
                influence: 0.0,
                dependence: 0.5,
            },
            max_cost: 64.0,
            allowed_authorities: vec!["data".to_owned(), "user".to_owned()],
            epsilon: 0.2,
        }
    }
}

/// Scoped certificate computed from calibrated evidence and non-evidence caps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Certificate {
    /// Claim or action being certified.
    pub claim: String,
    /// Sum of independent log e-values.
    pub log_evidence: f64,
    /// Adaptive family spending allocation.
    pub alpha: f64,
    /// Non-evidence caps such as support, context, source, and risk.
    pub caps: Vec<(String, f64)>,
}

impl Certificate {
    /// Excess log evidence after family spending and caps, clamped at zero.
    pub fn score(&self) -> f64 {
        let family_charge = if self.alpha > 0.0 {
            (1.0 / self.alpha).ln()
        } else {
            f64::INFINITY
        };
        let evidence = (self.log_evidence - family_charge).max(0.0);
        self.caps
            .iter()
            .map(|(_, cap)| (*cap).max(0.0))
            .fold(evidence, f64::min)
    }

    /// True when this certificate clears a contract threshold.
    pub fn clears(&self, contract: &Contract) -> bool {
        self.score() >= contract.threshold.min_cert
    }
}

/// Online certifiability vector collapsed to the fields used by the MVP route.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Certifiability {
    /// Estimated terminal verification cost.
    pub verification_cost: f64,
    /// Calibrated false-pass upper bound.
    pub false_pass_upper: f64,
    /// Independent evidence weight.
    pub evidence_weight: f64,
    /// Dependence loss in [0, 1].
    pub dep_loss: f64,
    /// Support loss in [0, 1].
    pub support_loss: f64,
    /// Context loss in [0, 1].
    pub context_loss: f64,
    /// Semantic loss in [0, 1].
    pub semantic_loss: f64,
    /// Source recall loss in [0, 1].
    pub source_loss: f64,
    /// Rebuttal recall loss in [0, 1].
    pub rebuttal_loss: f64,
    /// Realization loss in [0, 1].
    pub realization_loss: f64,
    /// Substrate omission loss in [0, 1].
    pub substrate_loss: f64,
    /// Lower confidence movement of the compression frontier.
    pub frontier_delta: f64,
    /// Cumulative residual risk.
    pub risk: RiskVector,
    /// Intent entropy estimate.
    pub intent_entropy: f64,
}

impl Certifiability {
    /// Low-risk defaults for exact local C0 tasks.
    pub const fn exact_local() -> Self {
        Self {
            verification_cost: 1.0,
            false_pass_upper: 0.01,
            evidence_weight: 1.0,
            dep_loss: 0.0,
            support_loss: 0.0,
            context_loss: 0.0,
            semantic_loss: 0.0,
            source_loss: 0.0,
            rebuttal_loss: 0.0,
            realization_loss: 0.0,
            substrate_loss: 0.0,
            frontier_delta: 0.0,
            risk: RiskVector::zero(),
            intent_entropy: 0.0,
        }
    }

    /// Contract-level domain predicate for the implemented route.
    pub fn dom_ok(self, contract: &Contract) -> bool {
        self.verification_cost <= contract.max_cost
            && self.false_pass_upper <= contract.threshold.max_false_pass
            && self.evidence_weight >= contract.threshold.min_evidence_weight
            && self.dep_loss <= contract.epsilon
            && self.support_loss <= contract.epsilon
            && self.context_loss <= contract.epsilon
            && self.semantic_loss <= contract.epsilon
            && self.source_loss <= contract.epsilon
            && self.rebuttal_loss <= contract.epsilon
            && self.realization_loss <= contract.epsilon
            && self.substrate_loss <= contract.epsilon
            && self.risk.within(contract.risk_budget)
    }
}

/// Priority-ordered route decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Route {
    /// Solve under the current contract.
    Solve,
    /// Ask for clarification.
    Ask,
    /// Narrow scope or rehydrate exact support.
    Narrow,
    /// Retrieve source/rebuttal evidence.
    Ground,
    /// Decompose into smaller contracts.
    Decompose,
    /// Archive as proposal data only.
    Archive,
    /// Refuse to certify or act.
    Abstain,
}

/// Apply the MVP form of the spec's priority-ordered route rule.
pub fn route(contract: &Contract, xi: Certifiability) -> Route {
    if xi.false_pass_upper >= 1.0
        || xi.evidence_weight <= 0.0
        || xi.dep_loss >= 1.0
        || xi.semantic_loss >= 1.0
        || xi.context_loss >= 1.0
        || xi.realization_loss >= 1.0
        || !xi.risk.within(contract.risk_budget)
    {
        return Route::Abstain;
    }
    if xi.intent_entropy > contract.epsilon {
        return Route::Ask;
    }
    if xi.support_loss > contract.epsilon
        || xi.substrate_loss > contract.epsilon
        || xi.dep_loss > contract.epsilon
        || xi.realization_loss > contract.epsilon
    {
        return Route::Narrow;
    }
    if xi.source_loss > contract.epsilon || xi.rebuttal_loss > contract.epsilon {
        return Route::Ground;
    }
    if xi.dom_ok(contract) {
        Route::Solve
    } else if xi.frontier_delta <= 0.0 {
        Route::Archive
    } else {
        Route::Decompose
    }
}

#[cfg(test)]
mod tests {
    use super::{route, Certifiability, Contract, RiskVector, Route};

    #[test]
    fn route_blocks_componentwise_risk() {
        let contract = Contract::c0();
        let mut xi = Certifiability::exact_local();
        xi.risk = RiskVector {
            privacy: 0.1,
            ..RiskVector::zero()
        };
        assert_eq!(route(&contract, xi), Route::Abstain);
    }

    #[test]
    fn route_solves_exact_local_task() {
        assert_eq!(
            route(&Contract::c0(), Certifiability::exact_local()),
            Route::Solve
        );
    }
}
