//! `vpm-core`: shared types for the VPM-5.3 architecture.
//!
//! This crate realises the type-level skeleton of the VPM-5.3 spec:
//!
//! - Task, contract `Γ`, and the policy bundle `Π_*` (see
//!   `docs/architecture/01-contract-ledger-invariants.md`).
//! - Mechanism `μ` with typed state, transition, and observation map
//!   (`Definition 1`, eq. 8 of `docs/architecture/02-typed-mechanisms-evidence.md`).
//! - Speech-act / executable mode partition `M_cert` ∪ `M_soft`
//!   (eqs. 53–54 of `docs/architecture/04-natural-language.md`).
//! - Authority labels and the dependence class `Dep`
//!   (`docs/architecture/06-verification-authority.md`).
//! - Vector risk `r ∈ ℝ^{J_R}_+` and componentwise budgets
//!   (Invariant 6, `docs/architecture/01-contract-ledger-invariants.md`).
//!
//! Nothing in this crate performs inference or verification: those live in
//! `vpm-verify`, `vpm-ledger`, and the Python `vpm` package. This crate
//! exists so every other crate (and the Python bindings) can agree on a
//! single, blake3-content-addressed type system.

#![cfg_attr(not(test), forbid(unsafe_code))]

use serde::{Deserialize, Serialize};
use std::fmt;

/// Content-addressed identifier used by ledgers, traces, and witnesses.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HashId(pub String);

impl HashId {
    /// Hash any serializable value using canonical JSON bytes and BLAKE3.
    pub fn of<T: Serialize + ?Sized>(value: &T) -> Self {
        let bytes = serde_json::to_vec(value).expect("VPM values serialize to JSON");
        Self(blake3::hash(&bytes).to_hex().to_string())
    }
}

impl fmt::Display for HashId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Small typed value universe for the MVP executable kernel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum Value {
    /// Signed integer value.
    Int(i64),
    /// UTF-8 text value.
    Text(String),
    /// Boolean value.
    Bool(bool),
}

impl Value {
    /// Return this value as an integer when its type matches.
    pub const fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(value) => Some(*value),
            Self::Text(_) | Self::Bool(_) => None,
        }
    }

    /// Return this value as text when its type matches.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(value) => Some(value),
            Self::Int(_) | Self::Bool(_) => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(value) => write!(f, "{value}"),
            Self::Text(value) => f.write_str(value),
            Self::Bool(value) => write!(f, "{value}"),
        }
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

/// Ledgered semantic role for an atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AtomKind {
    /// User or environment observation.
    Observation,
    /// Claim that can be certified.
    Claim,
    /// Executable action request.
    Action,
    /// Clarifying question.
    Question,
    /// Program or verifier result.
    Result,
    /// Retrieved support source.
    Source,
    /// Retrieved contradiction or defeating source.
    Rebuttal,
    /// Active or archival memory write.
    Memory,
}

/// Rendering/execution mode partition used by the gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Certified atom in an executable or factual mode.
    Certified,
    /// Soft style/glue span without gated factual authority.
    Soft,
    /// Explicitly scoped assumption.
    Assumption,
    /// Refusal or abstention span.
    Refusal,
}

/// Minimal semantic atom shared by compiler, ledger, verifier, and renderer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticAtom {
    /// Stable content hash over the atom payload.
    pub id: HashId,
    /// Atom role.
    pub kind: AtomKind,
    /// Predicate or operator name.
    pub predicate: String,
    /// Typed arguments.
    pub args: Vec<Value>,
    /// Certified/soft/assumption/refusal mode.
    pub mode: Mode,
    /// Contract scope this atom belongs to.
    pub scope: String,
    /// Parent atom or ledger hashes.
    pub parents: Vec<HashId>,
}

impl SemanticAtom {
    /// Build a semantic atom and derive its stable id.
    pub fn new(
        kind: AtomKind,
        predicate: impl Into<String>,
        args: Vec<Value>,
        mode: Mode,
        scope: impl Into<String>,
        parents: Vec<HashId>,
    ) -> Self {
        #[derive(Serialize)]
        struct Payload<'a> {
            kind: AtomKind,
            predicate: &'a str,
            args: &'a [Value],
            mode: Mode,
            scope: &'a str,
            parents: &'a [HashId],
        }

        let predicate = predicate.into();
        let scope = scope.into();
        let payload = Payload {
            kind,
            predicate: &predicate,
            args: &args,
            mode,
            scope: &scope,
            parents: &parents,
        };
        Self {
            id: HashId::of(&payload),
            kind,
            predicate,
            args,
            mode,
            scope,
            parents,
        }
    }
}

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

/// Compatibility module for code that wants the contract namespace.
pub mod contract {
    pub use super::{
        route, AtomKind, Certifiability, Certificate, Contract, EvidenceThreshold, HashId, Mode,
        RiskVector, Route, SemanticAtom, Value,
    };
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
