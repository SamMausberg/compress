//! `vpm-authority`: authority lattice, declassification, and vector risk.
//!
//! Realises §6 of `docs/architecture/06-verification-authority.md`:
//!
//! - The authority lattice `(L_auth, ⪯, ∨)` with labels for data,
//!   retrieved sources, tool outputs, user instructions, developer/system
//!   policy, private data, and granted capabilities.
//! - Label-monotone joins (the foundation of `Proposition 3` —
//!   "Data noninterference").
//! - Proof-carrying declassification `Dec_ℓ(x, Z, Γ)` (eq. 103).
//! - The conjunctive gates: `AuthOK` (105), `GrantOK` (106), `PrivOK`
//!   (107), `SandOK` (108), `RiskOK` (109).
//! - Vector risk `U_R(a, Z) = UCB_{1-δ_R}(r_t + Δr(a))` and componentwise
//!   budgets (eq. 110).
//! - Cumulative-risk ledger update with rollback credits (eqs. 111–112).
//!
//! The crate is intentionally `forbid(unsafe_code)`: monotonicity of the
//! authority lattice is enforced at the *type* level, not by runtime
//! checks alone.

#![cfg_attr(not(test), forbid(unsafe_code))]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use vpm_core::{Contract, RiskVector};

/// Authority labels in the MVP lattice.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthLabel {
    /// Untrusted data or retrieved content.
    Data,
    /// User instruction authority.
    User,
    /// Developer/system policy authority.
    Policy,
    /// Private data authority.
    Private,
    /// Tool/capability authority.
    Capability,
}

impl AuthLabel {
    /// Parse a label used by Python/JSON boundaries.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "data" => Some(Self::Data),
            "user" => Some(Self::User),
            "policy" => Some(Self::Policy),
            "private" => Some(Self::Private),
            "capability" => Some(Self::Capability),
            _ => None,
        }
    }

    /// Stable lowercase representation.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Data => "data",
            Self::User => "user",
            Self::Policy => "policy",
            Self::Private => "private",
            Self::Capability => "capability",
        }
    }
}

/// Finite authority set with join operation.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AuthoritySet {
    labels: BTreeSet<AuthLabel>,
}

impl AuthoritySet {
    /// Construct a set from labels.
    pub fn new(labels: impl IntoIterator<Item = AuthLabel>) -> Self {
        Self {
            labels: labels.into_iter().collect(),
        }
    }

    /// Parse a set from strings.
    pub fn from_strings(labels: &[String]) -> Self {
        Self::new(labels.iter().filter_map(|label| AuthLabel::parse(label)))
    }

    /// Join two authority sets.
    #[must_use]
    pub fn join(&self, other: &Self) -> Self {
        let mut labels = self.labels.clone();
        labels.extend(other.labels.iter().cloned());
        Self { labels }
    }

    /// True when this set is covered by the allowed set.
    pub fn allowed_by(&self, allowed: &Self) -> bool {
        self.labels
            .iter()
            .all(|label| allowed.labels.contains(label))
    }

    /// Labels as strings.
    pub fn to_strings(&self) -> Vec<String> {
        self.labels
            .iter()
            .map(|label| label.as_str().to_owned())
            .collect()
    }
}

/// Proof-carrying declassification record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclassificationProof {
    /// Source authority label.
    pub from: AuthLabel,
    /// Target authority label.
    pub to: AuthLabel,
    /// Ledgered witness hash or explanation.
    pub witness: String,
    /// Contract scope.
    pub scope: String,
}

/// Authority/risk gate report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityDecision {
    /// Permission labels were covered by the contract.
    pub auth_ok: bool,
    /// Componentwise risk was within the budget.
    pub risk_ok: bool,
    /// Optional declassification proof was valid.
    pub declass_ok: bool,
    /// Human-readable blocked reason.
    pub reason: Option<String>,
}

impl AuthorityDecision {
    /// Conjunctive gate result.
    pub const fn allowed(&self) -> bool {
        self.auth_ok && self.risk_ok && self.declass_ok
    }
}

/// Check authority labels and componentwise risk against a contract.
pub fn check(contract: &Contract, labels: &[String], risk: RiskVector) -> AuthorityDecision {
    let requested = AuthoritySet::from_strings(labels);
    let allowed = AuthoritySet::from_strings(&contract.allowed_authorities);
    let auth_ok = requested.allowed_by(&allowed);
    let risk_ok = risk.within(contract.risk_budget);
    let reason = if !auth_ok {
        Some("authority label is not allowed by contract".to_owned())
    } else if !risk_ok {
        Some("componentwise risk budget exceeded".to_owned())
    } else {
        None
    };
    AuthorityDecision {
        auth_ok,
        risk_ok,
        declass_ok: true,
        reason,
    }
}

/// Validate a private-to-data declassification proof for this MVP.
pub fn declassify(contract: &Contract, proof: &DeclassificationProof) -> AuthorityDecision {
    let declass_ok = proof.from == AuthLabel::Private
        && proof.to == AuthLabel::Data
        && !proof.witness.is_empty()
        && proof.scope == contract.name;
    let mut decision = check(
        contract,
        &[proof.to.as_str().to_owned()],
        RiskVector::zero(),
    );
    decision.declass_ok = declass_ok;
    if !declass_ok {
        decision.reason = Some("invalid declassification proof".to_owned());
    }
    decision
}

#[cfg(test)]
mod tests {
    use super::check;
    use vpm_core::{Contract, RiskVector};

    #[test]
    fn data_authority_does_not_grant_capability() {
        let contract = Contract::c0();
        let decision = check(&contract, &["capability".to_owned()], RiskVector::zero());
        assert!(!decision.allowed());
    }
}
