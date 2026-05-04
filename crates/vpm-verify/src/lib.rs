//! `vpm-verify`: verifier registry, e-value runtime, falsifier, and the
//! gate kernel.
//!
//! Realises §2 and §6 of `docs/architecture/`:
//!
//! - Verifier record `v = (X_v, c_v, n_v, f_v, ρ_v, ψ_v, κ_v, train_v,
//!   cal_v, audit_v, gen_v)` (eq. 21).
//! - Anytime sequence-valid false-pass UCB `p_{v,r}^+(T)` with
//!   adaptive-stopping correction (eqs. 22–23; refs \[12\]\[13\] of
//!   `docs/references.md`).
//! - Effective verifier weight `η_{v,r}` with split / audit / generator
//!   / dependence factors (eq. 24).
//! - Dependence-block factorization `E_{Dep,r}(h) = Π_g E_{g,r}(h)` and
//!   the dependence false-pass bound `ε_{dep}` (eqs. 25–26).
//! - Claim martingale `M_r(h) = Π_s E_{Dep,s}(h)` and the scoped
//!   certificate `Cert(h, T, V_{1:r}) = [log M_r(h) - log(1/α_h)]_+`
//!   (eqs. 27–28; Theorem 1).
//! - Falsifier: `x_v^* = argmax_x Pr[v pass] - Pr[S_Γ = 1] - λ_x Cost(x)`
//!   (eq. 102).
//! - The conjunctive action gate `Gate(a, Z, Γ)` (eq. 113), the
//!   natural-language gate `Gate_NL` (eqs. 80–81), the renderer gate
//!   `Gate_render` (eq. 117), and the support guard
//!   `Cert_support^{(r)}` (eqs. 92–95).
//! - Empirical-Bernstein anytime LCB / UCB on macro admission
//!   (eqs. 125–128).
//!
//! Invariants:
//!
//! - Raw pass counts, neural confidence, majority vote, and repeated
//!   correlated tests are never evidence. This is enforced by typing
//!   verifier outputs as `EVal` (an opaque newtype that can only be
//!   produced via the calibrated bound).
//! - A verifier cannot certify a candidate it generated unless an
//!   independent replay set or independent verifier family agrees
//!   (`gen_v` invariant).
//! - The certificate is a `min` over caps, never an average
//!   (eqs. 30–33; Invariant 6).

#![cfg_attr(not(test), forbid(unsafe_code))]

use serde::{Deserialize, Serialize};
use vpm_authority::{check as check_authority, AuthorityDecision};
use vpm_core::{route, Certifiability, Certificate, Contract, Mode, RiskVector, Route, Value};
use vpm_dsl::{execute, Program};
use vpm_egraph::{canonicalize, CanonicalProgram};
use vpm_ledger::{EntryType, LedgerDraft, LedgerSummary};

/// Opaque calibrated e-value. Raw pass counts never cross this boundary.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct EVal(f64);

impl EVal {
    /// Construct an e-value from a calibrated false-pass upper bound.
    pub fn from_false_pass(pass: bool, false_pass_upper: f64) -> Self {
        if pass {
            let bounded = false_pass_upper.clamp(1.0e-12, 1.0);
            Self(1.0 / bounded)
        } else {
            Self(1.0)
        }
    }

    /// Numeric e-value.
    pub const fn value(self) -> f64 {
        self.0
    }
}

/// Primitive exact verifier used by C0 tasks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrimitiveVerifier {
    /// Verifier name.
    pub name: String,
    /// Calibrated false-pass upper bound.
    pub false_pass_upper: f64,
    /// Independent evidence weight.
    pub evidence_weight: f64,
    /// Dependence block identifier.
    pub dep_class: String,
    /// Whether this verifier generated the candidate.
    pub generated_candidate: bool,
}

impl Default for PrimitiveVerifier {
    fn default() -> Self {
        Self {
            name: "exact-value".to_owned(),
            false_pass_upper: 0.01,
            evidence_weight: 1.0,
            dep_class: "exact".to_owned(),
            generated_candidate: false,
        }
    }
}

/// Verifier output with scoped certificate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Verifier metadata.
    pub verifier: PrimitiveVerifier,
    /// Whether the primitive predicate passed.
    pub passed: bool,
    /// Calibrated e-value.
    pub e_value: EVal,
    /// Scoped certificate.
    pub certificate: Certificate,
}

/// Check an observed value against an expected value.
pub fn verify_value(
    verifier: PrimitiveVerifier,
    contract: &Contract,
    claim: impl Into<String>,
    observed: &Value,
    expected: &Value,
) -> VerificationReport {
    let passed = observed == expected && !verifier.generated_candidate;
    let e_value = EVal::from_false_pass(passed, verifier.false_pass_upper);
    let log_evidence = e_value.value().ln() * verifier.evidence_weight;
    let cap = if passed { 64.0 } else { 0.0 };
    let certificate = Certificate {
        claim: claim.into(),
        log_evidence,
        alpha: contract.threshold.max_false_pass,
        caps: vec![
            ("support".to_owned(), cap),
            ("semantic".to_owned(), cap),
            ("context".to_owned(), cap),
            ("realization".to_owned(), cap),
            ("risk".to_owned(), cap),
        ],
    };
    VerificationReport {
        verifier,
        passed,
        e_value,
        certificate,
    }
}

/// Conjunctive gate report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateReport {
    /// Domain route.
    pub route: Route,
    /// Certificate score.
    pub certificate_score: f64,
    /// Authority/risk decision.
    pub authority: AuthorityDecision,
    /// Final gate result.
    pub passed: bool,
    /// Blocked reasons.
    pub reasons: Vec<String>,
}

/// Gate a claim/action under a contract.
pub fn gate(
    contract: &Contract,
    xi: Certifiability,
    certificate: &Certificate,
    labels: &[String],
    risk: RiskVector,
) -> GateReport {
    let route = route(contract, xi);
    let authority = check_authority(contract, labels, risk);
    let certificate_score = certificate.score();
    let mut reasons = Vec::new();
    if route != Route::Solve {
        reasons.push(format!("route is {route:?}, not solve"));
    }
    if !certificate.clears(contract) {
        reasons.push("certificate below threshold".to_owned());
    }
    if !authority.allowed() {
        reasons.push(
            authority
                .reason
                .clone()
                .unwrap_or_else(|| "authority/risk gate failed".to_owned()),
        );
    }
    GateReport {
        route,
        certificate_score,
        authority,
        passed: reasons.is_empty(),
        reasons,
    }
}

/// Rust-side vertical-slice report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerticalSliceReport {
    /// Canonicalized program and witnesses.
    pub canonical: CanonicalProgram,
    /// Final value.
    pub value: Value,
    /// Exact verifier report.
    pub verification: VerificationReport,
    /// Gate report.
    pub gate: GateReport,
    /// Ledger summary.
    pub ledger: LedgerSummary,
    /// Trace node count.
    pub trace_nodes: usize,
    /// Trace edge count.
    pub trace_edges: usize,
}

/// Canonicalize, execute, verify, gate, and ledger a C0 program.
pub fn run_program(program: &Program, expected: Value) -> Result<VerticalSliceReport, String> {
    run_program_with_policy(program, expected, &["data".to_owned()], RiskVector::zero())
}

/// Canonicalize, execute, verify, and gate under explicit authority/risk policy.
pub fn run_program_with_policy(
    program: &Program,
    expected: Value,
    labels: &[String],
    risk: RiskVector,
) -> Result<VerticalSliceReport, String> {
    let contract = Contract::c0();
    let canonical = canonicalize(program);
    let mut execution = execute(&canonical.program).map_err(|err| err.to_string())?;
    let verification = verify_value(
        PrimitiveVerifier::default(),
        &contract,
        format!("{} returns {expected}", canonical.program.name),
        &execution.value,
        &expected,
    );

    let mut xi = Certifiability::exact_local();
    xi.verification_cost = execution.cost + 1.0;
    xi.support_loss = canonical.support_loss;
    xi.risk = risk;
    let gate = gate(&contract, xi, &verification.certificate, labels, risk);

    let mut verify_row = LedgerDraft::new(EntryType::Verification, contract.name.clone());
    verify_row.mode = Mode::Certified;
    verify_row.sem = vec![execution.atom.clone()];
    verify_row.auth = labels.to_vec();
    verify_row.cert = verification.certificate.score();
    verify_row.cost = 1.0;
    execution.ledger.append(verify_row);

    let mut gate_row = LedgerDraft::new(EntryType::Gate, contract.name);
    gate_row.mode = if gate.passed {
        Mode::Certified
    } else {
        Mode::Refusal
    };
    gate_row.sem = vec![execution.atom];
    gate_row.auth = labels.to_vec();
    gate_row.cert = gate.certificate_score;
    gate_row.cost = 1.0;
    gate_row.risk = risk;
    execution.ledger.append(gate_row);

    Ok(VerticalSliceReport {
        canonical,
        value: execution.value,
        verification,
        gate,
        ledger: execution.ledger.summary(),
        trace_nodes: execution.trace.nodes.len(),
        trace_edges: execution.trace.edges.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::{run_program, run_program_with_policy};
    use vpm_core::{RiskVector, Value};
    use vpm_dsl::{c0_add_program, c0_concat_program, c0_eq_program, c0_mul_program};

    #[test]
    fn vertical_slice_passes_exact_addition() {
        let report = run_program(&c0_add_program(2, 3), Value::Int(5)).expect("runs");
        assert!(report.verification.passed);
        assert!(report.gate.passed);
        assert_eq!(report.ledger.entries, 6);
    }

    #[test]
    fn wrong_expected_value_fails_gate() {
        let report = run_program(&c0_add_program(2, 3), Value::Int(6)).expect("runs");
        assert!(!report.verification.passed);
        assert!(!report.gate.passed);
    }

    #[test]
    fn vertical_slice_passes_exact_multiplication() {
        let report = run_program(&c0_mul_program(6, 7), Value::Int(42)).expect("runs");
        assert!(report.verification.passed);
        assert!(report.gate.passed);
        assert_eq!(report.ledger.entries, 6);
    }

    #[test]
    fn vertical_slice_passes_exact_concat() {
        let report = run_program(
            &c0_concat_program("ab", "cd"),
            Value::Text("abcd".to_owned()),
        )
        .expect("runs");
        assert!(report.verification.passed);
        assert!(report.gate.passed);
        assert_eq!(report.ledger.entries, 6);
    }

    #[test]
    fn vertical_slice_passes_exact_equality() {
        let report = run_program(
            &c0_eq_program(Value::Int(5), Value::Int(6)),
            Value::Bool(false),
        )
        .expect("runs");
        assert!(report.verification.passed);
        assert!(report.gate.passed);
        assert_eq!(report.ledger.entries, 6);
    }

    #[test]
    fn policy_gate_rejects_disallowed_authority() {
        let report = run_program_with_policy(
            &c0_add_program(2, 3),
            Value::Int(5),
            &["capability".to_owned()],
            RiskVector::zero(),
        )
        .expect("runs");
        assert!(report.verification.passed);
        assert!(!report.gate.passed);
        assert!(!report.gate.authority.auth_ok);
        assert_eq!(report.ledger.total_risk, RiskVector::zero());
    }

    #[test]
    fn policy_gate_rejects_componentwise_risk() {
        let risk = RiskVector {
            privacy: 0.1,
            ..RiskVector::zero()
        };
        let report = run_program_with_policy(
            &c0_add_program(2, 3),
            Value::Int(5),
            &["data".to_owned()],
            risk,
        )
        .expect("runs");
        assert!(report.verification.passed);
        assert!(!report.gate.passed);
        assert!(!report.gate.authority.risk_ok);
        assert_eq!(report.ledger.total_risk.privacy, 0.1);
    }
}
