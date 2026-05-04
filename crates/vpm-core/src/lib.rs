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

mod policy;
mod types;

pub use policy::{
    route, Certifiability, Certificate, Contract, EvidenceThreshold, RiskVector, Route,
};
pub use types::{AtomKind, HashId, Mode, SemanticAtom, Value};

/// Compatibility module for code that wants the contract namespace.
pub mod contract {
    pub use crate::{
        route, AtomKind, Certifiability, Certificate, Contract, EvidenceThreshold, HashId, Mode,
        RiskVector, Route, SemanticAtom, Value,
    };
}
