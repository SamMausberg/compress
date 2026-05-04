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
