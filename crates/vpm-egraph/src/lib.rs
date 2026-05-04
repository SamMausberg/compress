//! `vpm-egraph`: e-graph / equality saturation for VPM-5.3.
//!
//! Wraps the `egg` crate (Willsey et al., POPL 2021 — reference [6] of
//! `docs/references.md`) and adapts it to:
//!
//! - **Posterior-valued canonicalization** with reversible witness or a
//!   support-loss bound (eqs. 84–85 of
//!   `docs/architecture/05-compiler-posterior.md`).
//! - **Macro equivalence certificates** `Cert_eq(m)` over the expansion
//!   witness `W_m` (eq. 124 of
//!   `docs/architecture/07-compression-memory.md`).
//! - **Negative siblings**: when a verifier rejects a candidate at one
//!   e-class member, sound siblings inherit a negative certificate (§8,
//!   structural-efficiency paragraph).
//!
//! Soundness contract: this crate may merge e-classes only with a
//! reversible witness, and every merge is recorded in the ledger as a
//! `canon_witness` event so the support-loss bound `ε_canon` (eq. 85) can
//! be audited.

#![cfg_attr(not(test), forbid(unsafe_code))]
