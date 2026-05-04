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
