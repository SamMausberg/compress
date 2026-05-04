//! `vpm-ledger`: append-only ledger Λ, trace DAG, and provenance graph.
//!
//! Realises:
//!
//! - The ledger schema `Λ = (id, type, mode, scope, ctx, sem, parents,
//!   entail, src, rebut, real, auth, prov, split, gen, dep, shift,
//!   verifiers, cert, cost, r, taint, frontier, time, hash)` (eq. 3 of
//!   `docs/architecture/01-contract-ledger-invariants.md`).
//! - The content-addressed trace DAG `τ = (N, E)` with node payload
//!   `B_n, c_n, M_n, …, Front_n, r_n` and edge payload
//!   `(a_e, o_e, cost_e, r_e, Cert_e, …, front_e)` (eq. 140 of
//!   `docs/architecture/08-training-system.md`).
//! - The provenance graph used by the cross-fitting split policy and the
//!   support guard (`docs/architecture/05-compiler-posterior.md` and
//!   `docs/architecture/08-training-system.md`).
//!
//! Invariant: every node and edge is keyed by its blake3 content hash, so
//! "every parse, source set, rebuttal set, entailment witness, verifier
//! result, realization check, dependence block, frontier measurement, or
//! risk estimate is computed once per hash and reused by every head whose
//! split policy permits it" (§8, structural-efficiency paragraph).

#![cfg_attr(not(test), forbid(unsafe_code))]
