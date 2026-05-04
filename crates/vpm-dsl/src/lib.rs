//! `vpm-dsl`: typed DSL / bytecode and deterministic batch executor.
//!
//! Realises the "typed DSL/bytecode" and "batch executor" listed at the
//! head of §8 (`docs/architecture/08-training-system.md`):
//!
//! > Training is execution-first. Before neural learning, implement the
//! > typed DSL/bytecode, context normalizer, semantic-contract compiler,
//! > … batch executor, …
//!
//! Concretely:
//!
//! - A typed bytecode whose operator signatures match the typed
//!   primitives used in the dictionary `D_t` (eq. 11 of
//!   `docs/architecture/02-typed-mechanisms-evidence.md`).
//! - A deterministic executor that produces ledgered traces — every
//!   execution is recorded as a node / edge sequence in `vpm-ledger`.
//! - Batching by operator signature and e-class so candidate execution
//!   reuses identical sub-traces (§8, structural-efficiency paragraph).
//! - Memoization keyed by `(state, lower_bound, upper_bound, prune_reason)`
//!   for partial executions (§8 same paragraph).
//!
//! The executor is the only component allowed to write `Cert_obs`,
//! `Cert_cf`, and `Cert_atom` rows for executable modes (`M_cert ∋ math,
//! code, action`). Style / glue spans are handled by `vpm-py` →
//! `vpm.language.render`.

#![cfg_attr(not(test), forbid(unsafe_code))]
