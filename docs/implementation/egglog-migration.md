# `egg` → `egglog` migration: when, not now

VPM-5.3 cites `egg` (Willsey et al., POPL 2021) as reference [6] for the
e-graph and equality-saturation machinery used in §5 (compiler /
canonicalization, eqs. 84–85) and §7 (macro equivalence certificates,
eq. 124). The Rust implementation lives in [`crates/vpm-egraph`](../../crates/vpm-egraph).

`egglog` is the spiritual successor to `egg`: a Datalog-based engine
that unifies e-graphs with relational reasoning, supports composable
analyses, and is where new ecosystem investment is going. This document
captures the conditions under which a migration would make sense.

## Why we are on `egg`

- The spec directly cites `egg` (ref [6]).
- `egg`'s Rust API for **rebuild + e-class analysis** is what the
  reversible-witness check (eq. 85, `ε_canon = Pr_q[S_Γ(c) ≠ S_Γ(c̄)]`)
  is designed against.
- Macro equivalence certificates `Cert_eq(m)` (eq. 124) are recorded as
  `canon_witness` events tied to specific e-graph edges; the witness
  format follows `egg`'s rewrite-rule trace structure.

## Conditions that would justify migrating

- **Performance.** If saturation latency on `C_1` (hidden-schema
  splits, typed program synthesis) becomes the binding constraint
  during inference, and `egglog` shows >2× speedup on the same
  workload.
- **Composable analyses.** If we need to layer analyses (e.g. cost +
  type + provenance) and `egg`'s analysis trait composition starts
  duplicating logic. `egglog` makes multi-analysis idiomatic.
- **Datalog-shape rewrites.** If the rewrite set we end up writing for
  the typed DSL (`crates/vpm-dsl`) reads more naturally as Datalog
  rules than as `egg::rewrite!` macros.
- **Active-set provenance.** If the active-archival promotion machinery
  (§7 eqs. 129–131) wants to query "which macro definitions are
  provenance-reachable from this e-class," which is a relational query.

## Conditions that would justify staying

- The spec semantics rely on the specific reversible-witness shape
  produced by `egg`'s rebuild, and `egglog`'s rebuild is a different
  shape that would force a re-derivation in the spec.
- We need stable, in-the-wild Rust API guarantees and `egglog` is still
  iterating on its public Rust surface.
- The cost of re-deriving `Cert_eq(m)` against a new engine outweighs
  the performance benefit at our actual workload size.

## What a migration would look like

1. Snapshot the existing `vpm-egraph` API surface and the witness
   format used by the ledger (`canon_witness` events).
2. Stand up `vpm-egraph-egglog` as a parallel crate exposing the same
   API on `egglog`, behind a workspace feature flag.
3. Run the C₁ + C₅ curricula on both implementations; compare
   saturation latency, witness validity, and derived `ε_canon` on
   replay-deterministic seeds.
4. Cut over only if the comparison meets the conditions above.

## Decision (May 2026)

Stay on `egg` 0.11. Revisit annually or when one of the conditions
above is met, whichever comes first.
