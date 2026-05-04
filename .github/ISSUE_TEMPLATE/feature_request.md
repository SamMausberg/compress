---
name: Feature request
about: A new primitive, verifier, retriever, capsule, or other architectural addition.
title: "[feat] "
labels: ["enhancement"]
---

## What

<!-- One sentence: the addition you're proposing. -->

## Architecture motivation

<!--
Per §2 of `docs/architecture/`, a repair operator ρ is admissible only
when its held-out marginal certified value is positive under all current
dual prices. State which bottleneck this addition targets and how you
expect the certifiability vector Ξ_Γ(T) to move.
-->

- Targeted bottleneck (e.g. `ε_sub`, `ε_ctx`, `ε_src`, `ΔFront^-`, …):
- Expected sign of `ΔS_ver`:
- Vector-risk impact `Δr` (impact, privacy, exfil, capability, influence,
  conflict, model, dep):
- Costs (`c_impl`, `c_train`, `c_cal`, `c_exact`):

## Sketch

<!-- High-level design. Where in the codebase does it live? -->

## Acceptance

- [ ] Architecture cross-reference added to the new module's docstring.
- [ ] Property tests for the relevant invariants in
      `tests/integration/sanity/`.
- [ ] Failure-mode entry added to `tests/integration/failure_modes/` if
      this addition is meant to clear a Criterion 1 clause.
- [ ] Docs page updated under `docs/architecture/` *only* if the change
      reflects a spec amendment; otherwise, only `docs/implementation/`.
