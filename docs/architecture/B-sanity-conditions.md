# Appendix B — Executable sanity conditions

The implementation test kernel must check these behavior-level invariants:

- domain routing solves cheap-verifier reusable tasks and abstains/narrows
  when $p_v^+ = 1$, $\eta_v = 0$, unresolved dependence, high shift, high
  intent entropy, high context entropy, high semantic entropy, unit source
  loss, unit rebuttal loss, unit realization loss, vector-risk exhaustion,
  or unit support/substrate loss;
- correlated verifiers do not multiply evidence unless residualized by
  dependence blocks;
- adaptive claim-family spending is finite;
- same-split verifier/candidate/context/semantic/source/rebuttal/realization/dependence/frontier
  events have zero certificate weight;
- pairwise split leakage is rejected rather than merely triple-overlap
  leakage;
- support, dependency, context, semantic, source, rebuttal, realization,
  critical-edge, and substrate caps are nonnegative and equal zero at unit
  loss;
- componentwise vector risk blocks actions even when a scalar average is
  small;
- data-authority tool output cannot satisfy instruction-authority gates;
- explicit grants do not bypass permission, certificate, impact, budget,
  privacy, sandboxing, cumulative-risk, influence-risk, dependency-risk,
  or conflict conjuncts;
- residuals cannot certify side-effecting actions;
- ambiguity-increasing, source-stale, rebuttal-stale, realization-drifting,
  dependence-leaking, frontier-negative, risk-increasing, or
  branch-increasing macros are rejected by sequence-valid admission;
- active macros have expansion/equivalence witnesses and positive held-out
  frontier movement;
- LCB admission and UCB demotion are symmetric;
- EVC halts when marginal verified utility is below cost;
- compiler canonicalization is reversible or support-bounded;
- $\mathrm{VNS}_\theta$ retrieval and critical-edge miss rates are
  calibrated against exact teacher traces;
- source retrieval miss rates are calibrated against sufficient evidence
  sets;
- rebuttal retrieval miss rates are calibrated against material defeaters;
- realization gates reject hidden certified atoms;
- entailment gates reject unsupported paraphrase;
- falsifiers reject training-set coincidences;
- dual compute allocation gives no budget to heads whose marginal certified
  return is negative;
- and renderer claims inherit context, source, rebuttal, realization,
  evidence, privacy, family budget, support, substrate, semantic,
  dependence, vector-risk, domain, and authority labels.

---

> **Implementation pointers.**
>
> - Each bullet above maps to a property test under
>   `tests/integration/sanity/`, named `test_<bullet>.py` (or
>   `test_<bullet>.rs` when the property lives in a Rust crate).
> - The full kernel is wired by `tests/integration/sanity/conftest.py`
>   into a single CI gate: `pytest -m sanity` must pass before any new
>   architectural component (primitive, verifier, retriever, capsule, etc.)
>   is admitted to the active library.
> - Property generators (Hypothesis on the Python side, `proptest` on the
>   Rust side) live in `tests/integration/sanity/strategies/`.
