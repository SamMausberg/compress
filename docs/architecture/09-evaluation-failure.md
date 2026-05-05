# §9 Evaluation, failure, and minimal implementation

Report certified utility, solve rate at threshold, answer accuracy stratified
by evidence level, domain-route confusion, intent-entropy failures,
context/reference recall, vague-predicate definition failures, semantic-parse
recall, mode confusion, entailment false-support recall, source-recall miss
rate, rebuttal-recall miss rate, round-trip realization loss,
citation/context loss, calibrated uncertainty quality, ECE, anytime verifier
false-pass/false-fail bounds, counterexample discovery, support loss,
critical-edge omission rate, substrate omission loss, vector-risk consumption
by channel, influence-risk consumption, active-memory growth, archival
growth, rate-distortion frontier movement, transfer lift, dual-allocation
regret, wall-clock/FLOPs, hidden search, human bits, safety-gate violations,
privacy/declassification failures, macro admissions/demotions, branch factor
after compression, interference entropy, and regression after sleep
compression. Required ablations remove $\mathrm{VNS}_\theta$ typed slots,
selective recurrence, discourse state, context normalizer, semantic atoms,
source-recall guard, rebuttal-recall guard, realization guard, dependence
residualization, vector-risk gate, frontier-valid compression, critical-edge
probes, entailment gate, executable projection, parse posterior, domain
routing, support-loss guard, substrate-recall guard, type masks, authority
masks, cumulative-risk budget, influence-risk gate, e-graph merging, verifier
feedback, falsifier, active tests, residual escrow, macro demotion,
cross-fitting, replay, dual compute allocation, and renderer gates.

**Criterion 1** *(Architectural failure).* VPM-5.3 fails as a base
architecture if it cannot beat enumerative search and same-budget
transformer, state-space, and neural program-synthesis baselines on
hidden-schema mechanism tasks; if $\mathrm{VNS}_\theta$ omits certifying
traces at nonvanishing rate after exact rehydration; if context
normalization collapses unresolved references or vague predicates into a
single unsupported parse; if semantic contractization collapses ambiguous
user language into a single unsupported parse; if source retrieval misses
minimally sufficient evidence sets at nonvanishing rate; if rebuttal
retrieval misses material defeaters at nonvanishing rate; if round-trip
realization permits hidden factual atoms; if entailment checking permits
unsupported paraphrases; if active memory grows linearly with solved tasks;
if compiler, sparse-state, context, semantic, source, rebuttal, realization,
or substrate support loss dominates after active tests; if near-miss attacks
keep verifier false-pass bounds high; if dependence residuals or split
leakage create certificate inflation; if gains require hidden test-time
compute; if residuals carry most capability without certificates; if
compression requires opaque macros without equivalence witnesses or positive
held-out frontier movement; if cumulative risk is hidden in many small gated
actions or by scalar cancellation; if influence risk is hidden in soft
spans; if safety gates are bypassed by data-only inputs; if verifiers
certify same-split generators; if support-recall or dependence calibration
is miscalibrated under shift; if broad language/world-knowledge claims are
rendered when $\mathrm{DomOK} = 0$ or $\mathrm{Gate}_{NL} = 0$; or if
inference depends on an external LLM as an uncertified cognitive component.

A minimal VPM-0 implements $\mathrm{VNS}_\theta$ with typed event encoders,
selective recurrence, slots, discourse state, context normal forms, semantic
atoms, source recall, rebuttal recall, round-trip realization checking,
executable projection, and calibrated substrate recall on typed grids,
strings, finite-state machines, small graphs/mazes, simple diagrams, data
transformations, theorem fragments, source-grounded QA over a fixed corpus,
controlled dialogue, and small causal worlds. exact primitive verifiers;
dependence-block residualization; falsifier-generated near misses; active
questions; an authority lattice; vector-risk ledgers; e-value certificates;
support-loss bounds; frontier-valid macro admission/demotion; and matched
state-space/enumerative baselines. The primary success signal is a
compression phase transition: after replay-safe macro admission, solved-task
cost falls, active memory grows sublinearly, held-out certified utility
rises, rate-distortion frontier decreases, support, context, semantic,
source, rebuttal, realization, dependency, and substrate loss remain
bounded, vector-risk budgets remain valid, and false-pass bounds do not
rise.

---

> **Implementation pointers.**
>
> - Evaluation harness, metric registry, and stratified reporters →
>   `python/vpm/evaluation/{metrics,report,strata}.py`.
> - Ablation runner that toggles each architectural component listed above
>   → `python/vpm/evaluation/ablations.py`.
> - Failure criterion (Criterion 1) is encoded as a
>   `vpm.audit.failure_modes:FailureMode` enum; CI integration tests
>   under `tests/integration/failure_modes/` assert each clause for the
>   shipped baseline.
> - VPM-0 minimal implementation entry point → `examples/vpm0/run.py` (see
>   `docs/implementation/README.md` for the milestone schedule).
> - Compression-phase-transition diagnostic →
>   `vpm.evaluation.phase_transition`.
