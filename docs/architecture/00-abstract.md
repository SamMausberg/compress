# VPM-5.3: Verifier-Native Compressive Predictive Mechanisms

*A support-preserving architecture with a language-normalizing neural substrate
for certified inference, compression, tool use, and continual mechanism
formation.*

**Architecture draft — May 2026.**

## Abstract

VPM-5.3 is a budgeted posterior machine over typed executable mechanisms, a
verifier-native neural substrate, proof-carrying compressed libraries, and an
explicit natural-language contract normalizer. The revision treats the
principal failure modes as first-class state: verifier dependence, support
loss, substrate omission, ambiguous context, semantic collapse, source and
rebuttal miss, realization drift, scalar-risk cancellation, active-memory
bloat, non-reusable residual entropy, and training compute spent away from the
certificate boundary. The substrate remains an amortizer rather than an
authority: it proposes typed event graphs, predictive-state tests, mechanisms,
atom plans, source/rebuttal sets, repairs and renderings, while certification
is produced only by split-clean independent evidence, calibrated recall
bounds, executable replay, authority proofs, and vector safety gates. Training
is execution-first and allocation-controlled: exact search, verification,
retrieval, red-team replay, and neural updates are scheduled by marginal
certified utility under dual prices for scarce labels, calibration data,
verifier calls, and exact execution. When $p_v^+$ is too large, verifier
dependence is unresolved, $\eta_s = 0$, support, context, semantic, source,
rebuttal, realization, substrate, dependency, compression-gain, or vector-risk
bounds fail, or intent entropy remains high, the system decomposes, grounds,
asks, narrows, archives, or abstains rather than claiming compression, safety,
or intelligence.

The system is a controlled rate-distortion architecture rather than a generic
text model. It can compress only reusable, contract-stable structure whose
macro has an expansion witness, positive sequence-valid held-out gain, bounded
branch increase, bounded context/semantic/source/rebuttal/realization error,
and acceptable capability risk; incompressible residuals remain archival or
lossy sketches with explicit distortion. Intelligence is identified with
certified frontier movement: reducing the support gap by adding primitives,
tests, memories, verifiers, tools, causal interventions, semantic formalisms,
and compression capsules that lower held-out certified cost without increasing
false-pass, miss, or risk budgets. Safety is vector-valued and
transcript-level: impact, privacy, exfiltration, capability, influence,
conflict, model, and dependency risks are accumulated by confidence sequences
and no scalar average can cancel a violated component.

---

> **Implementation pointers.** This section sets the global program; nothing
> implements only it. Concretely, the failure modes named above become typed
> error variants in `vpm-core` and dedicated training losses
> (`L_support`, `L_dep`, `L_ctx`, `L_sem`, `L_src`, `L_rebut`, `L_real`, …) in
> the Python `vpm.training` package. The “amortizer not authority” discipline
> appears in code as the strict separation between Python-side proposers
> (`vpm.substrate`, `vpm.compiler`, `vpm.retrieval`) and Rust-side certifiers
> (`vpm-verify`, `vpm-ledger`, `vpm-authority`).
