# Appendix A — Compact inference procedure

```text
VPM-Infer(O, A, Gamma, B):
  H <- VNS_EncodeUpdate(O, Gamma)                       # typed predictive-state substrate
  Psi <- ConversationNormalFormPosterior(O, H, Gamma)   # context, discourse, atoms, modes, ambiguity
  if not CtxOK(Psi) or not SemOK(Psi): return AskOrScopedUncertainty(Psi, Gamma)
  Gamma <- RefineContract(Gamma, Psi)
  Xi <- EstimateCertifiability(H, Psi, Gamma)
  Beta <- BottleneckVector(H, Psi, Gamma, Xi)
  route <- DomainRoute(Xi, Gamma)
  if route == ground:
      Rsrc <- RetrieveSources(Psi.atoms, Gamma)
      Rreb <- RetrieveRebuttals(Psi.atoms, Gamma)
      Xi <- UpdateSourceRebuttalBounds(Xi, Rsrc, Rreb)
  if route in {Ask, Abstain, archive}: return RouteResponse(route, H, Gamma)
  if route == narrow: O, Gamma <- NarrowScopeOrExactRehydrate(O, H, Psi, Gamma)
  C <- CompilerPosterior(O, Psi, Gamma, H)              # alternatives, provenance, authority
  R <- RetrieveActive(C, Gamma, H)                      # only admitted capsules; archive is proposal data
  S <- InitSlots(C, R, Psi, H, Gamma)
  Alpha <- AllocateClaimFamilyBudget(Gamma, Psi.atoms)
  for r in 0..Rmax:
    H <- VNS_EncodeUpdate(O, Gamma, S)
    S <- MechanismStateCell_theta(S, C, R, H, Gamma)
    eps, eps_sub, eps_ctx, eps_sem, eps_src, eps_rebut, eps_real, eps_dep, rvec, dfront <- CalibratedLosses(S, H, Psi, exact_recall_audit)
    if eps > Gamma.epsilon or eps_sub > Gamma.epsilon_sub or eps_dep > Gamma.epsilon_dep or eps_ctx > Gamma.epsilon_ctx
      or eps_sem > Gamma.epsilon_sem or eps_src > Gamma.epsilon_src or eps_rebut > Gamma.epsilon_rebut or eps_real > Gamma.epsilon_real or not RiskBudgetOK(rvec):
      S <- RehydrateOrWidenOrGroundOrAsk(S, Psi, Beta)
      route <- DomainRoute(EstimateCertifiability(H, Psi, Gamma), Gamma)
      if route in {Ask, Abstain, narrow}: return RouteResponse(route, H, Gamma)
    sketches <- ProjectInvariantsAndEClasses(S)
    programs <- EscalateIfEVCPositive(sketches, S, eps, eps_ctx, eps_src, eps_rebut, eps_real)
    programs <- ExpandWithWitnesses(programs)
    routes <- MultiFidelityRoute(programs, S, Gamma)
    routes <- DependencyBlockAndRiskRoute(routes, S, Gamma)
    results <- BatchExecuteVerifyEntailRebutRealize(routes, C, Psi, Gamma, Alpha)   # block e-value ledger
    attacks <- FalsifyNearMisses(programs, results, Gamma)
    S <- InjectEvidence(S, results, attacks, eps, eps_ctx, eps_sem, eps_src, eps_rebut, eps_real, eps_dep, rvec, dfront)
    y <- CandidateRender(S, Psi, Gamma)
    if ContractSatisfied(S, Gamma) and RenderGate(y, S, Gamma) and Gate_NL(y, S, Gamma) and RealOK(y, PlannedAtoms(S), S, Gamma):
        return RenderCertified(y, S, Gamma)
    if ClarificationDominates(S, Gamma):
        return RenderQuestion(S, Gamma)
    a <- ProposedAction(S)
    if a and Gate(a, S, Gamma) and VectorRiskOK(a, S, Gamma):
        O <- SafeToolOrEnvStep(a); C, R, S <- Update(O, C, R, S)
    if EVC(S) <= 0:
        break
  return BestEffortWithEvidenceLevelOrRefusal(S, Gamma)
```

---

> **Implementation pointers.**
>
> - The pseudocode is the canonical reference for `python/vpm/infer/loop.py`.
>   Each named call maps 1:1 to a function in the implementation:
>   - `VNS_EncodeUpdate` → `vpm.substrate.encode_update`
>   - `ConversationNormalFormPosterior` → `vpm.compiler.cnf_posterior`
>   - `Ctx/SemOK` / `RefineContract` / `EstimateCertifiability` /
>     `BottleneckVector` / `DomainRoute` → `vpm.infer.routing`
>   - `Retrieve{Sources,Rebuttals,Active}` → `vpm.retrieval.*`
>   - `CompilerPosterior` → `vpm.compiler.posterior`
>   - `MechanismStateCell_theta` → `vpm.infer.cell`
>   - `CalibratedLosses` → `vpm.infer.calibrated_losses`
>   - `BatchExecuteVerifyEntailRebutRealize` → `crates/vpm-verify::batch`
>     (Rust)
>   - `FalsifyNearMisses` → `crates/vpm-verify::falsifier` (Rust)
>   - `RenderCertified` / `RenderQuestion` → `vpm.language.render`
>   - `Gate(_NL)` / `RealOK` / `VectorRiskOK` → `crates/vpm-verify::gate*`
> - The outer `for r in 0..Rmax` loop is a typed `vpm.infer.loop.Loop`
>   coroutine; each iteration is a single rust-side transaction over the
>   ledger.
