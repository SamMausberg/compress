# §1 Contract, ledger, and non-negotiable invariants

A task is

$$
T = (O_{0:t}, A, B, R, H, \mathcal{F}_t), \tag{1}
$$

where $O_{0:t}$ are observations, $A$ admissible actions, $B$ resource budgets,
$R$ risk policy, $H$ scoped history, and $\mathcal{F}_t$ the filtration
containing every observation, proposal, verifier choice, tool call, and human
bit. A contract is

$$
\begin{aligned}
\Gamma = (\,& Y, S, V, P, L, F, Q, \Pi_{\mathrm{perm}}, \Pi_{\mathrm{priv}}, \Pi_{\mathrm{audit}}, \Pi_{\mathrm{split}}, \Pi_{\mathrm{risk}}, \Pi_{\mathrm{dom}}, \Pi_{\mathrm{sub}}, \Pi_{\mathrm{sem}}, \Pi_{\mathrm{src}}, \Pi_{\mathrm{ctx}}, \\
& \Pi_{\mathrm{rebut}}, \Pi_{\mathrm{real}}, \Pi_{\mathrm{style}}, \Pi_{\mathrm{infl}}, \Pi_{\mathrm{dep}}, \Pi_{\mathrm{cap}}, \Pi_{\mathrm{compute}}\,).
\end{aligned} \tag{2}
$$

where $Y$ is the output type, $S$ the success predicate, $V$ the evidence
threshold, $P$ provenance requirements, $L$ latency/resource limits, $F$
failure behavior, $Q$ rendering constraints, $\Pi_{\mathrm{perm}}$ permission
policy, $\Pi_{\mathrm{priv}}$ privacy/declassification policy,
$\Pi_{\mathrm{audit}}$ transcript requirements, $\Pi_{\mathrm{split}}$
certification-split policy, $\Pi_{\mathrm{risk}}$ vector
impact/privacy/capability/influence policy, $\Pi_{\mathrm{dom}}$
domain-admissibility policy, $\Pi_{\mathrm{sub}}$ neural-substrate support
policy, $\Pi_{\mathrm{sem}}$ semantic-contract policy, $\Pi_{\mathrm{src}}$
source/evidence-recall policy, $\Pi_{\mathrm{ctx}}$ context/reference policy,
$\Pi_{\mathrm{rebut}}$ contradiction-search policy, $\Pi_{\mathrm{real}}$
atom-to-text realization policy, $\Pi_{\mathrm{style}}$ nonclaim style policy,
$\Pi_{\mathrm{infl}}$ influence/manipulation policy, $\Pi_{\mathrm{dep}}$
verifier/retriever/model-dependence policy, $\Pi_{\mathrm{cap}}$
compression-capacity and rate-distortion policy, and $\Pi_{\mathrm{compute}}$
dual compute-allocation policy. Every semantic atom $a_j$, rendered span
$y_j$, action $a$, memory write $m$, and declassification $d$ is entered in a
ledger

$$
\begin{aligned}
\Lambda = (\,& id, type, mode, scope, ctx, sem, parents, entail, src, rebut, real, auth, prov, split, gen, dep, shift, verifiers, \\
& cert, cost, \mathbf{r}, taint, frontier, time, hash\,).
\end{aligned} \tag{3}
$$

Unledgered state may rank proposals but cannot be rendered as fact or used to
authorize side effects.

A contract is admissible only when the evidence economy is favorable. Define
the online certifiability vector

$$
\Xi_\Gamma(T) = \begin{pmatrix}
\hat c_v / (\widehat{U}_\Gamma + \epsilon),\ \hat p_v^+,\ \hat \eta_v,\ \hat \epsilon_{dep},\ \hat d_{shift},\ \hat \epsilon_{support},\ \hat \rho_{reuse},\ \hat b_{mech} \\
\widehat{H}_\Gamma,\ \hat \epsilon_{sub},\ \hat \epsilon_{sem},\ \hat \epsilon_{src},\ \hat \epsilon_{ctx},\ \hat \epsilon_{rebut},\ \hat \epsilon_{real},\ \widehat{\Delta\mathrm{Front}},\ \hat{\mathbf{r}}_{cum}
\end{pmatrix}, \tag{4}
$$

where $\hat c_v$ is terminal verification cost, $\hat p_v^+$ the usable
false-pass upper bound, $\hat \eta_v$ independent evidence weight,
$\hat \epsilon_{dep}$ unresidualized dependence among verifiers, retrievers,
generators, and learned checkers, $\hat d_{shift}$ calibrated
distribution-shift distance, $\hat \epsilon_{support}$ lost-support bound,
$\hat \rho_{reuse}$ expected mechanism reuse, $\hat b_{mech}$ residual
branching after known macros, $\widehat{H}_\Gamma$ posterior entropy of user
intent/contract, $\hat \epsilon_{sub}$ probability that the neural substrate
omits all certifying traces, $\hat \epsilon_{sem}$ semantic-contractization
loss, $\hat \epsilon_{src}$ source/evidence-recall loss for factual atoms,
$\hat \epsilon_{ctx}$ context/reference/vagueness loss, $\hat \epsilon_{rebut}$
material contradiction-recall loss, $\hat \epsilon_{real}$ atom-to-rendered-text
round-trip loss, $\widehat{\Delta\mathrm{Front}}$ the lower confidence movement
of the compression/intelligence frontier, and
$\hat{\mathbf{r}}_{cum} \in \mathbb{R}^{J_R}_+$ cumulative residual risk by
channel. The hard domain predicate is

$$
\begin{aligned}
\mathrm{DomOK}(T,\Gamma) = \mathbf{1}\{\,
& \hat c_v \leq \kappa_c(\widehat U_\Gamma + \epsilon) \,\wedge\, \hat p_v^+ \leq p_{max} \,\wedge\, \hat \eta_v \geq \eta_{min} \,\wedge\, \hat \epsilon_{dep} \leq \epsilon_{dep,max} \\
& \wedge\, \hat d_{shift} \leq d_{shift,max} \,\wedge\, \hat \epsilon_{support} \leq \epsilon_\Gamma \,\wedge\, \hat \rho_{reuse} \geq \rho_{min} \,\wedge\, \hat b_{mech} \leq b_{max} \\
& \wedge\, \hat \epsilon_{sub} \leq \epsilon_{sub,max} \,\wedge\, \hat \epsilon_{sem} \leq \epsilon_{sem,max} \,\wedge\, \hat \epsilon_{src} \leq \epsilon_{src,max} \,\wedge\, \hat \epsilon_{ctx} \leq \epsilon_{ctx,max} \\
& \wedge\, \hat \epsilon_{rebut} \leq \epsilon_{rebut,max} \,\wedge\, \hat \epsilon_{real} \leq \epsilon_{real,max} \,\wedge\, \widehat{\Delta\mathrm{Front}} \geq -\epsilon_{front} \,\wedge\, \hat{\mathbf{r}}_{cum} \preceq \mathbf{r}_{max}\,\}, \\
\mathrm{IntentOK}(T,\Gamma) &= \mathbf{1}\{\widehat H_\Gamma(T) \leq h_\Gamma\}.
\end{aligned} \tag{5}
$$

Routing is part of inference rather than a post-hoc caveat. The cases are
priority-ordered, so hard certificate failure cannot be masked by grounding or
decomposition:

$$
\mathrm{Route}(T,\Gamma) = \begin{cases}
\text{Abstain}, & \hat p_v^+ \vee \hat \eta_v = 0 \vee \hat \epsilon_{dep} = 1 \vee \hat c_v \geq \kappa_c(\widehat U_\Gamma + \epsilon) \vee \hat \epsilon_{sem} = 1 \vee \hat \epsilon_{ctx} = 1 \vee \hat \epsilon_{real} = 1, \\
\text{Ask}, & (\mathrm{IntentOK} = 0 \vee \hat \epsilon_{sem,max} \vee \hat \epsilon_{ctx} \geq \epsilon_{ctx,max}) \wedge C_{ask} \leq B_{ask}, \\
\text{narrow}, & \hat \epsilon_{support} \geq \epsilon_\Gamma \vee \hat \epsilon_{sub} \geq \epsilon_{sub,max} \vee \hat \epsilon_{dep} \geq \epsilon_{dep,max} \vee \hat \epsilon_{real} \geq \epsilon_{real,max} \vee \neg(\hat{\mathbf{r}}_{cum} \preceq \mathbf{r}_{max}), \\
\text{ground}, & (\hat \epsilon_{src} \geq \epsilon_{src,max} \vee \hat \epsilon_{rebut} \geq \epsilon_{rebut,max}) \wedge C_{src} \leq B_{src,c}, \\
\text{solve}, & \mathrm{DomOK}(T,\Gamma) = 1 \wedge \mathrm{IntentOK}(T,\Gamma) = 1, \\
\text{decompose}, & \exists\{T_i,\Gamma_i\}_{i \leq k} : \prod_i \mathrm{DomOK}(T_i,\Gamma_i) = 1 \wedge C_{glue} \leq B_{glue}, \\
\text{archive}, & \hat \rho_{reuse} < \rho_{min} \wedge \widehat{\Delta\mathrm{Front}} \leq 0 \wedge \widehat U_\Gamma \leq C_{cert}, \\
\text{Abstain}, & \text{otherwise}.
\end{cases} \tag{6}
$$

The primary operating region is therefore
$\mathcal{P}_{cert} = \{\mathcal{P} : \Pr_{T \sim \mathcal{P}}(\mathrm{Route}(T,\Gamma) \in \{\text{solve},\text{decompose},\text{ground}\})$
is high$\}$; broad language, aesthetics, social judgment, and novelty remain
admissible only as a typed mixture of certified atoms, scoped assumptions,
explicitly marked uncertainty, questions, refusals, or nonclaim style.

**Proposition 1** *(Verifier, support, and meaning collapse).* If all
terminal verifiers for a claim have $\hat \eta_v = 0$ or $\hat p_v^+ = 1$, or
if unresolved dependence has $\hat \epsilon_{dep} = 1$, or if
$\hat \epsilon_{support} = 1$, $\hat \epsilon_{sem} = 1$,
$\hat \epsilon_{ctx} = 1$, $\hat \epsilon_{real} = 1$, or
$\hat \epsilon_{src} = 1$ for a factual atom requiring external grounding, or
$\hat \epsilon_{rebut} = 1$ for an open-world atom requiring contradiction
search, or if any mandatory risk coordinate has exhausted its budget, then no
positive scoped certificate can be obtained for that atom under the gate.

*Proof.* For $\hat \eta_v = 0$, $E_{v,r} = 1$. For $\hat p_v^+ = 1$ and a
pass, $E_{v,r} \leq 1$; a failure also gives $E_{v,r} \leq 1$. Hence $M_r$
cannot exceed its family-spending threshold. Unit support, dependency,
substrate omission, semantic loss, context loss, source-recall loss,
rebuttal-recall loss, or realization loss sets the corresponding cap
$[-\log(1+\epsilon_0)]_+ = 0$; an exhausted mandatory risk coordinate sets
$\mathrm{Cert}_{risk} = 0$. The scoped certificate is a minimum over
observation, entailment, source, rebuttal, support, substrate, context,
semantic, realization, domain, privacy, and authority caps, and the gate is
conjunctive. $\square$

Certified utility is measured on the whole transcript, not merely on final
text:

$$
\begin{aligned}
S_{ver}(f; \mathcal{P}, B) = \mathbb{E}_{T \sim \mathcal{P}}\Big[\,
& U_\Gamma(T,f)\mathbf{1}\{\mathrm{Cert}_T(f) \geq V_\Gamma\}\mathbf{1}\{\mathrm{Gate}(f,T,\Gamma) = 1\} \\
& -\,\lambda_F F - \lambda_L L - \lambda_M |\mathcal{A}_{act}| - \lambda_H H_{bits} - \lambda_C C_{hidden} - \lambda_R \mathrm{Risk} - \lambda_U U_{unsupported}\,\Big].
\end{aligned} \tag{7}
$$

A comparison is valid only under matched training data, wall-clock, FLOPs,
memory growth, tool authority, search budget, verifier budget, human bits, and
transcript logging. Unreported test-time search, external inference, or latent
human feedback sets $C_{hidden} = \infty$ for the comparison.

**Invariant 1** *(Proposal, evidence, authority, and compression are
separate).* Retrieved text, popularity, neural confidence, tool output,
imitation traces, macro frequency, and search success may propose hypotheses.
They cannot raise $\mathrm{Cert}$, grant authority, enter active memory, or
declassify data except through the verifier, gate, and compression rules
below.

**Invariant 2** *(Claim-family validity).* A certificate is not a scalar
confidence score. For every adaptive family of candidate claims, actions,
memory writes, and releases, the ledger stores a spending measure $\alpha_j$
with $\sum_j \alpha_j \leq \alpha_\Gamma$. Rendering or action thresholds are
applied to excess log-evidence after this family charge. A mined claim whose
selection used the audit split has certificate weight zero until replayed on a
disjoint split.

**Invariant 3** *(Support conservation).* A pruning, canonicalization,
retrieval, sparse-attention, semantic parser, source-ranking, rebuttal-ranking,
renderer, or retrieval-ranking step may discard alternatives only with a
ledgered, recall-calibrated upper bound on lost posterior mass, context
commitments, source coverage, material contradictions, semantic
interpretations, atom realizations, or certified utility. Otherwise it is a
proposal heuristic; the final answer must ask, abstain, or re-run an exact
bounded stage.

**Invariant 4** *(Semantic faithfulness).* A generated sentence is not
evidence about its own meaning. Every factual, mathematical, executable,
private, or action-guiding span is first reduced to semantic atoms and then
checked by scoped entailment from ledgered parents. Style, empathy, refusal,
and creative spans may be uncertified only when the renderer proves that they
contain no gated atom and inherit no private or capability-bearing taint.

**Invariant 5** *(Certifiability before ambition).* An answer is active only
on a certified or decomposable contract. If cheap independent tests do not
exist, user intent is underdetermined, context or vague predicates are
unresolved, semantic interpretations are not separated, source or rebuttal
recall is unbounded, atom-to-text realization is unbounded, verifier cost
exceeds expected utility, support recall is unbounded, cumulative risk is
exhausted, or the base substrate cannot recall a certifying trace family, the
architecture must reduce the task, ground, ask, narrow, or abstain; dense
language fluency is not evidence.

**Invariant 6** *(Vector risk and dependence before aggregation).* Risk and
evidence are never averaged before gating. Let $\mathcal{J}_R$ index impact,
privacy, exfiltration, capability, influence, conflict, model, and dependence
risk. Every claim, action, memory write, release, macro admission, retrieval
step, and renderer pass carries
$\mathbf{r} \in \mathbb{R}^{J_R}_+$ and a dependence class $\mathrm{Dep}$. A
scalar utility may rank proposals only after the componentwise inequalities
$\mathbf{r} \preceq \mathbf{B}_R$ and the dependence policy
$\Pi_{\mathrm{dep}}$ hold; otherwise the state is narrowed, split,
residualized, or abstained.

---

> **Implementation pointers.**
>
> - Ledger $\Lambda$, the certifiability vector $\Xi_\Gamma(T)$, the
>   `DomOK`/`IntentOK` predicates, and the `Route` decision rule live in
>   `crates/vpm-ledger` and `crates/vpm-core`.
> - The contract structure $\Gamma$ and its policies $\Pi_*$ are typed in
>   `crates/vpm-core` and re-exported to Python via `vpm._native`.
> - Invariants 1–6 are enforced as **runtime checks** in `crates/vpm-verify`
>   and as **property tests** under `tests/integration/invariants/`.
> - The vector-risk lattice $\mathbf{r} \in \mathbb{R}^{J_R}_+$ and dependence
>   class `Dep` are part of `crates/vpm-authority`.
