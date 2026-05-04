# Glossary

Symbols, types, and abbreviations used throughout the architecture.
Section numbers refer to [docs/architecture/](./architecture/).

## Top-level objects

| Symbol | Name | Section |
|---|---|---|
| $T$ | Task: $(O_{0:t}, A, B, R, H, \mathcal{F}_t)$ | §1 |
| $\Gamma$ | Contract — output type, success predicate, evidence threshold, policies $\Pi_*$ | §1 |
| $\Lambda$ | Append-only ledger | §1 |
| $\mu$ | Mechanism: $(\mathcal{S}, \mathcal{T}, \mathcal{O}, \Pi, U, \chi, \nu, \alpha, \ell, \omega, r, \lambda)$ | §2 |
| $[\mu]_\Gamma$ | Mechanism equivalence class under $\sim_\Gamma$ | §2 |
| $Z_t$ | Recurrent inference state | §2 |
| $\mathcal{D}_t$ | Active dictionary of capsules / macros | §2, §7 |
| $\mathcal{M}_t(d, w, \ell)$ | Certified support — admissible mechanism closure | §2 |
| $H_t^\theta$ | Substrate state of $\mathrm{VNS}_\theta$ | §3 |
| $\mathrm{VNS}_\theta$ | Verifier-Native Substrate (proposal/prediction model) | §3 |
| $\mathfrak{n}$ | Conversation normal form | §4 |
| $\mathcal{Z}_t$ | Hierarchical compressed library | §7 |
| $\mathcal{A}_{act} \subset \mathcal{A}_{arc}$ | Active vs archival memory | §7 |

## Routes and gates

| Symbol | Meaning | Section |
|---|---|---|
| $\mathrm{Route}(T, \Gamma)$ | $\{\text{Abstain}, \text{Ask}, \text{narrow}, \text{ground}, \text{solve}, \text{decompose}, \text{archive}\}$ | §1 |
| $\mathrm{DomOK}, \mathrm{IntentOK}$ | Hard domain / intent predicates | §1 |
| $\mathrm{CtxOK}, \mathrm{SemOK}$ | Context / semantic gates | §4 |
| $\mathrm{EntOK}, \mathrm{CiteOK}, \mathrm{RebutOK}, \mathrm{RealOK}$ | NL atom-level gates | §4 |
| $\mathrm{Gate}_{NL}$ | Final natural-language gate on rendered text | §4 |
| $\mathrm{Gate}(a, Z, \Gamma)$ | Action gate (auth, perm, risk, sandbox, …) | §6 |
| $\mathrm{Gate}_{render}$ | Renderer gate | §6 |

## Calibrated error / loss bounds

| Symbol | Meaning | Section |
|---|---|---|
| $\hat \epsilon_{support}$ | Lost-support bound | §1, §5 |
| $\hat \epsilon_{sub}, \epsilon_{crit}$ | Substrate omission, critical-edge omission | §3 |
| $\hat \epsilon_{ctx}, \hat \epsilon_{sem}$ | Context, semantic loss | §4 |
| $\hat \epsilon_{src}, \hat \epsilon_{rebut}$ | Source-recall, rebuttal-recall loss | §4 |
| $\hat \epsilon_{real}$ | Atom-to-rendered-text round-trip loss | §4 |
| $\hat \epsilon_{dep}$ | Unresidualized dependence | §2 |
| $\hat d_{shift}$ | Calibrated distribution-shift distance | §1 |
| $\widehat H_\Gamma$ | Posterior entropy of user intent | §1 |
| $\widehat{\Delta\mathrm{Front}}$ | Lower-confidence frontier movement | §1, §7 |
| $\hat{\mathbf{r}}_{cum}$ | Cumulative residual vector risk | §1, §6 |

## Verifier machinery

| Symbol | Meaning | Section |
|---|---|---|
| $v$ | Verifier $(\mathcal{X}_v, c_v, n_v, f_v, \rho_v, \psi_v, \kappa_v, \text{splits})$ | §2 |
| $X_{v, r}, p_{v, r}^+$ | Pass/fail indicator and false-pass UCB | §2 |
| $E_{v, r}, E_{\mathrm{Dep}, r}$ | E-value, dependence-block e-value | §2 |
| $M_r(h)$ | Claim martingale | §2 |
| $\mathrm{Cert}(h, T)$ | Scoped certificate (minimum over caps) | §2 |
| $V_\Gamma, V_a, \theta_k$ | Evidence thresholds | §1, §2, §4 |
| $\alpha_h, \alpha_\Gamma$ | Per-claim and total family-spending budgets | §2 |
| $\eta_{v, r}$ | Effective verifier weight | §2 |
| $\delta_{cal}$ | Total calibration-event budget | §2 |
| $\mathrm{LCB}^{seq}, \mathrm{UCB}^{seq}$ | Anytime sequence-valid lower / upper bounds | §7 |

## Authority and risk

| Symbol | Meaning | Section |
|---|---|---|
| $\mathcal{L}_{auth}$ | Authority lattice (data $\preceq$ instruction $\preceq$ capability …) | §6 |
| $\alpha(\cdot)$ | Authority-label join | §6 |
| $\lambda(\cdot)$ | Data/authority taint | §6 |
| $\mathbf{r} \in \mathbb{R}^{J_R}_+$ | Vector risk: impact, privacy, exfil, capability, influence, conflict, model, dep | §1, §6 |
| $\mathbf{B}_R, \mathbf{b}_R$ | Componentwise risk budget | §6 |
| $\mathrm{Cred}(a)$ | Per-action rollback credit | §6 |
| $\mathrm{Dec}_\ell$ | Declassification predicate | §6 |
| $\Pi_{perm}, \Pi_{priv}, \Pi_{infl}, \Pi_{dep}, \Pi_{cap}, \ldots$ | Contract policies | §1 |

## Compression / memory

| Symbol | Meaning | Section |
|---|---|---|
| $W_m$ | Expansion witness for macro $m$ | §7 |
| $\mathrm{Cert}_{eq}(m)$ | Scoped equivalence certificate | §7 |
| $g_i(m)$ | Bounded admission gain | §7 |
| $\widehat A_t(m), \mathrm{Rad}_t(m)$ | Empirical-Bernstein mean and radius | §7 |
| $\Delta\mathrm{Front}_t(m)$ | Held-out rate-distortion frontier movement | §7 |
| $\Delta I_m$ | Interference charge between active macros | §7 |
| $\mathrm{Admit}_{act}, \mathrm{Demote}$ | Memory admission/demotion rules | §7 |
| $\mathcal{R}^{resid}$ | Incompressible residual store | §7 |
| $\mathcal{R}_t(D, B)$ | Constrained operational rate-distortion frontier | §7 |

## Modes (NL)

- $\mathcal{M}_{cert}$ = {fact, math, code, private, action, quote, advice, definition} — every atom in this partition requires entailment, source, rebuttal, authority, privacy, context, support, substrate, family-spending, and realization certificates.
- $\mathcal{M}_{soft}$ = {style, creative, preference, uncertainty, question, refusal, social, transition} — requires mode separation, influence safety, privacy, no hidden certified-mode atom; never becomes a factual claim by fluent wording.

## Curriculum gates

- $C_0$: deterministic grids, strings, FSMs, small graphs, arithmetic, finite-state tasks, data transformations.
- $C_1$: hidden-schema splits, equality saturation, typed program synthesis, theorem-proving fragments.
- $C_2$: noisy/partial observations, active tests, small causal worlds, verifiable planning.
- $C_3$: formal tools, agent workflows, tool use, authority, declassification, prompt-injection, rollback, influence-risk, conflict.
- $C_4$: controlled dialogue/artifacts with context normal forms, semantic atom extraction, source-grounded QA, contradiction search, entailment checking, round-trip realization checking, calibrated uncertainty, intent-entropy gates.
- $C_5$: continual compression with replay-safe macro admission.

## Abbreviations

| | |
|---|---|
| BCE | Binary cross-entropy |
| CE | Cross-entropy |
| CNF | Conversation normal form (§4) |
| ECE | Expected Calibration Error |
| EVC | Expected value of computation (§5, eq. 101) |
| LCB / UCB | Lower / upper confidence bound (sequence-valid in §7) |
| NL | Natural language (§4) |
| PSR | Predictive State Representations (§3, ref [11]) |
| RT | Round-trip realization (§4) |
| SSM | Selective state-space model (Mamba, ref [4]) |
| VNS | Verifier-Native Substrate (§3) |
| VPM | Verifier-Native Compressive Predictive Mechanisms (this document) |
