# §7 Compression, memory, and capacity limits

Memory is two-tier:

$$
\mathcal{A}_{act} \subset \mathcal{A}_{arc}, \quad \zeta = ([\mu], \sigma, \chi_\mu, \nu, \text{scope, failures, transfer},\, L_\mathcal{D}, C_{exec}, \text{age, auth, drift, ctx, sem, src, rebut, real, mode}). \tag{118}
$$

Only $\mathcal{A}_{act}$ participates in low-latency inference; archival items
are retrieval evidence, replay material, and sleep-phase candidates. The
library objective is

$$
\begin{aligned}
\min_{\mathcal{D}, \mathcal{A}}\, & K(\mathcal{D}) + K(\mathcal{A}) + \sum_i L_{\mathcal{D}, \mathcal{A}}(\tau_i \mid \Gamma_i) + \lambda_{reg}\mathrm{Regress} + \lambda_{maint}\mathrm{Maint} \\
& + \lambda_{ent} H(\mathcal{A}_{act}) + \lambda_{auth}\mathrm{CapRisk} + \lambda_{ctx}\mathrm{CtxErr} + \lambda_{sem}\mathrm{SemErr} + \lambda_{src}\mathrm{SrcMiss} + \lambda_{rebut}\mathrm{RebutMiss} + \lambda_{real}\mathrm{RealErr}.
\end{aligned} \tag{119}
$$

The compressed library is hierarchical:

$$
\mathcal{Z}_t = (\mathcal{D}_t^{prim},\, \mathcal{D}_t^{local},\, \mathcal{D}_t^{global},\, \mathcal{D}_t^{ctx},\, \mathcal{D}_t^{sem},\, \mathcal{D}_t^{src},\, \mathcal{D}_t^{safe},\, \mathcal{R}_t^{resid}), \tag{120}
$$

where active macros encode reusable executable structure,
context/semantic/source/safety dictionaries encode reusable interfaces and
obligations, and $\mathcal{R}^{resid}$ stores incompressible or unstable
residuals with explicit distortion labels. A proposed macro is admitted at
the lowest level whose code length plus dispatch entropy improves the
held-out rate-distortion frontier:

$$
\begin{aligned}
\Delta\mathrm{Front}_t(m) = \mathrm{LCB}_t^{seq}\!\Big[\,\Delta L + \Delta C_{exec} + \Delta S_{ver} - K(m) - H(\mathrm{dispatch}_m \mid \mathrm{scope})\Big] - \lambda_I \Delta I_m - \lambda_R\lambda_R^\top \Delta \mathbf{r}_m - \lambda_{dep}\Delta \epsilon_{dep, m},
\end{aligned} \tag{121}
$$

$$
\Delta I_m = \sum_{m' \in \mathcal{A}_{act}} |\mathrm{Cov}(g(m), g(m'))| + \Pr\{m\ \text{shadows a stricter verified primitive}\}. \tag{122}
$$

The interference charge prevents the active library from filling with
near-duplicate shortcuts that reduce local description length while
increasing retrieval entropy, dependency, maintenance, or unsafe capability
concentration. For a candidate macro/capsule $m$ and replay draw
$T_i \sim \mathcal{R}_t$, define bounded gain $g_i(m) \in [-G, G]$:

$$
\begin{aligned}
g_i(m) = \,& \Delta L_i + \Delta \mathrm{Solve}_i + \Delta \mathrm{Latency}_i + \Delta \mathrm{Reliability}_i + \Delta \mathrm{SampleEff}_i + \Delta \mathrm{Transfer}_i \\
& - K(m) - \eta\Delta \mathrm{Branch}_i - \xi \Delta \mathrm{Ambig}_i - \rho \Delta \mathrm{Risk}_i - \kappa \Delta \mathrm{CalErr}_i \\
& - \nu \Delta \mathrm{Regress}_i - \mu \Delta \mathrm{Maint}_i - \lambda_{fp} \Delta \Phi_i - \lambda_{auth} \Delta \mathrm{CapRisk}_i \\
& - \lambda_{ctx} \Delta \mathrm{CtxErr}_i - \lambda_{sem} \Delta \mathrm{SemErr}_i - \lambda_{src} \Delta \mathrm{SrcMiss}_i - \lambda_{rebut} \Delta \mathrm{RebutMiss}_i - \lambda_{real} \Delta \mathrm{RealErr}_i - \lambda_{mode} \Delta \mathrm{ModeConf}_i.
\end{aligned} \tag{123}
$$

A macro is not a compressed capability unless it has an expansion witness
$W_m$ and a scoped equivalence certificate

$$
\mathrm{Cert}_{eq}(m) = \inf_{x \in \Omega_m} \mathrm{Cert}\!\big(\mathrm{Exec}(m, x) = \mathrm{Exec}(W_m, x) \wedge \Lambda(\mathrm{Exec}(m, x)) \equiv_\Gamma \Lambda(\mathrm{Exec}(W_m, x))\big). \tag{124}
$$

Opaque residual shortcuts may rank proposals but are not active macros.

Replay samples and macro candidates are selected adaptively, so admission
uses a sequence-valid, multiplicity-corrected radius. With $N_t$ candidates
tested up to time $t$,

$$
\widehat A_t(m) = \frac{1}{n_t(m)} \sum_{i=1}^{n_t(m)} g_i(m), \quad \widehat\sigma_t^2(m) = \frac{1}{n_t(m) - 1} \sum_i (g_i(m) - \widehat A_t(m))^2, \tag{125}
$$

$$
\mathrm{Rad}_t(m) = \sqrt{\frac{2\,\widehat \sigma_t^2(m)\,\log(3N_t/\delta)}{n_t(m)}} + \frac{3G\,\log(3N_t/\delta)}{n_t(m)} + D_{adapt}(m), \tag{126}
$$

$$
\mathrm{LCB}_t^{seq} A(m) = \widehat A_t(m) - \mathrm{Rad}_t(m) - D_{drift}(m) - D_{leak}(m) - D_{select}(m), \tag{127}
$$

$$
\mathrm{UCB}_t^{seq} A(m) = \widehat A_t(m) + \mathrm{Rad}_t(m) + D_{drift}(m) + D_{leak}(m) + D_{select}(m). \tag{128}
$$

Admission is a decision rule, not a learned preference:

$$
\begin{aligned}
\mathrm{Admit}_{act}(m) = \mathbf{1}\Big\{\,
& \mathrm{LCB}_t^{seq} A(m) > 0 \wedge \mathrm{Cert}_{act} \wedge \mathrm{Cert}_{eq}(m) \geq \theta_{eq} \wedge \mathrm{NoCapEsc}(m) \\
& \wedge\, \mathrm{ReplayPass}(m) \wedge H(\mathrm{Args}_m \mid \mathrm{scope}) \leq h_m \wedge \mathrm{DecodeCost}(m) \leq B_{dec} \\
& \wedge\, \epsilon_{ctx}(m) \leq \epsilon_{ctx,max} \wedge \epsilon_{sem}(m) \leq \epsilon_{sem,max} \wedge \epsilon_{src}(m) \leq \epsilon_{src,max} \wedge \epsilon_{rebut}(m) \leq \epsilon_{rebut,max} \wedge \epsilon_{real}(m) \leq \epsilon_{real,max} \\
& \wedge\, \epsilon_{dep}(m) \leq \epsilon_{dep,max} \wedge \Delta\mathrm{Front}_t(m) > 0 \wedge \Delta \mathbf{r} \preceq \mathbf{B}_R - \mathbf{r}_t\,\Big\}.
\end{aligned} \tag{129}
$$

Demotion is symmetric:

$$
\begin{aligned}
\mathrm{Demote}(m) = \mathbf{1}\Big\{\,
& \mathrm{UCB}_t^{seq} A(m) < 0 \vee \mathrm{Regress}(m) \vee \mathrm{Stale}(m) \vee \mathrm{BranchExplosion}(m) \\
& \vee\, \mathrm{ScopeMismatch}(m) \vee \mathrm{CapEsc}(m) \vee \mathrm{CtxShift}(m) \vee \mathrm{SemShift}(m) \\
& \vee\, \mathrm{SrcStale}(m) \vee \mathrm{RebutStale}(m) \vee \mathrm{RealDrift}(m) \vee \mathrm{DepShift}(m) \vee \Delta\mathrm{Front}_t(m) < 0\,\Big\}.
\end{aligned} \tag{130}
$$

The active set is a constrained selection problem:

$$
\mathcal{A}_{act} = \arg\max_{S \subset \mathcal{A}_{arc}}\, \sum_{m \in S} \mathrm{LCB}_t^{seq} A(m) - \lambda_{pair}\!\!\sum_{m \neq m'}\!\mathrm{Overlap}(m, m') - \lambda_{ret} H(R_S) - \lambda_{mode}\mathrm{ModeConf}(S) \tag{131}
$$

subject to $C_{lat}(S) \leq B_{lat}$, $C_{mem}(S) \leq B_{mem}$,
$\mathrm{CapRisk}(S) \leq B_{cap}$,
$\epsilon_{ctx}(S) \leq \epsilon_{ctx,max}$,
$\epsilon_{sem}(S) \leq \epsilon_{sem,max}$,
$\epsilon_{src}(S) \leq \epsilon_{src,max}$,
$\epsilon_{rebut}(S) \leq \epsilon_{rebut,max}$,
$\epsilon_{real}(S) \leq \epsilon_{real,max}$,
$\epsilon_{dep}(S) \leq \epsilon_{dep,max}$,
$\Delta\mathbf{r} \preceq \mathbf{B}_R - \mathbf{r}_t$, and
$\sum_{m \in S} \Delta I_m \leq I_{max}$. Retrieval is approximate but
certification is exact; approximate nearest neighbors can miss a useful
macro but cannot certify an unsafe or unsupported one.

**Proposition 4** *(Macro search gain).* If a raw search has branching $b$
and depth $L$, while macro $m$ changes depth to $L'$ and branching to
$\gamma b$, then search improves only if

$$
b^{L - L'} > \gamma^{L'}. \tag{132}
$$

With pruning factors $p_d$, replace $b$ at depth $d$ with $bp_d$. Thus
syntax compression is insufficient; admission must charge branch, ambiguity,
replay, maintenance, and false-pass risk.

**Proposition 5** *(Sequence-valid macro admission prevents expected
degradation).* Assume $g_i(m) \in [-G, G]$, the replay distribution covers
the deployment scope up to the drift penalty, and the candidate count is
charged by $N_t$. Uniformly over adaptive macro proposals and stopping
times, with probability at least $1 - \delta$, every admitted macro has
nonnegative amortized gain on its certified scope and is extensionally
equivalent to its expansion witness.

*Proof.* The empirical-Bernstein confidence sequence gives
$A_t(m) \geq \mathrm{LCB}_t^{seq} A(m)$ simultaneously over candidates and
stopping times after drift, leakage, and selection charges. Admission
requires this lower bound, the equivalence certificate, replay, scope, and
capability checks to pass. $\square$

The architecture cannot compress arbitrary additional data. Let $D_{1:n}$ be
tasks, $U_{1:n}$ their language interfaces, $A_{1:n}^{sem}$ their semantic
atoms, $S_{1:n}^{src}$ their source states, $R_{1:n}^{style}$ their residual
style, and $\mathcal{Z}$ the learned library. For any code satisfying
contract error $\epsilon$,

$$
\mathbb{E} L(D_{1:n} \mid \mathcal{Z}, \Gamma, V) \geq H(D_{1:n} \mid \mathcal{Z}, \Gamma, V) - O(\epsilon n), \tag{133}
$$

$$
\begin{aligned}
\mathbb{E} L(U_{1:n} \mid \mathcal{Z}, \Gamma, V) \geq \,& H(C_{1:n}^{ctx}, A_{1:n}^{sem}, S_{1:n}^{src}, S_{1:n}^{rebut}, R_{1:n}^{real} \mid \mathcal{Z}, \Gamma, V) \\
& + H(R_{1:n}^{style} \mid C_{1:n}^{ctx}, A_{1:n}^{sem}, S_{1:n}^{src}, S_{1:n}^{rebut}, R_{1:n}^{real}, \mathcal{Z}) - O(\epsilon n).
\end{aligned} \tag{134}
$$

The operational compression frontier is the constrained rate-distortion
value

$$
\mathcal{R}_t(D, B) = \min_{\mathcal{D}, \mathcal{A}}\, K(\mathcal{D}) + K(\mathcal{A}) + \mathbb{E}[L_{\mathcal{D}, \mathcal{A}}(T)]\quad \text{s.t.}\ \ \mathbb{E}[d_T(T, \tilde T)] \leq D,\ C_{lat} \leq B_{lat},\ C_{mem} \leq B_{mem},\ \mathbf{r} \preceq \mathbf{B}_R,\ \epsilon_{dep} \leq \epsilon_{dep,max}. \tag{135}
$$

A library expansion is useful only if it moves this frontier downward on a
held-out scope; memorizing exceptions moves archival storage, not active
compression.

**Proposition 6** *(Frontier-valid compression).* Assume candidate macros
are evaluated on split-clean replay with sequence-valid lower bounds, exact
expansion witnesses, and componentwise risk/dependence gates. If every
active admission satisfies $\Delta\mathrm{Front}_t(m) > 0$ and every demoted
macro has $\mathrm{UCB}_t^{seq} \Delta\mathrm{Front}_t(m) < 0$ or a gate
violation, then active-memory growth is bounded by the number of positive
frontier moves plus the covering map of certified scopes. Additional data
that have conditional entropy larger than the best admitted code improvement
remain residual and cannot be compressed without increasing distortion or
violating a gate.

*Proof.* Each admitted macro decreases the held-out rate-distortion LCB
after dispatch, interference, maintenance, dependence, and vector-risk
charges. A finite memory budget and nonnegative code lengths forbid
infinitely many positive decreases above the confidence radius in the same
certified scope. If the conditional entropy of new tasks is not reduced by
any admissible code, the source-coding lower bound applies to
$\mathcal{R}^{resid}$; storing it in $\mathcal{A}_{act}$ would increase
$K(\mathcal{A})$ or dispatch entropy without positive frontier movement.
$\square$

Compression saturates when

$$
\sup_m \mathrm{LCB}_t^{seq} A(m) \leq 0, \tag{136}
$$

which occurs under non-reusable structure, unstable grounding, source
churn, contradiction churn, high context entropy, high semantic entropy,
high realization error, high $H(Y \mid M, C, \Gamma)$, nondiscriminating
tests, non-composable tasks, verifier cost above value, active-dictionary
entropy, branch explosion, mode confusion, influence risk, or safety scope
mismatch. The repair is not larger active memory alone; it is a better
primitive, stronger test, narrower scope, context normalization, semantic
factorization, source/rebuttal refresh, realization checking, hierarchical
factorization, context-conditioned archival storage, lossy rate-distortion
with explicit error, or abstention.

---

> **Implementation pointers.**
>
> - Two-tier memory $\mathcal{A}_{act} \subset \mathcal{A}_{arc}$ (118) →
>   `python/vpm/memory/{active,archival}.py`; capsule schema $\zeta$ in
>   `vpm-core::Capsule`.
> - Hierarchical library $\mathcal{Z}_t$ (120) → `vpm.memory.library`.
> - Frontier movement $\Delta\mathrm{Front}_t(m)$ (121–122),
>   empirical-Bernstein bounds $\mathrm{LCB}_t^{seq}/\mathrm{UCB}_t^{seq}$
>   (125–128) → `vpm.memory.frontier` plus `crates/vpm-verify::eb_seq`.
> - Equivalence certificate $\mathrm{Cert}_{eq}(m)$ (124) →
>   `crates/vpm-egraph::equiv_cert` (uses `egg` saturation against the
>   expansion witness).
> - Admission / demotion (129–130) → `vpm.memory.admit`.
> - Active-set selection problem (131) → `vpm.memory.active_set` (LP /
>   greedy with budget constraints).
> - Saturation diagnostics (136) → `vpm.evaluation.saturation`.
