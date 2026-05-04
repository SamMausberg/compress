# §2 Typed mechanisms and calibrated evidence

**Definition 1** *(Mechanism).* A mechanism is

$$
\mu = (\mathcal{S}, \mathcal{T}, \mathcal{O}, \Pi, U, \chi, \nu, \alpha, \ell, \omega, r, \lambda), \tag{8}
$$

where $\mathcal{S}$ is typed latent state, $\mathcal{T}$ transition,
$\mathcal{O}$ observation/render map, $\Pi$ executable procedure or policy,
$U$ contract utility, $\chi$ verifier obligations, $\nu$ verifier obligations,
$\alpha$ authority requirements, $\ell$ code length, $\omega$ scope/fragility
metadata, $r$ replay statistics, and $\lambda$ data/authority taint.
Equivalence is contractual, counterfactual, and trace-preserving:

$$
\mu_i \sim_\Gamma \mu_j \iff \begin{cases}
\forall q \in Q_\Gamma : \mathrm{Exec}(\mu_i, q) = \mathrm{Exec}(\mu_j, q), \\
\forall a \in A_\Gamma^{cf} : \mathrm{Exec}(\mathrm{do}(a, \mu_i), q) = \mathrm{Exec}(\mathrm{do}(a, \mu_j), q), \\
\Lambda(\mathrm{Exec}(\mu_i, q)) \equiv_\Gamma \Lambda(\mathrm{Exec}(\mu_j, q)).
\end{cases} \tag{9}
$$

Posterior mass is assigned to $[\mu]_\Gamma$, not syntax.

The recurrent state is

$$
Z_t = (C_t, M_t, S_t, Q_t, V_t, P_t, A_t, G_t, K_t, D_t, E_t^{sem}, C_t^{ctx}, R_t^{src}, R_t^{rebut}, \Omega_t^{real}, h_t), \tag{10}
$$

where $C$ is a scene/contract posterior, $M$ a mechanism posterior, $S$
predictive state, $Q$ active tests, $V$ verifier records, $P$ provenance
graph, $A$ authority labels, $G$ e-graph/canonicalization state, $K$
support-loss certificates, $D$ discourse state, $E^{sem}$ semantic atoms and
entailment obligations, $C^{ctx}$ context/reference commitments, $R^{src}$
source/evidence coverage state, $R^{rebut}$ material-rebuttal coverage state,
$\Omega^{real}$ atom-to-rendering obligations, and $h$ compact neural state.

The active dictionary is

$$
\mathcal{D}_t = \{\,d_k = (\tau_k^{in}, \tau_k^{out}, f_k, \chi_k, \nu_k, p_k, \Omega_k, r_k, \lambda_k, \sigma_k^{sem}, \rho_k^{ctx}, \rho_k^{src}, \rho_k^{rebut}, \rho_k^{real})\,\}_{k=1}^{K_D}. \tag{11}
$$

For depth $d$, width $w$, and description budget $\ell$, define the certified
support

$$
\mathcal{M}_t(d, w, \ell) = \{\mu \in \mathrm{cl}_{\circ, \mathrm{if}, rec, \mathrm{test}, tool, sem, src}(\mathcal{D}_t \cup \mathcal{P}_t^{\mathrm{prim}}) : L_{\mathcal{D}_t}(\mu) \leq \ell,\ d(\mu) \leq d,\ w(\mu) \leq w,\ \nu(\mu) \neq \emptyset\}. \tag{12}
$$

The architecture is intelligent only over this closure. Its supported
certified utility and support gap are

$$
I_t(\mathcal{P}) = \mathbb{E}_{T \sim \mathcal{P}}\left[\max_{\mu \in \mathcal{M}_t(d,w,\ell)} U_T(T, \mu)\,\mathbf{1}\{\mathrm{Cert}(\mu, T) \geq V_\Gamma\}\right], \tag{13}
$$

$$
G_t(\mathcal{P}) = \mathbb{E}_{T \sim \mathcal{P}}\left[\max_{\mu \in \mathcal{M}^*} U_T(T, \mu) - \max_{\mu \in \mathcal{M}_t(d,w,\ell)} U_T(T, \mu)\right]_+, \tag{14}
$$

$$
G_t^{NL}(\mathcal{P}) = \mathbb{E}\Big[\Delta_{ctx}(T) + \Delta_{sem}(T) + \Delta_{src}(T) + \Delta_{rebut}(T) + \Delta_{real}(T) + \Delta_{mode}(T) + \Delta_{ent}(T)\Big]_+. \tag{15}
$$

Thus more parameters do not by themselves increase intelligence; support
expands only by adding primitives, typed compositions, context formalisms,
semantic atoms, entailment tests, source/rebuttal-recall mechanisms,
realization checkers, verifiers, memories, or tool contracts that reduce
$G_t + G_t^{NL}$ under audit.

The active architecture state contains a bottleneck vector rather than a
scalar capability score:

$$
\mathrm{Bott}_t(T) = \big(G_t,\, G_t^{NL},\, \epsilon_{support},\, \epsilon_{sub},\, \epsilon_{ctx},\, \epsilon_{sem},\, \epsilon_{src},\, \epsilon_{rebut},\, \epsilon_{real},\, \epsilon_{dep},\, d_{shift},\, b_{mech},\, H_\Gamma,\, \hat{\mathbf{r}}_{cum},\, \Delta\mathrm{Front}^-\big). \tag{16}
$$

A repair operator $\rho \in \mathcal{R}_{arch}$ is an admissible primitive,
verifier, parser, retriever, memory capsule, causal test, macro, renderer
checker, safety rule, or training-data acquisition. It is applied only when
its held-out marginal certified value is positive under all current dual
prices:

$$
\rho_t^* = \arg\max_{\rho \in \mathcal{R}_{arch}} \frac{\mathrm{LCB}_{1-\delta_\rho}\{\Delta S_{ver}(\rho) - \lambda_B \|\Delta \mathrm{Bott}_t^+(\rho)\|_1 - \lambda_R^\top \Delta \mathbf{r}(\rho) - \lambda_M \Delta C_{maint}\}}{c_{impl}(\rho) + c_{train}(\rho) + c_{cal}(\rho) + c_{exact}(\rho)}, \tag{17}
$$

$$
\mathrm{Apply}(\rho_t^*) = \mathbf{1}\{\mathrm{LCB}_{1-\delta_\rho}(\Delta S_{ver}) > 0 \,\wedge\, \Delta \mathrm{Bott}_t^+ \preceq 0 \,\wedge\, \Delta \mathbf{r} \preceq \mathbf{b}_R - \mathbf{r}_{cum}\}. \tag{18}
$$

This rule changes the architecture at the weakest certified bottleneck: a
compression failure triggers factorization or residual coding, a support
failure triggers exact rehydration or a new primitive, a language failure
triggers context/semantic/source/rebuttal/realization machinery, an
intelligence failure triggers new tests or tools, a safety failure tightens
authority/risk structure, and a training inefficiency triggers compute
reallocation. Parameter scaling is one possible repair only when it has
positive lower-confidence frontier movement after these charges.

Novelty is an expansion operator with witnessed boundary failures:

$$
\mathrm{Expand}(T) = 1 \iff \Pr_{q_t}[S_T = 1 \mid \mathcal{M}_t] < \epsilon_s, \quad \mathrm{LCB}_{1-\delta}(\Delta S_{ver}^{new} - \lambda_c \Delta \mathrm{Cost} - \lambda_R \Delta \mathrm{Risk}) > 0,
$$

$$
\exists\,\mathcal{W}_T : \forall \mu \in \mathcal{M}_t(d, w, \ell)\ \exists\, x \in \mathcal{W}_T\ \mathrm{Cert}(S_T(\mu, x) = 0) \geq \theta_{bdry}. \tag{19}
$$

Without boundary witnesses the system widens search or asks; it does not
infer that the world is impossible merely because the current DSL failed.

An adaptive claim family at round $r$ is

$$
\mathcal{J}_r = \{h_{rj} : h_{rj}\ \text{is}\ \mathcal{F}_{r-1}\text{-measurable}\}, \quad \sum_{r, j} \alpha_{rj} \leq \alpha_\Gamma. \tag{20}
$$

The requirement that $h_{rj}$ is measurable before the certifying sample is
drawn is the anti-p-hacking condition; a hypothesis generated from a test
split cannot be certified by that split.

A verifier is

$$
v = (\mathcal{X}_v, c_v, n_v, f_v, \rho_v, \psi_v, \kappa_v, \mathrm{train}_v, \mathrm{cal}_v, \mathrm{audit}_v, \mathrm{gen}_v), \tag{21}
$$

with domain, cost, calibration trials, observed false passes, dependence
embedding, scope penalty, staleness/adaptivity penalty, training split,
calibration split, audit split, and generator identity. For a false claim
$h$ and an adaptively chosen test $x_T$, let
$X_{v,r}(h, x_T) \in \{0, 1\}$ be pass/fail. The verifier is usable only on
an event $\mathcal{C}_{v,r}$ with
$\Pr(\mathcal{C}_{v,r}) \geq 1 - \delta_{v,r}$ such that

$$
\Pr_0(X_{v,r} = 1 \mid \mathcal{F}_{r-1},\, h\ \text{false},\, \mathcal{C}_{v,r}) \leq p_{v,r}^+(T), \tag{22}
$$

where

$$
p_{v,r}^+(T) = \min\{1,\, \mathrm{UCB}_{1-\delta_{v,r}}^{seq}(p_v; n_v, f_v) + \psi_v(T) + \kappa_v + D_{drift}(v, T) + D_{select}(v, T)\}, \quad \sum_{v, r} \delta_{v,r} \leq \delta_{cal}. \tag{23}
$$

The confidence sequence rather than a fixed-binomial interval is required
because verifier choice, replay frequency, and stopping time are adaptive
[13]. Conditional residual calibration is required for correlated tests;
duplicate passes do not accumulate evidence.

Effective verifier weight is

$$
\eta_{v,r} = \eta_{scope}\,\eta_{stale}\,\eta_{split}\,\eta_{dep}\,\eta_{gen}\,\eta_{audit} \in [0, 1], \tag{24}
$$

where $\eta_{split} = 0$ if any certification event violates
$\Pi_{\mathrm{split}}$, $\eta_{audit} = 0$ if candidate selection inspected
the audit trace, $\eta_{gen} = 0$ if the verifier family generated the
candidate and has no independent replay split, and $\eta_{dep} = 0$ for
duplicate or correlated verifiers without conditional residual evidence. Let
$\mathcal{G}_r$ be the dependence partition induced by shared generator, data
split, retrieval index, training run, calibration source, prompt, tool, and
failure mode. For a block $g \in \mathcal{G}_r$, either a residual pass
variable $X_{g,r}^+$ is calibrated conditionally on all earlier block outputs
or the block contributes only its strongest single calibrated pass:

$$
E_{g,r}(h) = 1 + \eta_{g,r}\!\left(\frac{X_{g,r}^+(h)}{p_{g,r}^+(T)} - 1\right), \quad \eta_{g,r} = 0\ \ \text{if}\ X_{g,r}^+\ \text{is unavailable}, \tag{25}
$$

$$
E_{\mathrm{Dep},r}(h) = \prod_{g \in \mathcal{G}_r} E_{g,r}(h), \quad \epsilon_{dep} = \mathrm{UCB}_{1-\delta_{dep}}^{seq}\Pr\{\exists g : \eta_{g,r} = 0\ \text{but treated as independent}\} + D_{select}^{dep} + D_{shift}^{dep}. \tag{26}
$$

The claim martingale is formed from dependence blocks, not raw verifier
calls:

$$
E_{v,r}(h) = 1 + \eta_{v,r}\!\left(\frac{X_{v,r}(h, x_r)}{p_{v,r}^+(T)} - 1\right), \quad M_r(h) = \prod_{s \leq r : h \in \mathcal{J}_s} E_{\mathrm{Dep},s}(h). \tag{27}
$$

If a verifier fails, $E_{v,r} \leq 1$; if independence, split hygiene, scope,
or residual calibration is absent, $\eta_{v,r} = 0$ and the pass contributes
exactly one. The certificate after family spending is

$$
\mathrm{Cert}(h, T, V_{1:r}) = [\log M_r(h) - \log(1/\alpha_h)]_+, \quad \alpha_h > 0,\ \sum_h \alpha_h \leq \alpha_\Gamma. \tag{28}
$$

**Theorem 1** *(Anytime, adaptive, family-valid false-pass control).*
On $\mathcal{C} = \cap_{v,r}\mathcal{C}_{v,r}$, if every invoked verifier
satisfies $\mathbb{E}_0[E_{v,r}(h) \mid \mathcal{F}_{r-1}] \leq 1$ for every
false $h \in \mathcal{J}_r$, then $M_r(h)$ is a nonnegative test
supermartingale [12]. For every $u \geq 0$,

$$
\Pr_0\big(\exists h \in \cup_{v,r}\mathcal{J}_r : \mathrm{Cert}(h, T, V_{1:\infty}) \geq u\big) \leq \delta_{cal} + e^{-u}\sum_h \alpha_h. \tag{29}
$$

Thus a rendered family with $\sum_h \alpha_h \leq \alpha_\Gamma$ has anytime
false-pass probability at most $\delta_{cal} + e^{-u}\alpha_\Gamma$ at excess
level $u$.

*Proof.* Conditional validity gives the supermartingale property. Ville's
inequality yields
$\Pr_0(\sup_r M_r(h) \geq e^u/\alpha_h \mid \mathcal{C}) \leq \alpha_h e^{-u}$.
Union over the ledgered family and add $\Pr(\mathcal{C}^c) \leq \delta_{cal}$.
$\square$

A certificate is scoped:

$$
\begin{aligned}
\mathrm{Cert}(h, T) = \min\{\,
& \mathrm{Cert}_{obs},\, \mathrm{Cert}_{cf},\, \mathrm{Cert}_{auth},\, \mathrm{Cert}_{priv},\, \mathrm{Cert}_{risk},\, \mathrm{Cert}_{scope},\, \mathrm{Cert}_{support},\, \mathrm{Cert}_{dep}, \\
& \mathrm{Cert}_{sub},\, \mathrm{Cert}_{sem},\, \mathrm{Cert}_{ctx},\, \mathrm{Cert}_{src},\, \mathrm{Cert}_{rebut},\, \mathrm{Cert}_{real},\, \mathrm{Cert}_{dom}\,\},
\end{aligned} \tag{30}
$$

$$
\mathrm{Cert}_{dom} = \begin{cases} \infty, & \mathrm{DomOK}(T,\Gamma) = 1, \\ 0, & \text{otherwise}, \end{cases} \tag{31}
$$

$$
\mathrm{Cert}_{dep} = [-\log(\epsilon_{dep} + \epsilon_0)]_+, \tag{32}
$$

$$
\mathrm{Cert}_{risk} = \min_{j \in J_R}\big[-\log\big((r_{cum,j} / (B_{R,j} + \epsilon)) + \epsilon_0\big)\big]_+. \tag{33}
$$

Non-applicable caps are set to $\infty$ by the contract. Evidence level $E_k$
is $\theta_k \leq \mathrm{Cert} < \theta_{k+1}$. Every rendered claim, action,
memory write, and declassification has a required $\theta$ and a
family-spending charge in the present scope. Raw pass counts, neural
confidence, majority vote, and repeated correlated tests are never evidence.

---

> **Implementation pointers.**
>
> - `Definition 1` (Mechanism) and the equivalence relation $\sim_\Gamma$ are
>   typed in `crates/vpm-core::mechanism` and exposed to Python via
>   `vpm._native.Mechanism`.
> - The recurrent state $Z_t$ is the canonical Rust struct produced and
>   consumed by `vpm-dsl::executor`.
> - The verifier registry, the e-value $E_{v,r}(h)$, the dependence block
>   factorization $E_{\mathrm{Dep},r}(h)$, the martingale $M_r(h)$, and the
>   scoped certificate $\mathrm{Cert}(h, T)$ live in `crates/vpm-verify`.
> - Anytime confidence-sequence bounds (Howard et al. [13]) and the
>   $p_{v,r}^+$ upper bound use `scipy.stats` plus a hand-rolled empirical
>   Bernstein bound in `vpm.verifiers.confidence`.
> - The repair operator search of (17)–(18) is the outer training loop in
>   `vpm.training.repair`.
