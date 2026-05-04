# §5 Compiler, posterior, and support-preserving inference

The compiler preserves alternatives:

$$
q_\phi(c, \mathfrak{n}, \widehat\Gamma, A, P \mid O, H) = \{(c_k, \mathfrak{n}_k, \widehat\Gamma_k, A_k, P_k, w_k, b_k, b_k^{ctx}, b_k^{sem}, b_k^{real})\}_{k=1}^K, \quad \sum_k w_k = 1, \tag{83}
$$

where $b_k$ is an upper bound on mass discarded by parser pruning,
$b_k^{ctx}$ by context/reference pruning, $b_k^{sem}$ by semantic-contract
pruning, and $b_k^{real}$ by realization-plan pruning. If $\Gamma$ is fixed
externally, only latent fields are inferred. Canonicalization is
posterior-valued, not destructive:

$$
q(\bar c \mid c) = \sum_{g \in \mathrm{Sym}(c)} q(g \mid c)\,\delta_{\mathrm{canon}(g \cdot c)}, \quad q(g \mid c) \propto \exp\!\big[-K(g \cdot c) - \lambda_d D(g \cdot c, c_{ref})\big]. \tag{84}
$$

Canonicalization may merge states only with a reversible witness or a
support-loss bound

$$
\epsilon_{canon} = \Pr_q[S_\Gamma(c) \neq S_\Gamma(\bar c)] \leq \epsilon_T. \tag{85}
$$

For candidate $\mu$,

$$
\begin{aligned}
L_\mathcal{D}(\mu \mid \Gamma) = \,& \ell_0(\bar \mu) + \sum_{k \in \mathrm{calls}(\bar \mu)} [-\log p(k \mid \tau, \Gamma)] + \lambda_b B(\mu) + \lambda_\Omega \Omega(\mu) + \lambda_r R_{reg}(\mu) \\
& + \lambda_s S_{scope}(\mu) + \lambda_{ctx} S_{ctx}(\mu) + \lambda_{sem} S_{sem}(\mu) + \lambda_{src} S_{src}(\mu) + \lambda_{rebut} S_{rebut}(\mu) + \lambda_{real} S_{real}(\mu) + \lambda_{fp} e^{-\mathrm{Cert}(\mu, T)},
\end{aligned} \tag{86}
$$

$$
\begin{aligned}
\mathcal{E}(\mu, c, \mathfrak{n}, V; T, \Gamma) = \,& D_T(Y, O) + \lambda_O D_{obs} + \lambda_{cf} D_{cf} + \beta L_\mathcal{D} + \lambda_\chi V_\chi + L_{auth} \\
& + \lambda_c \mathrm{Cost} + \lambda_{cal}\mathrm{Cal} + \lambda_{pr} \epsilon_{prune} + \lambda_R \mathrm{Risk} + \lambda_{ctx} \epsilon_{ctx} + \lambda_{sem} \epsilon_{sem} + \lambda_{src} \epsilon_{src} \\
& + \lambda_{rebut} \epsilon_{rebut} + \lambda_{real} \epsilon_{real} + \lambda_{ent} V_{ent}.
\end{aligned} \tag{87}
$$

A trainable score $s_\theta$ is a bounded proposal prior,
$|s_\theta| \leq s_{max}$:

$$
q([\mu], c, \mathfrak{n}, V \mid T, \Gamma) \propto q_\phi([\mu], \mathfrak{n} \mid T)\,\exp\!\big[-\mathcal{E}(\mu, c, \mathfrak{n}, V; T, \Gamma) + s_\theta([\mu], c, \mathfrak{n}, T)\big]. \tag{88}
$$

It can reduce search cost but cannot reduce $p_{v,r}^+$, increase
$\eta_{v,r}$, lower a gate threshold, or remove a lost-support, context-loss,
semantic-loss, source-recall, rebuttal-recall, realization-loss, privacy, or
cumulative-risk penalty.

At recurrent step $r$,

$$
B^{(r)} = (Z^{(r)}, E^{(r)}, M^{(r)}, Q^{(r)}, V^{(r)}, C^{(r)}, A^{(r)}, G^{(r)}, K^{(r)}, D^{(r)}, E_{sem}^{(r)}, C_{ctx}^{(r)}, R_{src}^{(r)}, R_{rebut}^{(r)}, \Omega_{real}^{(r)}, h^{(r)}). \tag{89}
$$

Slot $i$ has type posterior $\pi_i \in \Delta^{|T|}$ and authority label
$a_i \in \mathcal{L}_{auth}$. Sparse interaction is typed, tainted, and
budgeted:

$$
s_{ijk}^{(r)} = u_\theta(z_i, z_j, m_k, h, \Gamma) + \log m_{type}(i, j, k) + \log m_{auth}(i, j, k) - \lambda_c c_{ijk}, \tag{90}
$$

$$
\mathcal{N}_i^{(r)} = \mathrm{TopK}_{j,k}\, s_{ijk}^{(r)}, \quad m_i^{(r)} = \sum_{(j,k) \in \mathcal{N}_i^{(r)}} \Pr(j, k \mid i, B^{(r)}). \tag{91}
$$

Let $R_i^{(r)}$ be the event that every omitted interaction whose inclusion
can change $S_\Gamma$, context commitments, semantic atoms, source coverage,
material rebuttals, realization obligations, or a gate by more than
$\Delta_T$ is absent. The support guard is not the model probability alone
but a calibrated recall bound against exact expansions:

$$
\hat \epsilon_{mass}^{(r)} = \left[\sum_i (1 - m_i^{(r)})\right]_0^1, \tag{92}
$$

$$
\hat \epsilon_{rec}^{(r)} = \mathrm{UCB}_{1-\delta_{rec}}^{seq}\!\big(\Pr\{\exists i : R_i^{(r)} = 0\};\, \mathcal{D}_{rec}^{exact}\big), \tag{93}
$$

$$
\epsilon_{prune}^{(r)} = \min\{1,\, \hat \epsilon_{mass}^{(r)} + \hat \epsilon_{rec}^{(r)} + D_{shift}^{(r)}(T) + D_{select}^{(r)}(T)\}, \tag{94}
$$

$$
\mathrm{Cert}_{support}^{(r)} = [-\log(\epsilon_{prune}^{(r)} + \epsilon_0)]_+. \tag{95}
$$

If $\epsilon_{prune}^{(r)} > \epsilon_{max}(\Gamma)$, the cell must enlarge
$K$, widen the beam, rehydrate archival candidates, recover dropped
context/semantic/source/rebuttal/realization alternatives, or enter an exact
e-graph/program stage. If it does not, the final certificate is capped by
$\mathrm{Cert}_{support}^{(r)}$; for $\epsilon_{prune}^{(r)} = 1$ the cap is
zero.

The differentiable update proposes; typed projection, exact execution, and
verification dispose:

$$
\widehat B^{(r+1)} = \mathrm{TypedMessage}_\theta(B^{(r)}, C, \Gamma), \quad B^{(r+1)} = \Pi_{type, auth, budget}\!\big\{B^{(r)} + \eta_m\,(\widehat B^{(r+1)} - B^{(r)}) - \eta_e \nabla_B \tilde{\mathcal{E}}_\theta(B^{(r)}; T)\big\}. \tag{96}
$$

Resolution is staged,

$$
\iota \to \sigma \to \pi \to \eta, \tag{97}
$$

where $\iota$ are invariants, $\sigma$ sketches/e-classes, $\pi$ executable
programs/policies, and $\eta$ optional residuals. Stage $j+1$ is entered iff

$$
\begin{aligned}
\mathbb{E}[\Delta \mathrm{Cert} + \Delta U_\Gamma \mid j]\, & - \lambda_c \Delta \mathrm{Cost} - \lambda_L \Delta L - \lambda_R \Delta \mathrm{Risk} - \lambda_{supp} \Delta \epsilon_{prune} \\
& - \lambda_{ctx} \Delta \epsilon_{ctx} - \lambda_{sem} \Delta \epsilon_{sem} - \lambda_{src} \Delta \epsilon_{src} - \lambda_{rebut} \Delta \epsilon_{rebut} - \lambda_{real} \Delta \epsilon_{real} > 0.
\end{aligned} \tag{98}
$$

Residuals are escrowed: they may rank proposals, choose repairs, or render
uncertainty, but cannot certify claims or authorize side effects unless
wrapped by independent verifiers:

$$
\mathrm{Cert}(\eta, T) \leq \min_{v \in \nu(\eta)} \mathrm{Cert}(v, T), \quad \mathrm{Gate}(a_\eta, Z, \Gamma) = 0\ \text{if}\ \alpha(a_\eta) \not\preceq \alpha(\Gamma). \tag{99}
$$

Tests are actions over uncertainty. With $\Phi = e^{-\mathrm{Cert}}$,

$$
\begin{aligned}
e^* = \arg\max_{e \in A_{allow}} \frac{1}{c(e)} \Big\{\,
& \mathbb{E}_{o \sim \mathcal{P}[o \mid e, T]}[\Delta S_{ver}(o)] + \lambda_I I(o; M \mid e, T) + \lambda_V \mathbb{E}[\Phi_r - \Phi_{r+1}] \\
& -\, \lambda_R \mathrm{Risk}(e) - \lambda_H H_{bits}(e) - \lambda_A \mathrm{Annoy}(e) - \lambda_P \epsilon_{support}(e) \\
& -\, \lambda_{ctx} \epsilon_{ctx}(e) - \lambda_{sem} \epsilon_{sem}(e) - \lambda_{src} \epsilon_{src}(e) - \lambda_{rebut} \epsilon_{rebut}(e) - \lambda_{real} \epsilon_{real}(e)\,\Big\}.
\end{aligned} \tag{100}
$$

The loop halts when the contract threshold is met, a permitted clarification
dominates, or

$$
\begin{aligned}
\mathrm{EVC}_r = \mathbb{E}[\max_b U_\Gamma(b, r+1) - \max_b U_\Gamma(b, r)]\, & - \lambda_F \Delta F - \lambda_L \Delta L - \lambda_R \Delta \mathrm{Risk} - \lambda_P \Delta \epsilon_{prune} \\
& -\, \lambda_{ctx} \Delta \epsilon_{ctx} - \lambda_{sem} \Delta \epsilon_{sem} - \lambda_{src} \Delta \epsilon_{src} - \lambda_{rebut} \Delta \epsilon_{rebut} - \lambda_{real} \Delta \epsilon_{real} \leq 0.
\end{aligned} \tag{101}
$$

---

> **Implementation pointers.**
>
> - Compiler $q_\phi$ (eq. 83) → `vpm.compiler.compile`; alternatives are
>   stored as a typed posterior in `vpm-core::CompilerPosterior`.
> - Canonicalization (84–85) → `crates/vpm-egraph::canonicalize`, with the
>   reversible-witness ledger entry produced via `vpm-ledger::canon_witness`.
> - Posterior-energy $\mathcal{E}$ and the trainable score $s_\theta$
>   (86–88) → `vpm.compiler.energy` and `vpm.compiler.score_head`.
> - Recurrent state $B^{(r)}$ and the typed-message update (89–91, 96) →
>   `vpm.infer.cell`.
> - Support guard / pruning bound (92–95) → `vpm.infer.support_guard` plus
>   `crates/vpm-verify::support_guard`.
> - Stage scheduler $\iota \to \sigma \to \pi \to \eta$ (97–98) →
>   `vpm.infer.staging`.
> - Test selection / EVC halt (100–101) → `vpm.infer.test_select` and
>   `vpm.infer.halt`.
