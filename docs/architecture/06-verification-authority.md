# §6 Verification, falsification, authority, and rendering

Every provisional pass is attacked. For verifier $v$,

$$
x_v^* = \arg\max_{x \in \mathcal{X}_T}\, \Pr[v(\mu, x) = \mathrm{pass}] - \Pr[S_\Gamma(\mu, x) = 1] - \lambda_x \mathrm{Cost}(x), \tag{102}
$$

using enumeration in exact domains and learned attacks in soft domains. The
verifier cannot certify a mechanism family it generated unless an independent
replay set or independent verifier family agrees. Calibration records update
$(n_v, f_v)$ on near-miss failures; certificates are recomputed after each
replay window.

Authority is a lattice $(\mathcal{L}_{auth}, \preceq, \vee)$ with labels for
data, retrieved sources, tool outputs, user instructions, developer/system
policy, private data, and granted capabilities. All transitions are label
monotone. Declassification is proof-carrying:

$$
\mathrm{Dec}_\ell(x, Z, \Gamma) = 1 \Rightarrow \mathrm{Cert}(\text{privacy proof}, \ell) \wedge \Pi_{priv}(\ell, x, \Gamma). \tag{103}
$$

External text and tool output are data unless lifted by a valid contract
rule:

$$
\mathrm{Auth}(\text{tool output}) = \text{data}, \quad \text{data} \not\succeq \text{instruction}, \quad \text{data} \not\succeq \text{capability}. \tag{104}
$$

Define

$$
\mathrm{AuthOK}(a, Z) = \mathrm{Auth}(a, Z) \succeq \alpha(a), \quad \mathrm{CertOK}(a, Z) = \mathrm{Cert}(a, Z) \geq V_a, \tag{105}
$$

$$
\begin{aligned}
\mathrm{GrantOK}(a, Z, \Gamma) = \mathrm{Rollback}(a)\, & \vee\, \big(\mathrm{ExplicitGrant}(a, Z) \wedge \mathrm{Auth}(\mathrm{grant}, Z) \succeq \text{instruction} \\
& \wedge\, \mathrm{Cert}(\mathrm{grant}, Z) \geq \theta_{grant} \wedge \mathrm{Fresh}(\mathrm{grant}) \wedge \mathrm{ScopeOK}(\mathrm{grant}, a, \Gamma)\big),
\end{aligned} \tag{106}
$$

$$
\mathrm{PrivOK}(a, Z, \Gamma) = \Pi_{priv}(\mathrm{read}(a) \cup \mathrm{write}(a), \lambda(a), \Gamma) \wedge \mathrm{Exfil}(a) \leq E_a, \tag{107}
$$

$$
\mathrm{SandOK}(a, Z) = \mathrm{Sandbox}(a, Z) \wedge \mathrm{RollbackPlan}(a, Z) \wedge \mathrm{Monitor}(a, Z), \tag{108}
$$

$$
\mathrm{RiskOK}(a, Z, \Gamma) = \mathbf{1}\Big\{\mathbf{U}_R(a, Z) \preceq \mathbf{B}_R(\Gamma) \wedge \sup_{w \in \mathcal{W}_R} w^\top \mathbf{U}_R(a, Z) \leq \sup_{w \in \mathcal{W}_R} w^\top \mathbf{B}_R(\Gamma)\Big\}, \tag{109}
$$

$$
\mathbf{U}_R(a, Z) = \mathrm{UCB}_{1-\delta_R}(\mathbf{r}_t + \Delta\mathbf{r}(a)), \quad \Delta \mathbf{r}(a) = (\Delta r^{impact}, \Delta r^{priv}, \Delta r^{exfil}, \Delta r^{cap}, \Delta r^{infl}, \Delta r^{conf}, \Delta r^{model}, \Delta r^{dep}). \tag{110}
$$

Cumulative safety is budgeted over the transcript and over channels, not
checked only per action and not averaged into a scalar:

$$
\mathbf{r}_{t+1} = \mathbf{r}_t + \widehat{\Delta\mathbf{r}}(a_t) - \mathbf{c}_t^{roll}, \tag{111}
$$

$$
0 \preceq \mathbf{c}_t^{roll} \preceq \mathrm{Cred}(a_t), \quad \mathrm{Cred}_j(a_t) > 0 \Rightarrow \mathrm{Cert}(\mathrm{monitor}_j \wedge \mathrm{restored}_j, Z_{t+1}) \geq \theta_{roll, j}. \tag{112}
$$

A rollback credit cannot restore privacy disclosure, capability transfer, or
external influence unless the corresponding external state has a certified
reversal witness; otherwise that coordinate is absorbing. The action gate is

$$
\begin{aligned}
\mathrm{Gate}(a, Z, \Gamma) = \mathbf{1}\Big\{\,
& \mathrm{DomOK}(Z, \Gamma) \wedge \mathrm{IntentOK}(a, Z, \Gamma) \wedge \Pi_{perm}(a, \Gamma) \wedge \mathrm{AuthOK}(a, Z) \wedge \mathrm{Pre}(a, Z) \wedge \mathrm{CertOK}(a, Z) \\
& \wedge\, \mathbb{E}[\mathrm{Impact}(a)] \leq I_a \wedge \mathrm{Budget}(a) \leq B_a \wedge \mathrm{GrantOK}(a, Z, \Gamma) \\
& \wedge\, \mathrm{PrivOK}(a, Z, \Gamma) \wedge \mathrm{SandOK}(a, Z) \wedge \mathrm{RiskOK}(a, Z, \Gamma) \wedge \Pi_{infl}(a, \Gamma) \wedge \mathrm{NoConflict}(a, Z, \Gamma)\,\Big\}.
\end{aligned} \tag{113}
$$

Explicit grant can satisfy reversibility for scoped irreversible actions only
when the grant itself has instruction authority, freshness, scope, and
certificate. It cannot bypass permission, authority, preconditions,
certificates, impact limits, budget, privacy, sandboxing, cumulative-risk
limits, exfiltration bounds, or policy conflict.

**Proposition 3** *(Data noninterference).* If updates are label-monotone,
data labels cannot dominate instruction or capability labels, and
$\mathrm{GrantOK}$ requires instruction-authority grant provenance, then
data-only inputs cannot make $\mathrm{Gate}(a, Z, \Gamma) = 1$ for an action
requiring instruction or capability authority unless the authority is already
present in $\Gamma$ or in a pre-existing explicit granted capability.

*Proof.* Monotone joins preserve the partial order. No sequence of joins
with labels below or incomparable to $\alpha(a)$ can produce a label
dominating $\alpha(a)$. Thus $\mathrm{AuthOK}$ remains false without an
independent grant. Data also cannot manufacture $\mathrm{GrantOK}$ because
grant provenance must dominate instruction. The gate is conjunctive, so any
failed conjunct blocks the action. $\square$

Renderers are semantic transducers from certified state to text. For a span
$y_j$, let $\mathrm{Atom}(y_j)$ be the output atoms and $P_a$ the ledger
parents of atom $a$:

$$
\lambda(y_j) = \bigvee_{a \in \mathrm{Atom}(y_j)} \!\Big(\lambda_a \vee \bigvee_{p \in P_a} \lambda(p)\Big), \tag{114}
$$

$$
\mathrm{Cert}(y_j) = \min_{a \in \mathrm{Atom}(y_j)\,:\, m(a) \in \mathcal{M}_{cert}}\, \mathrm{Cert}_{atom}(a), \tag{115}
$$

$$
\alpha(y_j) = \bigvee_{a \in \mathrm{Atom}(y_j)} \!\Big(\alpha_a \vee \bigvee_{p \in P_a} \alpha(p)\Big). \tag{116}
$$

A factual sentence, mathematical derivation, private datum, code execution
instruction, source quote, advice-like claim, or irreversible plan inherits
the maximum evidence, context, source, rebuttal, realization, privacy, and
authority requirement of its atoms. A purely stylistic span has no
certificate requirement only if the independent checker verifies
$\mathrm{Atom}(y_j) \cap \mathcal{M}_{cert} = \emptyset$ and $\Pi_{infl}$
accepts the span.

$$
\begin{aligned}
\mathrm{Gate}_{render}(y_j, Z, \Gamma) = \mathbf{1}\Big\{\,
& \mathrm{DomOK}(Z, \Gamma) \wedge \mathrm{IntentOK}(y_j, Z, \Gamma) \wedge \mathrm{Gate}_{NL}(y_j, Z, \Gamma) \\
& \wedge\, \Pi_{priv}(y_j, \lambda(y_j), \Gamma) \wedge \alpha(y_j) \preceq \alpha(\Gamma)\,\Big\}.
\end{aligned} \tag{117}
$$

Unsupported certified-mode spans are replaced by uncertainty, a question, a
scoped assumption, or refusal; unsupported soft spans are allowed only when
they remain nonclaim text under $\Pi_{style}$.

---

> **Implementation pointers.**
>
> - Falsifier $x_v^*$ (102) → `crates/vpm-verify::falsifier`; learned attacks
>   for soft domains in `vpm.verifiers.attacks`.
> - Authority lattice $(\mathcal{L}_{auth}, \preceq, \vee)$, declassification
>   proofs (103–104), `AuthOK` / `GrantOK` / `PrivOK` / `SandOK` / `RiskOK`
>   (105–110) → `crates/vpm-authority`.
> - Vector-risk ledger and rollback credits (111–112) →
>   `crates/vpm-ledger::risk_ledger`.
> - The action gate `Gate(a, Z, Γ)` (113) is the canonical entry point in
>   `crates/vpm-verify::gate`.
> - Proposition 3 is enforced by construction (Rust type signatures preserve
>   label monotonicity); a property test under
>   `tests/integration/test_authority_noninterference.rs` exercises it on
>   randomized transcripts.
> - Renderer span certificate / authority join (114–116) and
>   `Gate_render` (117) → `vpm.language.render` plus
>   `crates/vpm-verify::gate_render`.
