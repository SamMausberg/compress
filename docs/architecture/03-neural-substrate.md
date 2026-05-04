# §3 Verifier-native neural substrate

The neural base model is $\mathrm{VNS}_\theta$, a proposal and prediction
substrate whose public interface is typed state, not hidden text logits.
Observations from text, pixels, grids, proof states, tools, tables, code, and
environment traces are encoded as a typed event hypergraph

$$
x_t = \mathrm{Enc}_\theta(o_t, \Gamma) = \big(\{e_{tm}, \tau_{tm}, \lambda_{tm}, P_{tm}\}_{m \leq M_t},\, R_t^{obs}\big), \tag{34}
$$

where $\tau$ is type, $\lambda$ authority/data taint, $P$ provenance, and
$R^{obs}$ observed relations. The substrate state is

$$
\begin{aligned}
H_t^\theta &= \big(h_t^{ssm},\, \{s_{ti}\}_{i \leq N_t},\, R_t^{kv},\, \mathcal{A}_t^{act},\, \mathcal{A}_t^{arc},\, \mathcal{G}_t^\theta,\, D_t^\theta,\, E_t^{sem,\theta},\, C_t^{ctx,\theta},\, R_t^{src,\theta},\, R_t^{rebut,\theta},\, \Omega_t^{real,\theta}\big), \\
s_{ti} &= (z_{ti}, \pi_{ti}, \tau_{ti}, a_{ti}, p_{ti}, q_{ti}, \kappa_{ti}, \rho_{ti}^{sem}, \rho_{ti}^{ctx}, \rho_{ti}^{src}, \rho_{ti}^{real}).
\end{aligned} \tag{35}
$$

with long-context recurrent state $h^{ssm}$, object/event slots $s_i$,
episodic key-value memory $R^{kv}$, active/archival mechanisms, a
differentiable shadow of the e-graph, discourse state, semantic atoms,
context commitments, source-coverage state, rebuttal-coverage state, and
realization obligations. The update is selective recurrent plus sparse typed
binding:

$$
h_t^{ssm} = \mathrm{SSM}_\theta\!\Big(h_{t-1}^{ssm},\, \sum_m W_{\tau_{tm}} e_{tm},\, u_{t-1},\, \Gamma\Big), \tag{36}
$$

$$
\tilde s_{ti} = s_{t-1, i} + F_\theta\!\Big(s_{t-1, i},\, h_t^{ssm},\, \sum_{m \in \mathcal{B}_i} A_{im} e_{tm},\, \Gamma\Big), \tag{37}
$$

$$
\bar m_{ti} = \sum_{(j, k) \in \mathrm{TopK}} \mathrm{Attn}_\theta(\tilde s_{ti}, \tilde s_{tj}, d_k, h_t^{ssm}, \Gamma), \quad s_{ti} = \Pi_{type, auth}(\tilde s_{ti} + G_\theta(\bar m_{ti})). \tag{38}
$$

Birth/death of slots is a posterior operation with lost-mass accounting.
Authority and provenance labels are joined monotonically during neural
updates, so the substrate cannot manufacture instruction or capability
labels.

Projection from substrate to mechanisms is explicit:

$$
\begin{aligned}
\mathrm{Proj}_\theta(H_t^\theta, T, \Gamma) = \big(\,
& q_\theta(c, \mathfrak{n}, \widehat\Gamma, A, P \mid T),\, q_\theta([\mu] \mid H_t^\theta, \Gamma),\, \hat \epsilon_{support},\, \hat \epsilon_{sub},\, \hat \epsilon_{sem},\, \hat \epsilon_{src},\, \hat \epsilon_{ctx}, \\
& \hat \epsilon_{rebut},\, \hat \epsilon_{real},\, \hat p_v^+,\, \hat \eta_v,\, \widehat{\mathrm{EVC}},\, \widehat U_\Gamma,\, \widehat H_\Gamma,\, \hat \rho_{reuse},\, \hat b_{mech},\, \hat r_{cum}\,\big).
\end{aligned} \tag{39}
$$

These quantities choose proposals, routing, widening, and questions. They are
never certificates. Exact execution, independent verifiers, calibrated
support recall, and gates determine certification.

The base model competes with transformers and state-space models by
containing them as proposal subcases while adding typed binding and
executable projection. In compact notation,

$$
\mathrm{VNS}_\theta = \mathrm{ExecProj} \circ \mathrm{Bind}_{type, auth} \circ \mathrm{Attn}_{sparse} \circ \mathrm{SSM}_{sel} \circ \mathrm{Enc}_\theta. \tag{40}
$$

With all tokens as slots, full attention, identity executable projection, and
disabled gates, this contains a standard transformer proposal class [1]; with
a single stream, no slot births, and no attention it contains selective
state-space proposal classes [4]. Slot binding supplies variable identity and
perceptual object permanence [9]; predictive-state losses supply world-model
semantics rather than next-token fluency alone [10, 11].

World modeling is over tests and interventions, not only sequences. For a
test $q = (a_{1:k}, \varphi)$,

$$
\hat P_\theta(q \mid H_t^\theta) = P_\theta\!\big(\varphi(O_{t+1 : t+k}) = 1 \mid \mathrm{do}(a_{1:k}),\, H_t^\theta,\, \Gamma\big), \tag{41}
$$

and the substrate must predict observations, counterfactuals, verifier
outcomes, conversation normal forms, semantic parses, context/reference
commitments, entailment obligations, source coverage, material rebuttals,
round-trip realization, support recall, and executable traces:

$$
\begin{aligned}
L_{base} = \,& L_{obs} + \lambda_{psr} L_{psr} + \lambda_{cf} L_{cf} + \lambda_{bind} L_{bind} + \lambda_{memref} L_{memref} + \lambda_{proj} L_{proj} + \lambda_{route} L_{dom} \\
& + \lambda_{ctx} L_{ctx}^{base} + \lambda_{sem} L_{sem}^{base} + \lambda_{ent} L_{ent}^{base} + \lambda_{src} L_{src}^{base} + \lambda_{rebut} L_{rebut}^{base} + \lambda_{real} L_{real}^{base},
\end{aligned} \tag{42}
$$

$$
L_{psr} = -\sum_{q \in Q_t^{audit}} \log \hat P_\theta(q \mid H_t^\theta), \quad L_{proj} = -\log \sum_{\mu \in \mathcal{M}^*(T)} q_\theta([\mu] \mid H_t^\theta, \Gamma), \tag{43}
$$

$$
L_{memref} = -\log \Pr_\theta\{\text{retrieves every audited certifying parent of } \tau \mid H_t^\theta\}, \tag{44}
$$

$$
\begin{aligned}
L_{dom} = \,& \mathrm{CE}(\widehat{\mathrm{Route}}_\theta, \mathrm{Route}) + \lambda_{cost}\|\hat c_v - c_v\|_1 + \lambda_{supp}\|\hat \epsilon_{support} - \epsilon_{support}\|_1 + \lambda_{sub}\|\hat \epsilon_{sub} - \epsilon_{sub}\|_1 \\
& + \lambda_{sem}\|\hat \epsilon_{sem} - \epsilon_{sem}\|_1 + \lambda_{ent}\|\hat \epsilon_{ctx} - \epsilon_{ctx}\|_1 + \lambda_{src}\|\hat \epsilon_{src} - \epsilon_{src}\|_1 \\
& + \lambda_{rebut}\|\hat \epsilon_{rebut} - \epsilon_{rebut}\|_1 + \lambda_{real}\|\hat \epsilon_{real} - \epsilon_{real}\|_1 + \lambda_{risk}\|\hat r_{cum} - r_{cum}\|_1.
\end{aligned} \tag{45}
$$

Language modeling and perceptual reconstruction are auxiliary views of the
same state:

$$
L_{obs} = -\sum_j w_j \log p_\theta(o_{tj} \mid H_t^\theta, \Gamma), \quad w_j = w_{claim}\mathbf{1}\{P_j \neq \emptyset\} + w_{style}\mathbf{1}\{j\ \text{nonclaim}\} + w_{neg}\mathbf{1}\{j\ \text{unsupported claim}\}. \tag{46}
$$

The renderer may use dense language capacity for style, compression, and
nonclaim glue, but factual, mathematical, executable, private, advice-like,
and side-effecting spans require semantic atoms, ledger parents, and
entailment witnesses. Substrate recall is calibrated exactly as other
pruning operations. Let $\mathcal{T}_{cert}(T)$ be the set of teacher traces
meeting the contract and let $\mathrm{miss}_\theta(T) = 1$ if the substrate
plus retrieval assigns no candidate in the bounded exact rehydration
neighborhood of any $\tau \in \mathcal{T}_{cert}(T)$. Then

$$
\epsilon_{sub}(T) = \mathrm{UCB}_{1-\delta_{sub}}^{seq}\!\big(\Pr\{\mathrm{miss}_\theta(T) = 1\};\, \mathcal{D}_{sub}^{audit}\big) + D_{shift}^{sub}(T) + D_{select}^{sub}(T), \quad \mathrm{Cert}_{sub} = [-\log(\epsilon_{sub}(T) + \epsilon_0)]_+. \tag{47}
$$

If $\epsilon_{sub} > \epsilon_{sub, max}$, inference must widen exact search,
rehydrate archive, ask for narrower input, or abstain. This prevents a fluent
but misbound base model from becoming an unobserved bottleneck.

The substrate is also audited for shortcut compression. Let
$\mathrm{Rehyd}_K(H_t^\theta)$ enumerate the exact symbolic/e-graph/program
states within edit, retrieval, and macro-expansion radius $K$ of the neural
projection. For a certifying trace family $\mathcal{T}_{cert}(T)$ define

$$
\mathrm{Crit}(T) = \{e : \exists \tau \in \mathcal{T}_{cert}(T)\ \text{such that removing edge}\ e\ \text{changes}\ \mathrm{Gate}(\tau)\ \text{or}\ S_\Gamma(\tau)\}, \tag{48}
$$

$$
\epsilon_{crit}(T) = \mathrm{UCB}_{1-\delta_{crit}}^{seq}\Pr\{\mathrm{Crit}(T) \not\subseteq \mathrm{Rehyd}_K(H_t^\theta)\} + D_{shift}^{crit} + D_{select}^{crit}. \tag{49}
$$

The neural state may be compressed, but critical edges may not be compressed
away without an exact reconstruction path. The execution gate uses
$\min\{\mathrm{Cert}_{sub},\, [-\log(\epsilon_{crit} + \epsilon_0)]_+\}$, and
training includes counterfactual edge probes that delete or swap critical
parents.

---

> **Implementation pointers.**
>
> - $\mathrm{VNS}_\theta$ is implemented in `python/vpm/substrate/`:
>   `encoder.py` (typed event hypergraph, eq. 34), `ssm.py` (selective SSM
>   block, eq. 36; Mamba reference [4]), `slots.py` (slot binding update,
>   eqs. 37–38; Slot Attention reference [9]), `projection.py` (executable
>   projection, eq. 39), `losses.py` (eqs. 42–46).
> - The differentiable e-graph shadow $\mathcal{G}_t^\theta$ mirrors the
>   exact e-graph in `crates/vpm-egraph`.
> - Substrate-recall calibration $\epsilon_{sub}(T)$ and the critical-edge
>   audit $\epsilon_{crit}(T)$ live in `vpm.verifiers.substrate` and the
>   teacher-trace ledger in `crates/vpm-ledger`.
> - Counterfactual edge-deletion / parent-swap probes (training-time) live
>   in `vpm.training.probes`.
