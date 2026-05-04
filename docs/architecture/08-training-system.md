# §8 Training system and inefficiency controls

Training is execution-first. Before neural learning, implement the typed
DSL/bytecode, context normalizer, semantic-contract compiler, source/evidence
retriever, rebuttal retriever, entailment checker, round-trip realization
checker, batch executor, verifier registry, falsifier, trace DAG, e-class
DAG, provenance graph, authority lattice, declassification checker,
influence-risk checker, cumulative-risk ledger, memory admission/demotion,
and renderers. Neural modules are trained against traces produced by this
executable system, not against unverifiable text alone.

Data are split by role:

$$
\begin{aligned}
\mathcal{D}_{all} &= \mathcal{D}_{prop} \cup \mathcal{D}_{ctx} \cup \mathcal{D}_{sem} \cup \mathcal{D}_{src} \cup \mathcal{D}_{rebut} \cup \mathcal{D}_{ent} \cup \mathcal{D}_{real} \cup \mathcal{D}_{dep} \cup \mathcal{D}_{front} \cup \mathcal{D}_{ver} \cup \mathcal{D}_{cal} \cup \mathcal{D}_{audit} \cup \mathcal{D}_{replay} \cup \mathcal{D}_{red}, \\
\mathcal{D}_{fit} &= \mathcal{D}_{all} \setminus \mathcal{D}_{audit}.
\end{aligned} \tag{137}
$$

Proposal models train on $\mathcal{D}_{prop}$; context normalizers on
$\mathcal{D}_{ctx}$; semantic contractizers on $\mathcal{D}_{sem}$;
source-recall estimators on $\mathcal{D}_{src}$; rebuttal-recall estimators
on $\mathcal{D}_{rebut}$; entailment checkers on $\mathcal{D}_{ent}$;
realization checkers on $\mathcal{D}_{real}$; dependence residualizers on
$\mathcal{D}_{dep}$; frontier/admission predictors on $\mathcal{D}_{front}$;
verifiers train on $\mathcal{D}_{ver}$; false-pass and admission bounds are
estimated on $\mathcal{D}_{cal}$; claims are reported only against
$\mathcal{D}_{audit}$ or deployment replay; red-team attacks draw from
$\mathcal{D}_{red}$. Cross-fitting rotates the roles, but certification is
pairwise-disjoint at the event level. For $e = (\mu, a, v, T, r)$, set

$$
\begin{aligned}
G_e &= \mathrm{split}_{gen}(\mu) \cup \mathrm{split}_{ctx}(\mu) \cup \mathrm{split}_{sem}(\mu) \cup \mathrm{split}_{src}(a) \cup \mathrm{split}_{rebut}(a) \cup \mathrm{split}_{real}(a) \cup \mathrm{split}_{dep}(a) \cup \mathrm{split}_{front}(\mu), \\
W_e &= \mathrm{split}_{train}(v) \cup \mathrm{split}_{cal}(v) \cup \mathrm{split}_{audit}, \\
A_e &= G_e \cup \mathrm{split}_{ent}(a) \cup \mathrm{split}_{train}(v) \cup \mathrm{split}_{cal}(v).
\end{aligned} \tag{138}
$$

The certificate-carrying event must satisfy

$$
G_e \cap W_e = \emptyset, \quad \mathrm{split}_{train}(v) \cap \mathrm{split}_{cal}(v) = \emptyset, \quad \mathrm{split}_{audit} \cap A_e = \emptyset. \tag{139}
$$

The earlier three-way-intersection rule is insufficient: it allows a
candidate and verifier to share data whenever calibration is separate. The
same defect appears for context normalizers, semantic parsers,
source/rebuttal retrievers, entailment checkers, and realization checkers.
Here any leakage sets $\eta_{split} = 0$ and removes certificate weight; the
trace can still be used for proposal learning or replay prioritization.

A trace is a DAG

$$
\begin{aligned}
\tau &= (N, E), \\
n \in N &= (B_n, c_n, M_n, V_n, A_n, G_n, K_n, D_n, E_n^{sem}, C_n^{ctx}, R_n^{src}, R_n^{rebut}, \Omega_n^{real}, \mathrm{Dep}_n, \mathrm{Front}_n, \mathbf{r}_n), \\
e : n \to n' &= (a_e, o_e, \mathrm{cost}_e, \mathbf{r}_e, \mathrm{Cert}_e, \mathrm{loss}_e, \mathrm{ctx}_e, \mathrm{sem}_e, \mathrm{src}_e, \mathrm{rebut}_e, \mathrm{real}_e, \mathrm{dep}_e, \mathrm{front}_e),
\end{aligned} \tag{140}
$$

containing parses, context commitments, semantic atoms, source retrieval,
rebuttal retrieval, entailment witnesses, realization witnesses, dependence
blocks, vector-risk ledgers, frontier movements, invariants, sketches,
programs, verifier choices, repairs, falsifier failures, active questions,
support-loss bounds, gates, halts, memory writes, and final renderings.
Teacher search defines a truncated posterior over certified traces

$$
\begin{aligned}
p_*(\tau \mid T) \propto \,& \exp[-\mathcal{E}_*(\tau; T)]\,\mathbf{1}\{\mathrm{Cert}(\tau) \geq V_\Gamma\}\,\mathbf{1}\{\mathrm{Gate}(\tau) = 1\}\,\mathbf{1}\{\mathrm{Gate}_{NL}(\mathrm{Render}(\tau)) = 1\} \\
& \cdot \mathbf{1}\{\epsilon_{support}(\tau) \leq \epsilon_\Gamma\}\,\mathbf{1}\{\epsilon_{ctx}(\tau) \leq \epsilon_{ctx,max}\}\,\mathbf{1}\{\epsilon_{sem}(\tau) \leq \epsilon_{sem,max}\} \\
& \cdot \mathbf{1}\{\epsilon_{src}(\tau) \leq \epsilon_{src,max}\}\,\mathbf{1}\{\epsilon_{rebut}(\tau) \leq \epsilon_{rebut,max}\}\,\mathbf{1}\{\epsilon_{real}(\tau) \leq \epsilon_{real,max}\}.
\end{aligned} \tag{141}
$$

where $\mathcal{E}_*$ uses exact execution whenever available. Learned
heuristics may order expansions but may not prune a trace unless they
provide an admissible upper bound on lost certified utility. If no trace
meets the threshold, the target is ask/abstain with calibrated uncertainty,
not an answer.

The main objective is amortized free energy plus trace, value, calibration,
verifier, semantic, source, entailment, safety, compression, support, and
renderer losses:

$$
\begin{aligned}
L(\theta, \phi) = \mathbb{E}_{T \sim \mathcal{D}_{fit}}\Big[\,
& \mathcal{J}_T(q_{\theta, \phi}) + \lambda_{base}L_{base} + \lambda_{cmp}L_{cmp} + \lambda_{tr}L_{trace} \\
& + \lambda_{val}L_{value} + \lambda_{rep}L_{repair} + \lambda_{halt}L_{halt} + \lambda_{ver}L_{ver} + \lambda_{cal}L_{cal} \\
& + \lambda_{safe}L_{safe} + \lambda_{mem}L_{mem} + \lambda_{supp}L_{supp} + \lambda_{ren}L_{render} \\
& + \lambda_{ctx}L_{ctx} + \lambda_{sem}L_{sem} + \lambda_{src}L_{src} + \lambda_{rebut}L_{rebut} + \lambda_{ent}L_{ent} + \lambda_{real}L_{real} + \lambda_{tb}L_{TB} + \lambda_{mf}L_{mf} \\
& + \lambda_{split}L_{split} + \lambda_{sub}L_{sub} + \lambda_{dom}L_{dom} + \lambda_{dep}L_{dep} + \lambda_{front}L_{front} + \lambda_{probe}L_{probe}\,\Big],
\end{aligned} \tag{142}
$$

$$
\mathcal{J}_T(q) = \mathbb{E}_{q(\mu, c, \mathfrak{n}, V)}[\mathcal{E}(\mu, c, \mathfrak{n}, V; T, \Gamma)] + \mathrm{KL}\!\big(q(\mu, c, \mathfrak{n}, V \mid T) \,\|\, p_0(\mu, c, \mathfrak{n}, V \mid \Gamma)\big). \tag{143}
$$

The substrate loss $L_{base}$ is defined on audited observation,
predictive-state, memory-recall, context, semantic, source, rebuttal,
entailment, realization, projection, and routing targets above. Substrate
calibration is separately penalized by

$$
L_{sub} = \mathrm{ECE}(\hat \epsilon_{sub}) + \lambda_{miss}\,\mathrm{BCE}(\widehat{\mathrm{miss}}, \mathrm{miss}) + \lambda_{bound}\max\{0, \widehat{\Pr}(\mathrm{miss}) - \epsilon_{sub}\}. \tag{144}
$$

Dependence, frontier, and critical-edge losses remove three training
inefficiencies: duplicated correlated evidence, macros that improve training
likelihood but not held-out compression, and neural states that omit the
sparse edges later needed for certification.

$$
L_{dep} = \mathrm{BCE}(\widehat{\mathrm{Leak}}_{dep}, \mathrm{Leak}_{dep}) + \mathrm{ECE}(\hat \epsilon_{dep}) + \lambda_{res}\|\widehat X_g^+ - X_g^+\|_1 + \lambda_{bound}\max\{0, \widehat{\Pr}(\mathrm{Leak}_{dep}) - \epsilon_{dep}\}, \tag{145}
$$

$$
L_{front} = \|\Delta \widehat{\mathrm{Front}}_\theta - \Delta \mathrm{Front}\|_2^2 + \mathrm{BCE}(\widehat{\mathrm{Admit}}, \mathbf{1}\{\Delta\mathrm{Front} > 0\}) + \lambda_{dup}\widehat{\Delta I} + \lambda_d\max\{0, \widehat{d}_T - D\}, \tag{146}
$$

$$
L_{probe} = \mathrm{BCE}(\widehat{\mathrm{keep}}_{crit}, \mathbf{1}\{e \in \mathrm{Crit}(T)\}) + \lambda_{swap}\mathrm{CE}(\widehat S_T(\mathrm{swap}(e)), S_\Gamma(\mathrm{swap}(e))) + \lambda_{rehyd}\max\{0, \hat \epsilon_{crit} - \epsilon_{crit,max}\}. \tag{147}
$$

Compiler learning marginalizes over parse alternatives rather than
selecting a single parse:

$$
L_{cmp} = -\log \sum_{k=1}^K q_\phi(c_k, \mathfrak{n}_k, \Gamma_k, A_k, P_k \mid O, H)\mathbf{1}\{c_k \in C^*(T), \mathfrak{n}_k \in \mathcal{N}^*(T)\} + \lambda_\Sigma \mathrm{NLL}_{cal}(\Sigma_k) + \lambda_b b_k + \lambda_{bctx}b_k^{ctx} + \lambda_{bsem}b_k^{sem} + \lambda_{breal}b_k^{real}. \tag{148}
$$

Context, semantic, source, rebuttal, entailment, and realization losses are
optimized as recall-first gates rather than fluent-text objectives:

$$
L_{ctx} = -\log \sum_{\mathfrak{n} \in \mathcal{N}^*(T)} q_\psi(\mathfrak{n} \mid U, H, \Gamma) + \lambda_{ref}\mathrm{CE}(\widehat D, D) + \lambda_{gran}\mathrm{CE}(\widehat \gamma, \gamma) + \lambda_{ask}\mathrm{BCE}(\widehat{\mathrm{Ask}}, \mathbf{1}\{\mathrm{CtxOK} = 0\}), \tag{149}
$$

$$
L_{sem} = -\log \sum_{\mathfrak{n} \in \mathcal{N}^*(T)} q_\psi(\mathfrak{n} \mid U, H, \Gamma) + \lambda_{mode}\mathrm{CE}(\widehat m, m) + \lambda_{dep}\mathrm{CE}(\widehat \Delta, \Delta) + \lambda_{ask}\mathrm{BCE}(\widehat{\mathrm{Ask}}, \mathbf{1}\{\mathrm{SemOK} = 0\}), \tag{150}
$$

$$
L_{src} = \mathrm{BCE}(\widehat{\mathrm{miss}}_{src}, \mathrm{miss}_{src}) + \mathrm{ECE}(\hat \epsilon_{src}) + \lambda_{bound}\max\{0, \widehat{\Pr}(\mathrm{miss}_{src}) - \epsilon_{src}\}, \tag{151}
$$

$$
L_{rebut} = \mathrm{BCE}(\widehat{\mathrm{miss}}_{rebut}, \mathrm{miss}_{rebut}) + \mathrm{ECE}(\hat \epsilon_{rebut}) + \lambda_{bound}\max\{0, \widehat{\Pr}(\mathrm{miss}_{rebut}) - \epsilon_{rebut}\}, \tag{152}
$$

$$
L_{ent} = \mathrm{BCE}(\widehat{\mathrm{Ent}}(P_a, e_a, \Omega_a), \mathrm{Ent}^*) + \lambda_{hard}\max\{0, 1 + s_\theta(a^-) - s_\theta(a^+)\}, \tag{153}
$$

$$
L_{real} = \mathrm{BCE}(\widehat{\mathrm{Hidden}}_{cert}, \mathrm{Hidden}_{cert}) + \mathrm{ECE}(\hat \epsilon_{real}) + \lambda_{rt}\mathrm{CE}(\widehat{\mathrm{Atom}}(y), \mathrm{Atom}(y)) + \lambda_{bound}\max\{0, \widehat{\Pr}(\neg \mathrm{RealOK}) - \epsilon_{real}\}. \tag{154}
$$

Trace imitation is advantage-weighted and off-policy corrected, but only
when effective sample size is adequate:

$$
\bar w_e = \mathrm{clip}\!\left(\frac{p_*(\tau \mid T)}{q_{old}(\tau \mid T)}, 0, w_{max}\right)\max\{0, A_e\}, \quad \mathrm{ESS} = \frac{(\sum_e \bar w_e)^2}{\sum_e \bar w_e^2}, \tag{155}
$$

$$
L_{trace} = \mathbf{1}\{\mathrm{ESS} \geq N_{ess}\}\!\left[-\sum_{e \in \tau_*} \bar w_e \log \pi_\theta(a_e \mid B_e, \Gamma)\right]. \tag{156}
$$

If $\mathrm{ESS} < N_{ess}$, the batch is used for replay prioritization or
new teacher search, not for high-variance imitation. The value and halting
heads learn the expected marginal certified utility of another step:

$$
L_{value} = \sum_r \|v_\theta(B^{(r)}) - \mathrm{EVC}_r^{MC}\|_2^2, \quad L_{halt} = \sum_r \mathrm{BCE}(\sigma(g_\theta(B^{(r)})), \mathbf{1}\{\mathrm{EVC}_r \leq 0 \vee \mathrm{Cert}_r \geq V\}). \tag{157}
$$

Repairs and near-miss negatives train discrimination at the decision
boundary:

$$
L_{repair} = -\log p(\rho^* \mid B, \mu^-, x^*) + \lambda_{rank}\max\{0, 1 + E_\theta(\mu^+) - E_\theta(\mu^-)\}. \tag{158}
$$

Verifier training calibrates false-pass probability, not merely pass
accuracy:

$$
L_{ver} = \sum_v \big[-f_v\log \hat p_v - (n_v - f_v)\log(1 - \hat p_v)\big] + \lambda_{adv}\mathbb{E}_{x^*}\,\ell(v(\mu, x^*), S_\Gamma(\mu, x^*)). \tag{159}
$$

Calibration uses binned, distributional, and anytime penalties:

$$
L_{cal} = \mathrm{ECE}(q) + \mathbb{E}_q[(\Pr[S_\Gamma = 1 \mid B] - \mathbf{1}\{S_\Gamma = 1\})^2] + \sum_v \mathrm{ECE}(\hat p_v) + \lambda_{cs}\sum_{v,r}\max\{0, \hat p_{v,r} - p_{v,r}^+\}. \tag{160}
$$

Split hygiene and multi-fidelity verification are trained as predictors but
enforced as hard constraints:

$$
L_{split} = \sum_e \mathrm{BCE}(\widehat{\mathrm{leak}}_\theta(e), \mathbf{1}\{\Pi_{split}(e) = 0\}) + \lambda_{audit}\mathbf{1}\{\mathrm{Sel}(e) \cap \mathcal{D}_{audit} \neq \emptyset\}, \tag{161}
$$

$$
L_{mf} = \sum_{j < k} \mathrm{CE}(\hat r_{j \to k}, \mathbf{1}\{\text{stage}\ k\ \text{changed certificate or risk}\}). \tag{162}
$$

Cheap verifier stages may reject or route candidates. They may certify only
through their own calibrated e-values; otherwise certificate weight is
assigned to the terminal independent stage. Safety is trained as prediction
but enforced as a hard gate:

$$
L_{safe} = \mathrm{CE}(\hat \alpha_\theta, \alpha) + \|\widehat{\Delta\mathbf{r}}_\theta - \Delta \mathbf{r}\|_1 + \lambda_{cum}\|\widehat{\mathbf{r}}_{cum} - \mathbf{r}_{cum}\|_1 + \lambda_{infl}\ell_{infl} + \lambda_{conf}\ell_{conflict}. \tag{163}
$$

Gate is not relaxed during execution.

Support learning predicts lost mass and trains recall before precision:

$$
\begin{aligned}
L_{supp} = \sum_r \Big[\,
& \max\{0, \epsilon_{prune}^{(r)} - \epsilon_{max}\} + \mathrm{BCE}(\widehat m_i^{(r)}, \mathbf{1}\{(j, k) \in \mathcal{N}_i^{oracle}\}) \\
& + \lambda_{ctx}\mathrm{BCE}(\widehat{\mathrm{keep}}_i, \mathbf{1}\{C^{ctx} \subset C^{oracle}\}) + \lambda_{sem}\mathrm{BCE}(\widehat{\mathrm{keep}}_i, \mathbf{1}\{a \in \mathcal{A}^{oracle}\}) + \lambda_{src}\mathrm{BCE}(\widehat{\mathrm{keep}}_i, \mathbf{1}\{S \subset R^{oracle}\}) \\
& + \lambda_{rebut}\mathrm{BCE}(\widehat{\mathrm{keep}}_{rebut}, \mathbf{1}\{S^- \subset R^{oracle}\}) + \lambda_{real}\mathrm{BCE}(\widehat{\mathrm{keep}}_{real}, \mathbf{1}\{\Omega^{real} \in \Omega^{oracle}\})\,\Big].
\end{aligned} \tag{164}
$$

Compression learning predicts admission gain while the actual decision
remains the confidence-bound rule:

$$
\begin{aligned}
L_{mem} = \,& \|\widehat A_\theta(m) - A_t(m)\|_2^2 + \mathrm{BCE}(\hat d_\theta(m), \mathbf{1}\{\mathrm{LCB}\,A(m) > 0\}) + \lambda_{scope}\mathrm{CE}(\hat \omega_m, \omega_m) \\
& + \lambda_{ctx}\mathrm{CE}(\hat \rho_m^{ctx}, \rho_m^{ctx}) + \lambda_{sem}\mathrm{CE}(\hat \sigma_m^{sem}, \sigma_m^{sem}) + \lambda_{src}\mathrm{CE}(\hat \rho_m^{src}, \rho_m^{src}) + \lambda_{rebut}\mathrm{CE}(\hat \rho_m^{rebut}, \rho_m^{rebut}) + \lambda_{real}\mathrm{CE}(\hat \rho_m^{real}, \rho_m^{real}) \\
& + \lambda_{dep}\mathrm{CE}(\hat \epsilon_{dep,m}, \epsilon_{dep,m}) + \lambda_{front}\|\Delta \widehat{\mathrm{Front}}_m - \Delta \mathrm{Front}_m\|^2 + \lambda_R\|\widehat{\Delta \mathbf{r}}_m - \Delta \mathbf{r}_m\|_1.
\end{aligned} \tag{165}
$$

For generative mechanism proposals, a GFlowNet trajectory-balance loss
samples diverse certified mechanisms:

$$
\begin{aligned}
R(\mu, T) = \,& \epsilon + \exp[-\mathcal{E}(\mu, T)]\mathbf{1}\{\mathrm{Cert}(\mu, T) \geq V_T\}\mathbf{1}\{\epsilon_{support}(\mu) \leq \epsilon_T\} \\
& \cdot \mathbf{1}\{\epsilon_{ctx}(\mu) \leq \epsilon_{ctx,max}\}\mathbf{1}\{\epsilon_{sem}(\mu) \leq \epsilon_{sem,max}\}\mathbf{1}\{\epsilon_{src}(\mu) \leq \epsilon_{src,max}\}\mathbf{1}\{\epsilon_{rebut}(\mu) \leq \epsilon_{rebut,max}\}\mathbf{1}\{\epsilon_{real}(\mu) \leq \epsilon_{real,max}\} \\
& \cdot \mathbf{1}\{\epsilon_{dep}(\mu) \leq \epsilon_{dep,max}\}\mathbf{1}\{\Delta \mathrm{Front}(\mu) > 0\}\mathbf{1}\{\Delta\mathbf{r}(\mu) \preceq \mathbf{B}_R - \mathbf{r}_t\},
\end{aligned} \tag{166}
$$

$$
L_{TB} = \!\left(\log Z_\theta(T) + \sum_r \log P_F(a_r \mid s_r, T) - \log R(\mu, T) - \sum_r \log P_B(a_r \mid s_{r+1}, T)\right)^{\!2}. \tag{167}
$$

Renderer training is conditioned on certified atoms and cannot invent
unsupported claims:

$$
\begin{aligned}
L_{render} = \,& -\log p_\theta(y_{cert} \mid Z_{cert}, \mathrm{Atom}(y), \Gamma) + \lambda_{unc}\mathrm{CE}(\widehat E(y), E(y)) + \lambda_{ctx}\mathrm{CE}(\widehat C(y), C^{ctx}(y)) + \lambda_{src}\mathrm{CE}(\widehat P(y), P(y)) \\
& + \lambda_{rebut}\mathrm{BCE}(\widehat R^-(y), R^-(y)) + \lambda_{real}\mathrm{BCE}(\mathrm{RealOK}(y), \mathrm{RealOK}(y, A_y, Z, \Gamma)) + \lambda_{atom}\mathrm{CE}(\widehat{\mathrm{Atom}}(y), \mathrm{Atom}(y)) + \lambda_{hall}\mathbf{1}\{\mathrm{Def}(y) \neq \emptyset\}.
\end{aligned} \tag{168}
$$

Loss weights are normalized by gradient scale so cheap text or imitation
losses cannot overwhelm verification, calibration, context recall, semantic
recall, source recall, rebuttal recall, realization recall, support recall,
or safety:

$$
\lambda_i \leftarrow \mathrm{clip}_{[l_i, u_i]}\!\left\{\lambda_i\!\left(\frac{\mathrm{median}_j \bar g_j}{\bar g_i + \tau}\right)^{\!\alpha}\right\}, \quad \bar g_i = (1 - \beta_g)\bar g_i + \beta_g \|\nabla_\theta L_i\|_2. \tag{169}
$$

Training is block-coordinate with frozen audit labels and hard gates;
otherwise cheap language likelihood would create gradient paths that appear
to improve utility while eroding certification. Let

$$
\Theta = (\psi_{ctx}, \psi_{sem}, \rho_{src}, \rho_{rebut}, \rho_{ent}, \rho_{real}, \rho_{dep}, \xi_{ver}, \theta_{sub}, \phi_{cmp}, \omega_{ren}, \kappa_{mem}, \kappa_{front}). \tag{170}
$$

Then

$$
(\psi_{ctx}^+, \psi_{sem}^+) = \arg\min_\psi L_{ctx} + L_{sem} + \lambda_{ask}L_{ask}, \tag{171}
$$

$$
(\rho_{src}^+, \rho_{rebut}^+, \rho_{ent}^+, \rho_{real}^+, \rho_{dep}^+, \xi_{ver}^+) = \arg\min_{\rho, \xi} L_{src} + L_{rebut} + L_{ent} + L_{real} + L_{dep} + L_{ver} + L_{cal} + L_{split}, \tag{172}
$$

$$
(\theta_{sub}^+, \phi_{cmp}^+) = \arg\min_{\theta, \phi} \mathcal{J}_T + L_{base} + L_{cmp} + L_{trace} + L_{value} + L_{halt} + L_{supp} + L_{probe}, \tag{173}
$$

$$
(\omega_{ren}^+, \kappa_{mem}^+, \kappa_{front}^+) = \arg\min_{\omega, \kappa} L_{render} + L_{mem} + L_{safe} + L_{front}, \tag{174}
$$

where certificate, split, dependence, context, semantic, source, rebuttal,
realization, frontier, and vector-safety gates are evaluated as detached
constraints during training and exactly during execution.

Training compute is allocated by measured marginal certified utility under
dual prices for scarce exact execution, dependence calibration, risk
calibration, and replay. The optimizer is over budget quanta, not over
static epoch counts:

$$
\begin{aligned}
\mathbf{B}^* = \arg\max_{\sum_i B_i \leq B}\, \widehat S_{ver}(\mathbf{B})\, & - \lambda_{overfit}\widehat{\mathrm{Gap}} - \lambda_{unsafe}\widehat{\mathrm{GateFail}} - \lambda_{leak}\widehat{\mathrm{SplitLeak}} \\
& - \lambda_{supp}\hat \epsilon_{support} - \lambda_{sub}\hat \epsilon_{sub} - \lambda_{dep}\hat \epsilon_{dep} - \lambda_{ctx}\hat \epsilon_{ctx} - \lambda_{sem}\hat \epsilon_{sem} - \lambda_{src}\hat \epsilon_{src} \\
& - \lambda_{rebut}\hat \epsilon_{rebut} - \lambda_{real}\hat \epsilon_{real} - \lambda_R^\top \widehat{\mathbf{r}}_{cum} - \lambda_{front}\widehat{\Delta\mathrm{Front}}^- - \lambda_{out}\widehat{\mathrm{OutDom}} - \lambda_{idle}\widehat{\mathrm{IdleExact}}.
\end{aligned} \tag{175}
$$

$$
\mathbf{B} = (B_{sub}, B_{model}, B_{data}, B_{ctx}, B_{sem}, B_{src}, B_{rebut}, B_{real}, B_{cal}, B_{front}, B_{search}, B_{ver}, B_{red}, B_{replay}, B_{exact}, B_{risk}). \tag{176}
$$

At an interior optimum the observed marginal returns satisfy the KKT
balancing condition

$$
\frac{\partial \widehat S_{ver}}{\partial B_i} - \lambda_i^{loss}\frac{\partial \hat \epsilon_i}{\partial B_i} - \lambda_R^\top \frac{\partial \widehat{\mathbf{r}}}{\partial B_i} - \lambda_{front}\frac{\partial \widehat{\Delta\mathrm{Front}}}{\partial B_i} = \lambda_0, \tag{177}
$$

with boundary modules receiving budget only when their estimated left
derivative is positive. This prevents cheap imitation, renderer likelihood,
or oversized neural pretraining from consuming exact execution and
calibration budget while the active certificate bottleneck remains
elsewhere. Teacher search is spent on posterior-boundary tasks, not
uniformly.

$$
e^* = \arg\max_e\, \frac{H(M, C^{ctx}, D^{sem}, R^{src}, R^{rebut}, \Omega^{real}, \mathrm{Dep}, \mathrm{Front} \mid D) - \mathbb{E}_o H(M, C^{ctx}, D^{sem}, R^{src}, R^{rebut}, \Omega^{real}, \mathrm{Dep}, \mathrm{Front} \mid D \cup \{(e, o)\}) + \mathbb{E}\Delta S_{ver} + \lambda_{reg}\Pr(\text{regression} \mid e)}{c(e) + c_{label}(e) + c_{exec}(e) + c_{ctx}(e) + c_{sem}(e) + c_{src}(e) + c_{rebut}(e) + c_{real}(e) + c_{dep}(e) + c_{front}(e) + \boldsymbol{\lambda}_R^\top \mathbf{r}(e)}. \tag{178}
$$

Efficiency constraints are structural. All modules read and write a
content-addressed trace factor graph, so a parse, source set, rebuttal set,
entailment witness, verifier result, realization check, dependence block,
frontier measurement, or risk estimate is computed once per hash and reused
by every head whose split policy permits it. Candidate execution is batched
by operator signature and e-class; context and semantic parses are cached
by discourse hash, time scope, reference class, and ambiguity class; source
and rebuttal retrieval are cached by atom, scope, freshness, and split;
realization checks are cached by atom-plan hash and block identity;
dependence residuals are cached by block identity; entailment/verifier
calls are batched and cached by $(v, \mu, a, x, \mathrm{scope}, \mathrm{split})$;
partial executions are memoized as
$(\mathrm{state, lower\ bound, upper\ bound, prune\ reason})$; failed
verifications create negative certificates for siblings in the e-graph when
sound; replay is prioritized by surprise, regression, staleness, support
loss, context loss, source staleness, rebuttal staleness, realization
drift, dependence leakage, frontier regression, and capability risk; and
exact stages are invoked on the uncertainty boundary where their expected
certificate or recall gain exceeds cost. External LLM proposals may be used
only as ablated teachers with logged budget and no certificate authority;
an unlogged external model is hidden compute. At inference, $\mathrm{VNS}_\theta$
may render style, paraphrase, and nonclaim glue, but every factual,
mathematical, executable, advice-like, private, or side-effecting token
must trace to certified context, atoms, parents, source/rebuttal records,
entailment witnesses, and realization witnesses.

The curriculum has gates. $C_0$: deterministic grids, strings, FSMs, small
graphs, arithmetic, finite-state tasks, and data transformations with exact
verifiers. $C_1$: hidden-schema splits, equality saturation, typed program
synthesis, and theorem-proving fragments. $C_2$: noisy/partial
observations, active tests, small causal worlds, and verifiable planning.
$C_3$: formal tools, agent workflows, tool use, authority, declassification,
prompt-injection, rollback, influence-risk, and conflict. $C_4$: controlled
dialogue/artifacts with context normal forms, semantic atom extraction,
source-grounded QA, contradiction search, entailment checking, round-trip
realization checking, calibrated uncertainty, and intent-entropy gates.
$C_5$: continual compression with replay-safe macro admission. Advancement
requires held-out certified utility, calibrated anytime false-pass bounds,
calibrated dependence residuals, positive frontier movement, sublinear
active-memory growth, bounded support, context, semantic, source, rebuttal,
realization, dependency, and substrate loss, vector-risk budget validity,
and zero critical gate violations in adversarial suites.

---

> **Implementation pointers.**
>
> - Data-split topology (137–139) → `vpm.training.splits`.
> - Trace DAG (140) and content-addressed factor graph →
>   `crates/vpm-ledger::trace_dag` (blake3-keyed nodes/edges).
> - Teacher posterior $p_*$ (141) → `vpm.training.teacher`.
> - Block-coordinate objective $L(\theta, \phi)$ and the per-head losses
>   (142–168) → `vpm.training.losses` (one file per head).
> - Loss-weight normalization by gradient scale (169) →
>   `vpm.training.weight_balancer`.
> - Block-coordinate optimizer with frozen audit labels (170–174) →
>   `vpm.training.coordinator`.
> - Budget allocator $\mathbf{B}^*$ and KKT balancing (175–177) →
>   `vpm.training.budget`.
> - Active-learning teacher allocator (178) → `vpm.training.active_query`.
> - Curriculum $C_0$–$C_5$ → `python/vpm/tasks/{c0,c1,c2,c3,c4,c5}/`.
> - GFlowNet trajectory-balance loss (166–167; refs [7][8]) →
>   `vpm.training.gflow`.
