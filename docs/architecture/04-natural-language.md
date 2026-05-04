# §4 Natural-language contractization, rebuttal, and semantic realization

Natural language enters through a posterior over conversation normal forms,
not through a privileged sequence model. For dialogue $U_{0:t}$,

$$
q_\psi(\mathfrak{n} \mid U_{0:t}, H, \Gamma) = \{(\mathfrak{n}_k, w_k, b_k^{ctx}, b_k^{sem}, b_k^{real})\}_{k \leq K_n}, \quad \sum_k w_k = 1, \tag{50}
$$

where

$$
\mathfrak{n} = (\iota,\, D,\, C^{ctx},\, \mathcal{A}^{sem},\, \Delta,\, \mathcal{B},\, \mathcal{Q}^{ask},\, \mathcal{R}^{real}). \tag{51}
$$

Here $\iota$ is intent, $D$ discourse referents, $C^{ctx}$
temporal/deictic/granularity/definition commitments, $\mathcal{A}^{sem}$
semantic atoms, $\Delta$ dependencies, $\mathcal{R}^{disc}$ dialogue-state
transitions, $\mathcal{B}$ scoped background assumptions, $\mathcal{Q}^{ask}$
admissible clarification questions, and $\mathcal{R}^{real}$ realization
constraints. Each atom is

$$
a = (m,\, \varphi,\, \Omega,\, \tau,\, \mathrm{refs},\, \mathrm{deps},\, V_a,\, R_a,\, \alpha_a,\, \lambda_a,\, \kappa_a,\, \gamma_a,\, \chi_a,\, v_a), \tag{52}
$$

where $m \in \mathcal{M}_{mode}$ is the speech-act/executable mode, $\varphi$
the typed proposition or executable object, $\Omega$ its scope, $\tau$ its
type, $\kappa_a$ its context key, $\gamma_a$ its granularity/vagueness class,
$\chi_a$ its contradiction-search obligation, and $v_a$ its realization
class. The mode set is partitioned as

$$
\mathcal{M}_{cert} = \{\text{fact, math, code, private, action, quote, advice, definition}\}, \tag{53}
$$

$$
\mathcal{M}_{soft} = \{\text{style, creative, preference, uncertainty, question, refusal, social, transition}\}. \tag{54}
$$

Atoms in $\mathcal{M}_{cert}$ require entailment, source, rebuttal, authority,
privacy, context, support, substrate, family-spending, and realization
certificates. Atoms in $\mathcal{M}_{soft}$ require mode separation, influence
safety, privacy, and no hidden certified-mode atom; they do not become
factual claims by fluent wording.

Context loss is separated from semantic loss because many open-ended language
failures are not false propositions but unresolved references, shifting
time, implicit comparison classes, vague predicates, or missing definitions.
Let $C_*^{ctx}$ be the audited context normal form and $\preceq_{ctx}$ the
refinement order induced by $\Pi_{ctx}$. Define

$$
H_{ctx}(T) = H_q(\iota,\, D,\, C^{ctx} \mid U, H, \Gamma), \tag{55}
$$

$$
\epsilon_{ctx}(T) = \min\Big\{1,\, \mathrm{UCB}_{1-\delta_{ctx}}^{seq}\big(\Pr\{\mathfrak{n}^* \not\preceq_{ctx} C^{ctx}\ \text{for all retained}\ k\};\, \mathcal{D}_{ctx}^{audit}\big) + b^{ctx} + D_{shift}^{ctx}(T) + D_{select}^{ctx}(T)\Big\}, \tag{56}
$$

$$
\mathrm{Cert}_{ctx}(T) = [-\log(\epsilon_{ctx}(T) + \epsilon_0)]_+, \tag{57}
$$

$$
\mathrm{CtxOK}(T,\Gamma) = \mathbf{1}\{H_{ctx} \leq h_{ctx} \,\wedge\, \epsilon_{ctx} \leq \epsilon_{ctx,max} \,\wedge\, \mathrm{RefOK}(D) \,\wedge\, \mathrm{ScopeOK}(C^{ctx},\Gamma) \,\wedge\, \mathrm{GranOK}(\mathcal{A}^{sem})\}. \tag{58}
$$

For a certified-mode atom,

$$
\mathrm{Vague}(a) = 1 \Rightarrow \mathrm{DefOK}(a, C^{ctx}, \Gamma) \vee m(a) \in \{\text{question, uncertainty, refusal}\}; \tag{59}
$$

undefined vague predicates therefore force a definition, scope reduction,
question, or uncertainty span. They cannot be silently compiled into a crisp
factual answer.

Semantic contractization is admissible only when the posterior has retained
the plausible meanings and the retained meanings refine the context
commitments:

$$
H_{sem}(T) = H_q(\iota,\, D,\, \mathcal{A}^{sem},\, \Delta \mid U, H, \Gamma), \tag{60}
$$

$$
\epsilon_{sem}(T) = \min\Big\{1,\, \mathrm{UCB}_{1-\delta_{sem}}^{seq}\big(\Pr\{\mathfrak{n}^* \notin \mathrm{supp}_{K_n} q_\psi\};\, \mathcal{D}_{sem}^{audit}\big) + b^{sem} + D_{shift}^{sem}(T) + D_{select}^{sem}(T)\Big\}, \tag{61}
$$

$$
\mathrm{Cert}_{sem}(T) = [-\log(\epsilon_{sem}(T) + \epsilon_0)]_+, \tag{62}
$$

$$
\mathrm{SemOK}(T,\Gamma) = \mathbf{1}\{H_{sem} \leq h_{sem} \,\wedge\, \epsilon_{sem} \leq \epsilon_{sem,max} \,\wedge\, \mathrm{CtxOK}(T,\Gamma) \,\wedge\, \mathrm{ModeSep}(\mathcal{A}^{sem}) \,\wedge\, \mathrm{DepOK}(\Delta)\}. \tag{63}
$$

If $\mathrm{SemOK} = 0$ or $\mathrm{CtxOK} = 0$, the permitted actions are to
ask, narrow, or produce an explicitly scoped uncertainty/refusal span; no
factual answer is rendered. The clarification policy maximizes certified
entropy reduction rather than conversational fluency:

$$
q^* = \arg\max_{q \in \mathcal{Q}_{ask}} \frac{H_{ctx} + H_{sem} - \mathbb{E}_o[H_{ctx} + H_{sem}](U_{0:t} \cup (q, o)) + \lambda_c \mathbb{E}_o \Delta\,\mathrm{DomOK} - \lambda_A \mathrm{Annoy}(q)}{c(q) + \lambda_H H_{bits}(q)}. \tag{64}
$$

For a factual or advice-like atom $a$, retrieval must search both for support
and for material rebuttal. Let $\mathcal{S}_{a,+}^*$ be minimal sufficient
support sets and $\mathcal{S}_{a,-}^*$ material contradiction or
scope-defeater sets under $R_a, \Omega_a, C^{ctx}$. For retrieved support
$R_t^+(a)$ and rebuttal set $R_t^-(a)$,

$$
\mathrm{miss}_{src}(a) = \mathbf{1}\{\forall S \in \mathcal{S}_{a,+}^* : S \not\subseteq R_t^+(a)\}, \tag{65}
$$

$$
\mathrm{miss}_{rebut}(a) = \mathbf{1}\{\exists S \in \mathcal{S}_{a,-}^* : S \subseteq R_t^-(a)\}, \tag{66}
$$

$$
\epsilon_{src}(a) = \min\Big\{1,\, \mathrm{UCB}_{1-\delta_{src}}^{seq}\big(\Pr\{\mathrm{miss}_{src}(a) = 1\};\, \mathcal{D}_{src}^{audit}\big) + D_{stale}^{src}(a) + D_{select}^{src}(a) + D_{shift}^{src}(a)\Big\}, \tag{67}
$$

$$
\epsilon_{rebut}(a) = \min\Big\{1,\, \mathrm{UCB}_{1-\delta_{rebut}}^{seq}\big(\Pr\{\mathrm{miss}_{rebut}(a) = 1\};\, \mathcal{D}_{rebut}^{audit}\big) + D_{stale}^{rebut}(a) + D_{select}^{rebut}(a) + D_{shift}^{rebut}(a)\Big\}, \tag{68}
$$

$$
\mathrm{Cert}_{src}(a) = [-\log(\epsilon_{src}(a) + \epsilon_0)]_+, \quad \mathrm{Cert}_{rebut}(a) = [-\log(\epsilon_{rebut}(a) + \epsilon_0)]_+. \tag{69}
$$

For exact internal objects, such as arithmetic traces or proof terms, $R_a$
may set $\epsilon_{src}(a) = \epsilon_{rebut}(a) = 0$ and shift the burden to
execution and proof verifiers. For open-domain facts, source recall plus
contradiction recall is mandatory: retrieved text proposes hypotheses, and
absence of visible contradiction is not evidence unless the
contradiction-recall bound is small.

Entailment is explicit. Given ledger parents $P_a$, an entailment witness
$e_a$, and atom $a = (m, \varphi_a, \ldots)$,

$$
\mathrm{Cert}_{ent}(a) = \mathrm{Cert}\big(\mathrm{Ent}(P_a, e_a, C^{ctx}) \Rightarrow \varphi_a\big), \tag{70}
$$

$$
\mathrm{RebutOK}(a) = \mathbf{1}\{m(a) \notin \{\text{fact, advice, quote}\} \vee \epsilon_{rebut}(a) \geq \theta_{rebut}(m)\}, \tag{71}
$$

$$
\mathrm{Cert}_{atom}(a) = \min\Big\{\mathrm{Cert}_{ctx},\, \mathrm{Cert}_{sem},\, \mathrm{Cert}_{ent}(a),\, \mathrm{Cert}_{src}(a),\, \mathrm{Cert}_{rebut}(a),\, \min_{p \in P_a}\mathrm{Cert}(p),\, \mathrm{Cert}_{support},\, \mathrm{Cert}_{sub},\, \mathrm{Cert}_{dom},\, \mathrm{Cert}_{auth},\, \mathrm{Cert}_{priv}\Big\}, \tag{72}
$$

$$
\mathrm{EntOK}(a, Z, \Gamma) = \mathbf{1}\{P_a \neq \emptyset \,\wedge\, \mathrm{Cert}_{ent}(a) \geq \theta_{ent}(m) \,\wedge\, \mathrm{RebutOK}(a) \,\wedge\, \mathrm{Cert}_{atom}(a) \geq V_a\}. \tag{73}
$$

Paraphrase, synthesis, advice, and summarization are proof obligations over
atoms. A source can be quoted only if span alignment and scope are
certified:

$$
\mathrm{CiteOK}(a) = \mathbf{1}\{m = \text{quote} \Rightarrow \mathrm{Align}(y_a, P_a) \wedge \mathrm{NoContextLoss}(y_a, P_a, \Omega_a, C^{ctx})\}. \tag{74}
$$

Rendering is a semantic realization problem with an independent round-trip
checker. For a candidate text $y$ intended to realize atom multiset $A_y$,
let $\mathrm{Atom}(y)$ be the atom multiset extracted from $y$ by a checker
not used to generate $y$. With $\preceq_{sem}$ the semantic refinement order,

$$
\epsilon_{real}(y) = \min\Big\{1,\, \mathrm{UCB}_{1-\delta_{real}}^{seq}\big(\Pr\{\neg(\mathrm{Atom}(y) \preceq_{sem} A_y) \vee \mathrm{Hidden}_{cert}(y)\};\, \mathcal{D}_{real}^{audit}\big) + b^{real} + D_{shift}^{real}(y) + D_{select}^{real}(y)\Big\}, \tag{75}
$$

$$
\mathrm{Cert}_{real}(y) = [-\log(\epsilon_{real}(y) + \epsilon_0)]_+, \tag{76}
$$

$$
\mathrm{RealOK}(y, A_y, Z, \Gamma) = \mathbf{1}\{\epsilon_{real}(y) \leq \epsilon_{real,max} \wedge \mathrm{Atom}(y) \preceq_{sem} A_y \wedge \neg\,\mathrm{Hidden}_{cert}(y) \wedge \Pi_{real}(y, A_y, \Gamma)\}. \tag{77}
$$

Missing evidence is not silently converted into fluent prose:

$$
\mathrm{Def}(y) = \{a \in \mathrm{Atom}(y) : m(a) \in \mathcal{M}_{cert} \wedge \mathrm{EntOK}(a, Z, \Gamma) = 0 \vee \mathrm{CiteOK}(a) = 0 \vee \mathrm{RebutOK}(a) = 0\}, \tag{78}
$$

$$
\mathrm{ModeOK}(y, Z, \Gamma) = \mathbf{1}\{\forall a \in \mathrm{Atom}(y) : m(a) \in \mathcal{M}_{cert} \vee \mathrm{NoHiddenCertAtom} \wedge \Pi_{style}(a, \Gamma) \wedge \Pi_{infl}(a, \Gamma)\}, \tag{79}
$$

$$
\begin{aligned}
\mathrm{Gate}_{NL}(y, Z, \Gamma) = \mathbf{1}\{\,&\mathrm{SemOK} \wedge \mathrm{CtxOK} \wedge \mathrm{ModeOK}(y, Z, \Gamma) \wedge \mathrm{RealOK}(y, A_y, Z, \Gamma) \wedge \mathrm{Def}(y) = \emptyset \\
& \wedge \Pi_{priv}(y, \lambda(y), \Gamma) \wedge \alpha(y) \preceq \alpha(\Gamma) \wedge \mathrm{CumRisk}(y) \leq B_R(\Gamma)\,\}.
\end{aligned} \tag{80, 81}
$$

A best-effort answer with incomplete coverage is represented as
$y = y_{cert} \oplus y_{unc} \oplus y_{soft}$ where $\mathrm{Atom}(y_{cert})$
passes the gate, $\mathrm{Atom}(y_{soft}) \cap \mathcal{M}_{cert} = \emptyset$,
and every missing certified-mode atom $a$ is replaced by an uncertainty span
$u(a)$ with

$$
\mathrm{Cert}(u(a)) = \mathrm{Cert}\{\text{no admissible certificate for}\ a\ \text{was found within}\ B,\, \Pi_{ctx},\, \Pi_{src},\, \Pi_{rebut},\, \Pi_{real},\, \Pi_{dom}\}. \tag{82}
$$

Thus broad language is not solved by making a larger decoder; it is made
admissible only by retaining plausible contexts, searching for support and
defeaters, and proving that the surface text realizes only the certified
atom plan.

**Proposition 2** *(No unsupported or hidden factual rendering).* Assume the
context normalizer, semantic checker, source retriever, rebuttal retriever,
and round-trip realization checker have calibrated false-negative bounds
$\epsilon_{ctx}, \epsilon_{sem}, \epsilon_{src}, \epsilon_{rebut}, \epsilon_{real}$,
and entailment certificates are family-valid. If
$\mathrm{Gate}_{NL}(y, Z, \Gamma) = 1$, then every certified-mode atom in $y$
has scoped certificate at least its required threshold and no extra
certified-mode atom is hidden in $y$ except on an event of probability at
most $\delta_{ctx} + \delta_{sem} + \delta_{src} + \delta_{rebut} + \delta_{real} + e^{-u}\alpha_\Gamma$
at excess level $u$.

*Proof.* A certified-mode atom can pass $\mathrm{Gate}_{NL}$ only through
$\mathrm{EntOK}$, $\mathrm{RebutOK}$, $\mathrm{RealOK}$, and empty
$\mathrm{Def}(y)$. The atom certificate is a minimum over context, semantic,
entailment, source, rebuttal, parent, support, substrate, domain, authority,
and privacy caps. Hidden factual text is excluded by the independent
round-trip event. Context, semantic, source, rebuttal, and realization misses
contribute their calibration failure events; entailment false passes are
controlled by the family-valid e-value theorem. Union over rendered atoms
using the ledgered family budget gives the stated bound. $\square$

---

> **Implementation pointers.**
>
> - Conversation normal forms (eqs. 50–52) live in
>   `python/vpm/language/normal_form.py`; the mode partition
>   $\mathcal{M}_{cert}$ / $\mathcal{M}_{soft}$ (53–54) is an enum in
>   `vpm-core::mode`.
> - Context normalizer ($H_{ctx}, \epsilon_{ctx}$, 55–58) →
>   `vpm.language.context`.
> - Semantic contractizer ($H_{sem}, \epsilon_{sem}$, 60–63) →
>   `vpm.language.semantic`.
> - Source / rebuttal retrievers (65–69) → `vpm.retrieval.{source,rebuttal}`.
> - Entailment + atom certificate (70–73) → `vpm.verifiers.entailment` plus
>   `crates/vpm-verify::cert_atom`.
> - Independent round-trip realization checker (75–77) →
>   `vpm.language.realization` (the checker model is split-clean from the
>   renderer).
> - Final NL gate $\mathrm{Gate}_{NL}$ (80–81) is the public boundary in
>   `crates/vpm-verify::gate_nl`.
