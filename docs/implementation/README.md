# Implementation roadmap

This roadmap is *additional* to the architecture spec — the spec
([docs/architecture/](../architecture/)) is the source of truth for *what* is
being built. This document covers *the order in which we build it*.

The plan is anchored to **Criterion 1** (§9) and to the **VPM-0** minimal
implementation defined at the end of §9: anything we build must contribute
to clearing the failure clauses listed in Criterion 1.

## Ordering principle (from §8)

> Training is execution-first. Before neural learning, implement the typed
> DSL/bytecode, context normalizer, semantic-contract compiler,
> source/evidence retriever, rebuttal retriever, entailment checker,
> round-trip realization checker, batch executor, verifier registry,
> falsifier, trace DAG, e-class DAG, provenance graph, authority lattice,
> declassification checker, influence-risk checker, cumulative-risk
> ledger, memory admission/demotion, and renderers.

We respect this. Neural training (`vpm.training`) is M2 onwards, and only
on traces produced by the M0 / M1 executable system.

## Milestones

### M0 — Substrate-free executable kernel

Goal: every M0 component is exercised by the curriculum-$C_0$ tasks
*without any neural network involved*.

- **M0.1** — `vpm-core`: typed atoms, modes, contracts $\Gamma$, ledger
  schema $\Lambda$ (§1).
- **M0.2** — `vpm-ledger`: append-only ledger, content-addressed trace DAG,
  provenance graph, blake3 hashing (§1, §8).
- **M0.3** — `vpm-dsl`: typed bytecode + deterministic executor (§8).
- **M0.4** — `vpm-egraph`: e-graph + canonicalization wrapping `egg` with
  reversible-witness or support-bound checks (§5 eqs. 84–85; §7 ref. [6]).
- **M0.5** — `vpm-authority`: authority lattice, declassification proofs,
  vector-risk ledger (§6).
- **M0.6** — `vpm-verify`: verifier registry, sequence-valid e-values
  (§2 eqs. 21–29; refs. [12][13]), falsifier, action gate (§6 eq. 113).
- **M0.7** — `vpm.compiler`: parser + canonicalizer that produces typed
  $\mathfrak{n}$ posteriors for $C_0$ (§4 eqs. 50–51).
- **M0.8** — `vpm.evaluation`: metric harness covering Criterion 1 clauses
  expressible at this stage; `tests/integration/sanity/` skeleton from
  Appendix B.

Exit gate: full $C_0$ curriculum runs end-to-end with exact verifiers and
the action gate; `pytest -m sanity` passes; no neural module is imported.

### M1 — Language gates

- **M1.1** — `vpm.language.context`: context normalizer with calibrated
  miss bounds (§4 eqs. 55–58).
- **M1.2** — `vpm.language.semantic`: semantic contractizer (§4 eqs. 60–63).
- **M1.3** — `vpm.retrieval.{source, rebuttal}`: support / rebuttal
  retrievers with calibrated bounds (§4 eqs. 65–69).
- **M1.4** — `vpm.verifiers.entailment`: entailment witness checker
  (§4 eqs. 70–73).
- **M1.5** — `vpm.language.realization`: independent round-trip realization
  checker (§4 eqs. 75–77).
- **M1.6** — `vpm.language.render` + `vpm-verify::gate_nl` (§4 eq. 80–81;
  Proposition 2).

Exit gate: $C_1$ + $C_4$ controlled-dialogue subsets pass with
`Gate_NL` enforced; no certified-mode atom is rendered without entailment,
source, rebuttal, and round-trip witnesses.

### M2 — Neural substrate ($\mathrm{VNS}_\theta$)

- **M2.1** — `vpm.substrate.encoder`: typed event hypergraph (§3 eq. 34).
- **M2.2** — `vpm.substrate.ssm`: selective SSM block (§3 eq. 36; ref [4]).
- **M2.3** — `vpm.substrate.slots`: slot binding (§3 eqs. 37–38; ref [9]).
- **M2.4** — `vpm.substrate.projection`: executable projection (§3 eq. 39).
- **M2.5** — `vpm.substrate.losses`: $L_{base}$ family (§3 eqs. 42–46) and
  substrate-recall calibration (§3 eq. 47).
- **M2.6** — `vpm.training.probes`: critical-edge probes (§3 eqs. 48–49).

Exit gate: substrate trained on M0/M1 traces hits the
`Cert_sub`/`Cert_crit` thresholds on a held-out audit split; ablations
in §9 show non-trivial degradation when each component is removed.

### M3 — Compiler, support guard, mechanism cell

- **M3.1** — `vpm.compiler.posterior` + `vpm.compiler.score_head`
  (§5 eqs. 83, 88).
- **M3.2** — `vpm.infer.cell`: typed-message recurrent update
  (§5 eqs. 89–91, 96).
- **M3.3** — `vpm.infer.support_guard` + `crates/vpm-verify::support_guard`
  (§5 eqs. 92–95).
- **M3.4** — `vpm.infer.staging`: $\iota \to \sigma \to \pi \to \eta$ stage
  scheduler (§5 eqs. 97–98).
- **M3.5** — `vpm.infer.test_select` + `vpm.infer.halt` (§5 eqs. 100–101).

Exit gate: $C_2$ active-test tasks run; support guard rejects /
rehydrates pruned alternatives at the calibrated rate.

### M4 — Memory + compression

- **M4.1** — `vpm.memory.{active, archival, library}` (§7 eqs. 118–120).
- **M4.2** — `vpm.memory.frontier` + `crates/vpm-verify::eb_seq`
  (§7 eqs. 121–128).
- **M4.3** — `crates/vpm-egraph::equiv_cert` (§7 eq. 124).
- **M4.4** — `vpm.memory.{admit, active_set}` (§7 eqs. 129–131).

Exit gate: compression phase transition observed on $C_5$ replay
(§9, last paragraph).

### M5 — Training system

- **M5.1** — `vpm.training.splits` (§8 eqs. 137–139).
- **M5.2** — `vpm.training.teacher` (§8 eq. 141).
- **M5.3** — `vpm.training.losses.*` for every loss in §8 eqs. 142–168.
- **M5.4** — `vpm.training.weight_balancer` (§8 eq. 169).
- **M5.5** — `vpm.training.coordinator` (§8 eqs. 170–174).
- **M5.6** — `vpm.training.budget` (§8 eqs. 175–177).
- **M5.7** — `vpm.training.gflow` (§8 eqs. 166–167; refs [7][8]).

Exit gate: full block-coordinate training run completes; KKT balancing
shows non-zero shadow prices on every budget channel; Criterion 1 ablations
produce the expected regressions.

### M6 — Curriculum + adversarial suites

- All of $C_0$–$C_5$ exercised end-to-end with red-team replay
  (§8 paragraph on $\mathcal{D}_{red}$; Appendix B).
- `C3` tool-use probes run through a deterministic allowlisted sandbox
  with authority, risk, and primitive-type checks before any tool compute is
  charged.
- `C3` rollback probes maintain a cumulative risk ledger and apply rollback
  credits only when they are within cap and backed by monitor/restoration
  certificates.
- `C2` noisy causal-world probes estimate intervention effects from clean
  samples while bounded noisy observations remain in the audit trace.
- `C2` multi-step planning probes emit bounded grid action/state traces and
  shortest-path certificates around blocked cells.
- `C1` multi-step synthesis probes select typed two-step programs from
  candidate programs and verify their final value against the hidden-schema
  target.
- `C1` theorem fragments forward-chain compact propositional implications
  with explicit modus-ponens proof traces.
- `C4` dialogue probes expose calibrated uncertainty as a strict witness
  threshold: any missing source, rebuttal, entailment, or realization witness
  forces refusal.
- `C4` default dialogue probes are fed by the audited open-domain retriever,
  which certifies only unique corpus matches and routes unresolved prompts
  away from certified mode.
- `C5` macro replay feeds outcomes incrementally through the online frontier
  estimator before active-memory admission.
- `C5` macro replay uses a cross-stage scheduler that combines the candidate
  seed set with verifier-backed C0/C1/C2/C3 replay batches for the target
  operation, including expected C3 policy rejections.
- Runtime metadata includes an explicit `M6` stage for red-team replay,
  ablations, hard-domain probes, and external-LLM task export. The release
  audit treats any `StageSpec.blockers` as release blockers.

Exit gate: zero critical gate violations in adversarial suites; held-out
certified utility positive; all Criterion 1 failure clauses unfired.

## Test discipline

- Unit tests for everything algorithmic (`tests/unit/`).
- Property tests for invariants (`tests/integration/invariants/`,
  Hypothesis + `proptest`).
- Behavior-level sanity kernel (`tests/integration/sanity/`,
  Appendix B).
- Failure-mode regression suite (`tests/integration/failure_modes/`,
  Criterion 1 clause-by-clause).
- Benchmarks (`benches/`) — `criterion` for Rust and `pytest-benchmark`
  for Python; budgets in `vpm.training.budget` use these as ground
  truth for cost terms.

## Architectural decisions

- [`egglog-migration.md`](./egglog-migration.md) — when, and only when,
  the e-graph backend should move from `egg` to `egglog`. **Decision
  (May 2026): stay on `egg`.**

## Out of scope (for now)

- Distributed training.
- A user-facing dialogue product. VPM-0 is a research kernel, not an
  application.
- Non-curriculum data sources beyond the audited ones in §8 eq. 137.
- External LLMs as anything other than ablated teachers with logged
  budgets and zero certificate authority (§8, paragraph on hidden
  compute).
- Same-budget external LLM comparisons are accepted only through
  `VPM_LLM_BASELINE_JSON`; the JSON must include `solve_rate` and
  `compute_units`, and `eval-baselines` marks it invalid if the compute
  exceeds the matched VPM budget.
- Use `vpm export-llm-baseline tasks.jsonl --limit N` to write the exact
  held-out C1 prompts for an external LLM run, then score JSONL predictions
  with `vpm score-llm-baseline predictions.jsonl --limit N --output
  llm-baseline.json`. The prediction JSONL must provide `task_id`,
  `operation`, and `compute_units` per task; the exported prompts do not
  include gold operations.
- To run those prompts through OpenAI reproducibly, set `OPENAI_API_KEY`
  and run `vpm run-openai-llm-baseline tasks.jsonl predictions.jsonl
  --kind c1 --model MODEL`. One Responses API call is logged as one compute
  unit, so the scorer still enforces the matched budget before writing
  reusable baseline JSON.
- To produce both release-audit LLM artifacts in one run, use
  `vpm run-openai-release-baselines artifacts/openai-baselines --model
  MODEL`; it exports C1 and hard-domain tasks, runs the OpenAI-compatible
  baseline, scores both outputs, writes both JSON files, and prints the
  `export VPM_LLM_BASELINE_JSON=...` lines needed by `eval-release`.
- For held-out hard domains, use `vpm export-hard-llm-baseline
  hard-tasks.jsonl`, then `vpm score-hard-llm-baseline
  hard-predictions.jsonl --output hard-llm-baseline.json`. The hard-domain
  prediction JSONL must provide `task_id`, `answer`, and `compute_units`;
  exported prompts include evidence but not gold answers. Set
  `VPM_HARD_LLM_BASELINE_JSON` to the scored hard-domain baseline JSON when
  running `vpm eval-release`.
- The same OpenAI runner accepts `--kind hard` for hard-domain task JSONL.
- `vpm eval-release --json` runs the objective-facing release audit. It is
  expected to stay `passed=false` until the same-budget external LLM
  baselines are actually configured and executed.
- `vpm eval-objective --json` wraps `eval-release` with a prompt-to-artifact
  checklist for the active completion objective. Use it before claiming
  completion so remaining release blockers are tied back to the concrete
  architecture, training, baseline, hard-domain, and CI requirements they
  affect.

## Known risks

### Python 3.14 wheel coverage (as of 2026-05-04)

The project requires Python 3.14 (`pyproject.toml:10`). Several declared
deps lack 3.14 wheels on PyPI today:

- `torch` 2.11.x (also under CVE-2026-24747 — patched torch 2.11.z
  pending; floor stays at `>=2.11` until then per the TODO at
  `pyproject.toml:33-36`)
- `faiss-cpu` (optional, `[retrieve]` extra)
- `sentence-transformers` (optional, `[retrieve]` extra)
- `mkdocs-material` (docs group)
- `pre-commit`, `pyright`, `deptry` (dev group)

The maturin-built extension is fine: PyO3 0.28 + abi3-py314 work end
to end, and the smoke import in `ci.yml` (`from vpm._native import …`)
proves it.

**Mitigation strategy:**
1. Keep watching upstream releases (`pip-audit` + `cargo deny` already
   cover security drift; `dependabot.yml` covers maintenance drift).
2. If M2 (substrate, requires `torch`) is blocked at the start of the
   milestone by missing 3.14 wheels for `torch`, fall back to Python
   3.13 — single change in `pyproject.toml:10` (and the matching
   `uv python install` lines in `.github/workflows/ci.yml`).
3. **Do not pre-emptively pin to 3.13.** Python 3.14's free-threaded
   mode matters for the substrate's typed event hypergraph
   (parallelism over slots, §3 eq. 37); it's worth waiting on.

### LOC budgets

The repo enforces per-file and per-folder line caps via
`scripts/check-loc.py` (configured in `.loc-budget.toml`). This exists
because we collaborate heavily with AI assistants; their characteristic
failure modes (regenerating an 800-line file to add 12 lines; writing
200-line functions where 4×50 would do) are detectable purely by raw
line count, and a guideline alone won't constrain an autonomous AI's
output. CI runs the checker in hard mode (`LOC_HARD=1`); locally it's
advisory. Companion tools: `clippy::too_many_lines` (Rust),
`PLR0915` in ruff (Python).
