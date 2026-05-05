"""Prompt-to-artifact completion audit for the active VPM objective."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vpm.evaluation.release_audit import (
    ReleaseCriterion,
    ReleaseReadinessReport,
    evaluate_release_readiness,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OBJECTIVE = (
    "finish compress as a full end-to-end VPM implementation, moving beyond "
    "the current C0/C1 prototype into the complete M0-M6 architecture with "
    "substrate, compiler, support guard, memory/compression, training, "
    "calibrated gates, adversarial suites, clean CI, held-out evaluations, "
    "ablations, and matched LLM, transformer, SSM, and program-synthesis "
    "baselines on hard domains"
)


@dataclass(frozen=True)
class ObjectiveChecklistItem:
    """One objective requirement mapped to concrete repo evidence."""

    requirement_id: str
    requirement: str
    success_criteria: tuple[str, ...]
    artifacts: tuple[str, ...]
    verification_commands: tuple[str, ...]
    release_criteria: tuple[str, ...]
    passed: bool
    evidence: tuple[str, ...]
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly checklist item."""
        return {
            "requirement_id": self.requirement_id,
            "requirement": self.requirement,
            "success_criteria": self.success_criteria,
            "artifacts": self.artifacts,
            "verification_commands": self.verification_commands,
            "release_criteria": self.release_criteria,
            "passed": self.passed,
            "evidence": self.evidence,
            "blockers": self.blockers,
        }


@dataclass(frozen=True)
class ObjectiveAuditReport:
    """Objective completion report with release readiness attached."""

    objective: str
    checklist: tuple[ObjectiveChecklistItem, ...]
    release_readiness: ReleaseReadinessReport

    @property
    def passed(self) -> bool:
        """True when every objective item and release criterion passes."""
        return self.release_readiness.passed and all(item.passed for item in self.checklist)

    @property
    def blockers(self) -> tuple[str, ...]:
        """Deduplicated blockers across objective and release audits."""
        blockers = [blocker for item in self.checklist for blocker in item.blockers] + list(
            self.release_readiness.blockers
        )
        return tuple(dict.fromkeys(blockers))

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly objective audit report."""
        return {
            "objective": self.objective,
            "passed": self.passed,
            "blockers": self.blockers,
            "checklist": [item.to_dict() for item in self.checklist],
            "release_readiness": self.release_readiness.to_dict(),
        }


def evaluate_objective_completion(limit: int = 0) -> ObjectiveAuditReport:
    """Evaluate the active objective as an explicit prompt-to-artifact checklist."""
    release = evaluate_release_readiness(limit=limit)
    criteria = {criterion.criterion_id: criterion for criterion in release.criteria}
    return ObjectiveAuditReport(
        objective=OBJECTIVE,
        checklist=(
            objective_item(
                criteria,
                requirement_id="m0_m6_architecture",
                requirement="Complete runtime-visible M0-M6 VPM architecture.",
                success_criteria=(
                    "C0, C1, C2, C3, C4, C5, and M6 stages are registered in order.",
                    "Every stage has executable implemented components and no stage blockers.",
                    "M6 red-team replay covers failures, ablations, and hard-domain tasks.",
                ),
                artifacts=(
                    "python/vpm/tasks/__init__.py",
                    "python/vpm/tasks/c0/__init__.py",
                    "python/vpm/tasks/c1/__init__.py",
                    "python/vpm/tasks/c2/__init__.py",
                    "python/vpm/tasks/c3/__init__.py",
                    "python/vpm/tasks/c4/__init__.py",
                    "python/vpm/tasks/c5/__init__.py",
                    "python/vpm/tasks/hard_domains.py",
                    "tests/integration/sanity/test_m6_red_team.py",
                ),
                verification_commands=("uv run python -m vpm stages --json",),
                release_criteria=("stages_m0_m6", "m6_red_team"),
            ),
            objective_item(
                criteria,
                requirement_id="substrate_compiler_support_guard",
                requirement="Real substrate, compiler, and support guard are implemented.",
                success_criteria=(
                    "Substrate encoders, SSM, slot binding, projection, and losses exist.",
                    "Compiler posterior and score head produce support-preserving candidates.",
                    "Support guard rehydrates unsafe pruning and is covered by sanity tests.",
                ),
                artifacts=(
                    "python/vpm/substrate/__init__.py",
                    "python/vpm/substrate/encoder.py",
                    "python/vpm/substrate/ssm.py",
                    "python/vpm/substrate/slots.py",
                    "python/vpm/substrate/projection.py",
                    "python/vpm/substrate/losses.py",
                    "python/vpm/compiler/__init__.py",
                    "python/vpm/compiler/posterior.py",
                    "python/vpm/compiler/score_head.py",
                    "python/vpm/infer/support_guard.py",
                    "tests/integration/sanity/test_substrate_controls.py",
                    "tests/integration/sanity/test_compiler_posterior.py",
                ),
                verification_commands=(
                    "uv run pytest tests/integration/sanity/test_substrate_controls.py "
                    "tests/integration/sanity/test_compiler_posterior.py -q",
                ),
                release_criteria=("stages_m0_m6",),
            ),
            objective_item(
                criteria,
                requirement_id="memory_compression_training",
                requirement="Memory/compression and trainable research-system controls exist.",
                success_criteria=(
                    "C5 replay-safe macro admission and frontier estimation are executable.",
                    "Training splits, teacher posterior, losses, budget, active query, and "
                    "scheduler controls are implemented.",
                    "Example training entry points can train and reload C0/C1 prototypes.",
                ),
                artifacts=(
                    "python/vpm/memory/admit.py",
                    "python/vpm/memory/active.py",
                    "python/vpm/memory/frontier.py",
                    "python/vpm/memory/library.py",
                    "python/vpm/evaluation/compression.py",
                    "python/vpm/evaluation/phase_transition.py",
                    "python/vpm/evaluation/saturation.py",
                    "python/vpm/training/__init__.py",
                    "python/vpm/training/prototype.py",
                    "python/vpm/training/losses/registry.py",
                    "python/vpm/training/coordinator.py",
                    "python/vpm/training/budget.py",
                    "python/vpm/training/active_query.py",
                    "examples/vpm0/train.py",
                    "tests/integration/sanity/test_memory_controls.py",
                    "tests/integration/sanity/test_training_controls.py",
                    "tests/integration/sanity/test_training_losses.py",
                    "tests/integration/sanity/test_training_scheduler.py",
                ),
                verification_commands=(
                    "uv run pytest tests/integration/sanity/test_memory_controls.py "
                    "tests/integration/sanity/test_training_controls.py "
                    "tests/integration/sanity/test_training_losses.py "
                    "tests/integration/sanity/test_training_scheduler.py -q",
                    "uv run python examples/vpm0/train.py",
                ),
                release_criteria=("stages_m0_m6",),
            ),
            objective_item(
                criteria,
                requirement_id="calibrated_gates_adversarial_suites",
                requirement="Calibrated gates and adversarial suites are executable.",
                success_criteria=(
                    "Recall, dependence, entailment, uncertainty, and open-domain gates run.",
                    "Failure-mode, ablation, and red-team suites report blockers.",
                    "Authority, tool-use, rollback, and source-grounding probes are covered.",
                ),
                artifacts=(
                    "python/vpm/retrieval/calibration.py",
                    "python/vpm/verifiers/dependence.py",
                    "python/vpm/verifiers/entailment.py",
                    "python/vpm/evaluation/failure_modes.py",
                    "python/vpm/evaluation/ablations.py",
                    "python/vpm/evaluation/open_domain.py",
                    "python/vpm/evaluation/red_team.py",
                    "python/vpm/tasks/c3/tools.py",
                    "python/vpm/tasks/c3/rollback.py",
                    "tests/integration/failure_modes/test_criterion1.py",
                    "tests/integration/sanity/test_recall_shift.py",
                    "tests/integration/sanity/test_dependence_shift.py",
                    "tests/integration/sanity/test_entailment_attacks.py",
                    "tests/integration/sanity/test_open_domain_ambiguity.py",
                    "tests/integration/sanity/test_c3_tool_sandbox.py",
                    "tests/integration/sanity/test_c3_rollback_ledger.py",
                ),
                verification_commands=(
                    "uv run python -m vpm eval-failures --json",
                    "uv run python -m vpm eval-ablations --json",
                    "uv run python -m vpm eval-red-team --json",
                ),
                release_criteria=("m6_red_team",),
            ),
            objective_item(
                criteria,
                requirement_id="heldout_baselines_matched_budgets",
                requirement=(
                    "Held-out evaluations and ablations compare honestly against LLM, "
                    "transformer, SSM, and program-synthesis baselines under matched budgets."
                ),
                success_criteria=(
                    "Program-synthesis, transformer, SSM, and VPM baselines execute locally.",
                    "C1 and hard-domain LLM baseline artifacts are configured and validated.",
                    "Compute accounting rejects hidden or over-budget baseline work.",
                ),
                artifacts=(
                    "python/vpm/evaluation/baselines.py",
                    "python/vpm/evaluation/neural_baselines.py",
                    "python/vpm/evaluation/llm_baseline_c1.py",
                    "python/vpm/evaluation/llm_baseline_hard.py",
                    "python/vpm/evaluation/openai_release_baseline.py",
                    "python/vpm/evaluation/compute_accounting.py",
                    "tests/integration/sanity/test_baseline_audit.py",
                    "tests/integration/sanity/test_llm_baseline_harness.py",
                    "tests/integration/sanity/test_llm_baseline_openai.py",
                    "tests/integration/sanity/test_compute_accounting.py",
                ),
                verification_commands=(
                    "uv run python -m vpm eval-baselines --limit 0 --json",
                    "uv run python -m vpm run-openai-release-baselines "
                    "artifacts/openai-baselines --model MODEL",
                    "uv run python -m vpm eval-compute --json",
                ),
                release_criteria=(
                    "matched_baselines",
                    "hard_domain_llm_baseline",
                    "criterion1_failure_modes",
                ),
            ),
            objective_item(
                criteria,
                requirement_id="hard_domains",
                requirement=(
                    "Hard domains cover research math, formal tasks, tool use, and "
                    "source-grounded reasoning."
                ),
                success_criteria=(
                    "Held-out hard-domain tasks execute with exact verifier traces.",
                    "Hard-domain LLM comparison is same-budget and separately validated.",
                    "Hard-domain replay is part of the M6 release gate.",
                ),
                artifacts=(
                    "python/vpm/tasks/hard_domains.py",
                    "python/vpm/evaluation/hard_domains.py",
                    "python/vpm/evaluation/llm_baseline_hard.py",
                    "tests/integration/sanity/test_hard_domains.py",
                ),
                verification_commands=(
                    "uv run python -m vpm eval-hard-domains --json",
                    "uv run python -m vpm eval-release --limit 0 --json",
                ),
                release_criteria=(
                    "heldout_hard_domains",
                    "hard_domain_llm_baseline",
                    "m6_red_team",
                ),
            ),
            objective_item(
                criteria,
                requirement_id="clean_ci_release_quality",
                requirement="Clean CI and release-quality completion gates are present.",
                success_criteria=(
                    "Required CI workflows exist for tests, portability, GPU, coverage, and drift.",
                    "Local formatting, typing, tests, and LOC checks have explicit commands.",
                    "Release audit must pass before the objective can be considered complete.",
                ),
                artifacts=(
                    ".github/workflows/ci.yml",
                    ".github/workflows/multi-os.yml",
                    ".github/workflows/gpu.yml",
                    ".github/workflows/coverage.yml",
                    ".github/workflows/drift.yml",
                    "python/vpm/evaluation/release_audit.py",
                    "docs/implementation/README.md",
                    "README.md",
                    "pyproject.toml",
                ),
                verification_commands=(
                    "uv run pytest -q",
                    "uv run ruff check .",
                    "uv run ruff format --check .",
                    "uv run pyright python",
                    "python3 scripts/check-loc.py",
                    "uv run python -m vpm eval-objective --limit 0 --json",
                ),
                release_criteria=(
                    "ci_workflows",
                    "stages_m0_m6",
                    "criterion1_failure_modes",
                    "m6_red_team",
                    "heldout_hard_domains",
                    "matched_baselines",
                    "hard_domain_llm_baseline",
                ),
            ),
        ),
        release_readiness=release,
    )


def objective_item(
    criteria: dict[str, ReleaseCriterion],
    *,
    requirement_id: str,
    requirement: str,
    success_criteria: tuple[str, ...],
    artifacts: tuple[str, ...],
    verification_commands: tuple[str, ...],
    release_criteria: tuple[str, ...],
) -> ObjectiveChecklistItem:
    """Build one checklist item from path checks and release criteria."""
    evidence: list[str] = []
    blockers: list[str] = []
    missing_artifacts = tuple(path for path in artifacts if not (REPO_ROOT / path).exists())
    evidence.extend(
        f"{path}:{'exists' if path not in missing_artifacts else 'missing'}" for path in artifacts
    )
    blockers.extend(f"missing artifact: {path}" for path in missing_artifacts)
    for criterion_id in release_criteria:
        criterion = criteria.get(criterion_id)
        if criterion is None:
            blockers.append(f"missing release criterion: {criterion_id}")
            evidence.append(f"{criterion_id}:missing")
            continue
        evidence.append(
            f"{criterion_id}:passed={criterion.passed}:blockers={len(criterion.blockers)}"
        )
        blockers.extend(criterion.blockers)
    return ObjectiveChecklistItem(
        requirement_id=requirement_id,
        requirement=requirement,
        success_criteria=success_criteria,
        artifacts=artifacts,
        verification_commands=verification_commands,
        release_criteria=release_criteria,
        passed=not blockers,
        evidence=tuple(evidence),
        blockers=tuple(dict.fromkeys(blockers)),
    )


__all__ = [
    "ObjectiveAuditReport",
    "ObjectiveChecklistItem",
    "evaluate_objective_completion",
]
