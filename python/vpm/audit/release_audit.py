"""Release-readiness audit for the VPM research system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vpm.audit.failure_modes import FailureModeReport, evaluate_failure_modes
from vpm.audit.red_team import RedTeamReport
from vpm.evaluation.ablations import evaluate_ablations
from vpm.evaluation.baselines import (
    BaselineAudit,
    BaselineFamily,
    BaselineStatus,
    BaselineSuite,
    evaluate_baseline_suite,
    external_baseline,
)
from vpm.evaluation.hard_domains import HardDomainReport, evaluate_hard_domains
from vpm.tasks import stages

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ReleaseCriterion:
    """One concrete release-readiness criterion with evidence."""

    criterion_id: str
    summary: str
    passed: bool
    evidence: tuple[str, ...]
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly release criterion."""
        return {
            "criterion_id": self.criterion_id,
            "summary": self.summary,
            "passed": self.passed,
            "evidence": self.evidence,
            "blockers": self.blockers,
        }


@dataclass(frozen=True)
class ReleaseReadinessReport:
    """Objective-facing release-readiness report."""

    criteria: tuple[ReleaseCriterion, ...]

    @property
    def passed(self) -> bool:
        """True when every release criterion is satisfied."""
        return all(criterion.passed for criterion in self.criteria)

    @property
    def blockers(self) -> tuple[str, ...]:
        """Flattened release blockers."""
        return tuple(blocker for criterion in self.criteria for blocker in criterion.blockers)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly release-readiness report."""
        return {
            "passed": self.passed,
            "blockers": self.blockers,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
        }


def evaluate_release_readiness(limit: int = 0) -> ReleaseReadinessReport:
    """Evaluate release readiness against the active VPM objective."""
    failure_modes = evaluate_failure_modes()
    baselines = evaluate_baseline_suite(limit=limit)
    hard_domains = evaluate_hard_domains()
    hard_llm = external_baseline(
        BaselineFamily.LLM,
        "VPM_HARD_LLM_BASELINE_JSON",
        max_compute_units=float(hard_domains.tasks),
        task_kind="hard",
        expected_task_ids=tuple(trace.task_id for trace in hard_domains.traces),
    )
    red_team = RedTeamReport(
        failures=failure_modes,
        ablations=evaluate_ablations(),
        hard_domains=hard_domains,
    )
    return ReleaseReadinessReport(
        (
            stage_criterion(),
            failure_mode_criterion(failure_modes, baselines, hard_llm),
            red_team_criterion(red_team),
            hard_domain_criterion(hard_domains),
            baseline_criterion(baselines),
            hard_domain_llm_baseline_criterion(hard_llm),
            ci_criterion(),
        )
    )


def stage_criterion() -> ReleaseCriterion:
    """Audit runtime-visible M0-M6 curriculum stage coverage."""
    specs = stages()
    missing = tuple(
        spec.name for spec in specs if not spec.executable or not spec.implemented_components
    )
    expected = ("C0", "C1", "C2", "C3", "C4", "C5", "M6")
    names = tuple(spec.name for spec in specs)
    blockers: list[str] = []
    if names != expected:
        blockers.append(f"stage order mismatch: {names}")
    blockers.extend(f"{name} lacks executable components" for name in missing)
    blockers.extend(
        f"{spec.name} stage blocker: {blocker}" for spec in specs for blocker in spec.blockers
    )
    return ReleaseCriterion(
        criterion_id="stages_m0_m6",
        summary="M0-M6 curriculum stages are runtime-visible and executable.",
        passed=not blockers,
        evidence=tuple(
            f"{spec.name}:{','.join(spec.implemented_components)}:blockers={len(spec.blockers)}"
            for spec in specs
        ),
        blockers=tuple(blockers),
    )


def failure_mode_criterion(
    report: FailureModeReport,
    baselines: BaselineSuite | None = None,
    hard_llm: BaselineAudit | None = None,
) -> ReleaseCriterion:
    """Audit Criterion-1 executable failure modes and uncovered clauses."""
    uncovered = covered_uncovered_clauses(report, baselines, hard_llm)
    blockers = uncovered + tuple(
        f"triggered failure: {failure.mode.value}" for failure in report.failures
    )
    return ReleaseCriterion(
        criterion_id="criterion1_failure_modes",
        summary="Criterion-1 failure modes are covered and unfired.",
        passed=not blockers,
        evidence=(
            f"checks={len(report.checks)}",
            f"failures={len(report.failures)}",
            f"uncovered={len(uncovered)}",
        ),
        blockers=blockers,
    )


def covered_uncovered_clauses(
    report: FailureModeReport,
    baselines: BaselineSuite | None,
    hard_llm: BaselineAudit | None,
) -> tuple[str, ...]:
    """Drop clauses that are covered by executed audited baseline artifacts."""
    if not llm_baselines_executed(baselines, hard_llm):
        return tuple(report.uncovered_clauses)
    return tuple(
        clause
        for clause in report.uncovered_clauses
        if clause != "same-budget external LLM baseline"
    )


def llm_baselines_executed(
    baselines: BaselineSuite | None,
    hard_llm: BaselineAudit | None,
) -> bool:
    """True when both C1 and hard-domain LLM baselines are executed."""
    if baselines is None or hard_llm is None:
        return False
    c1_llm_executed = any(
        baseline.family is BaselineFamily.LLM and baseline.status is BaselineStatus.EXECUTED
        for baseline in baselines.baselines
    )
    return c1_llm_executed and hard_llm.status is BaselineStatus.EXECUTED


def red_team_criterion(report: RedTeamReport) -> ReleaseCriterion:
    """Audit executable red-team replay."""
    blockers: list[str] = []
    if not report.failures.passed:
        blockers.append("failure-mode replay did not pass")
    if not report.ablations.passed:
        blockers.append("ablation replay did not pass")
    if report.hard_domains.solve_rate < 1.0:
        blockers.append("hard-domain replay did not solve every task")
    return ReleaseCriterion(
        criterion_id="m6_red_team",
        summary="M6 red-team, ablation, and hard-domain replay passes.",
        passed=not blockers,
        evidence=(
            f"failures_passed={report.failures.passed}",
            f"ablations_passed={report.ablations.passed}",
            f"hard_domain_solve_rate={report.hard_domains.solve_rate:.3f}",
        ),
        blockers=tuple(blockers),
    )


def hard_domain_criterion(report: HardDomainReport) -> ReleaseCriterion:
    """Audit held-out hard-domain VPM execution."""
    blockers = ()
    if report.solve_rate < 1.0:
        blockers = ("held-out hard-domain solve rate below 1.0",)
    return ReleaseCriterion(
        criterion_id="heldout_hard_domains",
        summary="Held-out research math, formal, tool, and source-grounded tasks execute.",
        passed=not blockers,
        evidence=(
            f"tasks={report.tasks}",
            f"solve_rate={report.solve_rate:.3f}",
            f"baseline_solve_rate={report.baseline_solve_rate:.3f}",
        ),
        blockers=blockers,
    )


def baseline_criterion(report: BaselineSuite) -> ReleaseCriterion:
    """Audit matched baseline availability."""
    blockers = tuple(
        f"missing executed baseline family: {family}" for family in report.missing_families
    )
    return ReleaseCriterion(
        criterion_id="matched_baselines",
        summary="VPM is compared against program-synthesis, transformer, SSM, and LLM baselines.",
        passed=report.ready_for_claims,
        evidence=tuple(
            f"{baseline.family.value}:{baseline.status.value}:{baseline.reason}"
            for baseline in report.baselines
        ),
        blockers=blockers,
    )


def hard_domain_llm_baseline_criterion(baseline: BaselineAudit) -> ReleaseCriterion:
    """Audit same-budget external LLM comparison on held-out hard domains."""
    blockers = ()
    if baseline.status is not BaselineStatus.EXECUTED:
        blockers = (f"missing executed hard-domain LLM baseline: {baseline.reason}",)
    return ReleaseCriterion(
        criterion_id="hard_domain_llm_baseline",
        summary="Held-out hard domains have a same-budget external LLM baseline.",
        passed=not blockers,
        evidence=(
            f"status={baseline.status.value}",
            f"solve_rate={baseline.solve_rate}",
            f"compute_units={baseline.compute_units}",
            f"max_compute_units={baseline.max_compute_units}",
            f"reason={baseline.reason}",
        ),
        blockers=blockers,
    )


def ci_criterion() -> ReleaseCriterion:
    """Audit expected CI workflow files."""
    required = (
        ".github/workflows/ci.yml",
        ".github/workflows/multi-os.yml",
        ".github/workflows/gpu.yml",
        ".github/workflows/coverage.yml",
        ".github/workflows/drift.yml",
    )
    missing = tuple(path for path in required if not (REPO_ROOT / path).exists())
    return ReleaseCriterion(
        criterion_id="ci_workflows",
        summary="Core CI workflow files exist for tests, portability, GPU, coverage, and drift.",
        passed=not missing,
        evidence=required,
        blockers=tuple(f"missing workflow: {path}" for path in missing),
    )


__all__ = [
    "ReleaseCriterion",
    "ReleaseReadinessReport",
    "evaluate_release_readiness",
]
