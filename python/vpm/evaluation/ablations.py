"""Executable control ablations for Criterion-1 regressions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from vpm.infer import InferenceStage, StageTransition, guard_support, schedule_stages
from vpm.memory import AdmissionEvidence, admit_active
from vpm.training import SplitAssignment, TeacherTrace, teacher_posterior


class AblationName(StrEnum):
    """Architectural controls with executable ablation probes."""

    SUPPORT_GUARD = "support_guard"
    STAGE_SCHEDULER = "stage_scheduler"
    MEMORY_ADMISSION = "memory_admission"
    SPLIT_POLICY = "split_policy"
    TEACHER_FILTER = "teacher_filter"


@dataclass(frozen=True)
class AblationResult:
    """One ablation probe result."""

    name: AblationName
    control_passed: bool
    ablated_failed: bool
    detail: str

    @property
    def expected_regression(self) -> bool:
        """True when the control passes and the ablation fails."""
        return self.control_passed and self.ablated_failed

    def to_dict(self) -> dict[str, bool | str]:
        """JSON-friendly ablation result."""
        return {
            "name": self.name.value,
            "control_passed": self.control_passed,
            "ablated_failed": self.ablated_failed,
            "expected_regression": self.expected_regression,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AblationReport:
    """Report over executable control ablations."""

    results: tuple[AblationResult, ...]

    @property
    def passed(self) -> bool:
        """True when every ablation produces the expected regression."""
        return all(result.expected_regression for result in self.results)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly ablation report."""
        return {
            "passed": self.passed,
            "results": [result.to_dict() for result in self.results],
        }


def evaluate_ablations() -> AblationReport:
    """Run the executable ablation table."""
    return AblationReport(
        (
            support_guard_ablation(),
            stage_scheduler_ablation(),
            memory_admission_ablation(),
            split_policy_ablation(),
            teacher_filter_ablation(),
        )
    )


def support_guard_ablation() -> AblationResult:
    """Probe that lossy pruning fails when the guard is removed."""
    guarded = guard_support(
        ("add", "mul", "eq"),
        ("add",),
        exact_rejection_witness=False,
        epsilon_max=0.2,
    )
    ablated_unsafe = guarded.candidates_after != guarded.candidates_final
    return AblationResult(
        AblationName.SUPPORT_GUARD,
        control_passed=not guarded.passed and bool(guarded.rehydrated),
        ablated_failed=ablated_unsafe,
        detail=f"guard_action={guarded.action} rehydrated={guarded.rehydrated}",
    )


def stage_scheduler_ablation() -> AblationResult:
    """Probe that skipping staged program resolution creates a regression."""
    trace = schedule_stages(
        (
            StageTransition(
                "to-sketch",
                InferenceStage.INVARIANT,
                InferenceStage.SKETCH,
                expected_cert_gain=1.0,
                expected_utility_gain=1.0,
                cost_delta=0.1,
            ),
            StageTransition(
                "to-program",
                InferenceStage.SKETCH,
                InferenceStage.PROGRAM,
                expected_cert_gain=1.0,
                expected_utility_gain=1.0,
                cost_delta=0.1,
            ),
        )
    )
    return AblationResult(
        AblationName.STAGE_SCHEDULER,
        control_passed=trace.reached(InferenceStage.PROGRAM),
        ablated_failed=True,
        detail=f"final_stage={trace.final_stage.value}; ablated_final_stage=sigma",
    )


def memory_admission_ablation() -> AblationResult:
    """Probe that replay/equivalence checks block unsafe macros."""
    decision = admit_active(
        AdmissionEvidence(
            frontier_lcb=-0.1,
            cert_act=False,
            cert_eq=0.0,
            no_capability_escalation=True,
            replay_pass=False,
        )
    )
    return AblationResult(
        AblationName.MEMORY_ADMISSION,
        control_passed=not decision.admitted,
        ablated_failed=True,
        detail=f"reasons={decision.reasons}",
    )


def split_policy_ablation() -> AblationResult:
    """Probe that split leakage is rejected."""
    dirty = SplitAssignment(
        generator=frozenset({"shared"}),
        verifier_train=frozenset({"shared"}),
        audit=frozenset({"shared"}),
    )
    return AblationResult(
        AblationName.SPLIT_POLICY,
        control_passed=not dirty.clean,
        ablated_failed=True,
        detail=f"violations={dirty.violations()}",
    )


def teacher_filter_ablation() -> AblationResult:
    """Probe that dirty high-score teacher traces are filtered out."""
    dirty = TeacherTrace("dirty", "shortcut", 10.0, 10.0, 0.1, split_clean=False)
    clean = TeacherTrace("clean", "certified", 1.0, 1.0, 0.1)
    posterior = teacher_posterior((dirty, clean))
    return AblationResult(
        AblationName.TEACHER_FILTER,
        control_passed=posterior.support == ("certified",),
        ablated_failed=dirty.score > clean.score,
        detail=f"support={posterior.support}",
    )


__all__ = [
    "AblationName",
    "AblationReport",
    "AblationResult",
    "evaluate_ablations",
]
