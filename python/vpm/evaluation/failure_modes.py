"""Criterion-1 failure-mode detectors for executable VPM-0 slices."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from vpm.evaluation import evaluate_c2, evaluate_c3, evaluate_c4, evaluate_c5
from vpm.substrate import encode_task_graph
from vpm.tasks.c0 import arithmetic_task
from vpm.training.probes import edge_deletion_probe
from vpm.training.splits import SplitAssignment


class FailureMode(StrEnum):
    """Executable subset of Criterion 1 failure clauses."""

    SUPPORT_GUARD_BYPASS = "support_guard_bypass"
    STAGE_SCHEDULER_BYPASS = "stage_scheduler_bypass"
    SUBSTRATE_CRITICAL_EDGE_BYPASS = "substrate_critical_edge_bypass"
    SPLIT_LEAKAGE_BYPASS = "split_leakage_bypass"
    SAFETY_GATE_BYPASS = "safety_gate_bypass"
    DIALOGUE_GATE_BYPASS = "dialogue_gate_bypass"
    OPAQUE_MACRO_ADMISSION = "opaque_macro_admission"
    LINEAR_ACTIVE_MEMORY = "linear_active_memory"


UNCOVERED_CRITERION1_CLAUSES = (
    "same-budget transformer/SSM/program-synthesis baselines",
    "open-domain context and semantic ambiguity collapse",
    "source/rebuttal recall miss calibration under shift",
    "entailment false-support attacks outside controlled corpus",
    "dependence residualization calibration under shift",
    "hidden test-time compute accounting",
    "external LLM cognitive-component dependence",
)


@dataclass(frozen=True)
class FailureCheck:
    """One Criterion-1 executable failure check."""

    mode: FailureMode
    triggered: bool
    detail: str

    def to_dict(self) -> dict[str, bool | str]:
        """JSON-friendly failure check."""
        return {
            "mode": self.mode.value,
            "triggered": self.triggered,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FailureModeReport:
    """Report over executable Criterion-1 failure checks."""

    checks: tuple[FailureCheck, ...]
    uncovered_clauses: tuple[str, ...] = UNCOVERED_CRITERION1_CLAUSES

    @property
    def failures(self) -> tuple[FailureCheck, ...]:
        """Triggered failure checks."""
        return tuple(check for check in self.checks if check.triggered)

    @property
    def passed(self) -> bool:
        """True when every covered executable failure mode is unfired."""
        return not self.failures

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly failure-mode report."""
        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "failures": [check.to_dict() for check in self.failures],
            "uncovered_clauses": self.uncovered_clauses,
        }


def evaluate_failure_modes() -> FailureModeReport:
    """Run executable Criterion-1 failure checks over shipped curricula."""
    c2 = evaluate_c2(limit=1)
    c3 = evaluate_c3()
    c4 = evaluate_c4()
    c5 = evaluate_c5()
    clean_split = SplitAssignment(
        generator=frozenset({"gen"}),
        verifier_train=frozenset({"train"}),
        verifier_calibration=frozenset({"cal"}),
        audit=frozenset({"audit"}),
    )
    dirty_split = SplitAssignment(
        generator=frozenset({"shared"}),
        verifier_train=frozenset({"shared"}),
        audit=frozenset({"shared"}),
    )
    substrate_probe = edge_deletion_probe(
        encode_task_graph(arithmetic_task("add", 2, 5)),
        ("left", "expected"),
    )
    return FailureModeReport(
        checks=(
            FailureCheck(
                FailureMode.SUPPORT_GUARD_BYPASS,
                c2.support_guard_pass_rate < 1.0 and c2.rehydrated == 0,
                (
                    f"support_guard_pass_rate={c2.support_guard_pass_rate:.3f} "
                    f"rehydrated={c2.rehydrated}"
                ),
            ),
            FailureCheck(
                FailureMode.STAGE_SCHEDULER_BYPASS,
                c2.program_entry_rate < c2.solve_rate,
                (f"program_entry_rate={c2.program_entry_rate:.3f} solve_rate={c2.solve_rate:.3f}"),
            ),
            FailureCheck(
                FailureMode.SPLIT_LEAKAGE_BYPASS,
                not clean_split.clean or dirty_split.clean,
                (f"clean_split={clean_split.clean} dirty_violations={dirty_split.violations()}"),
            ),
            FailureCheck(
                FailureMode.SUBSTRATE_CRITICAL_EDGE_BYPASS,
                not substrate_probe.failed,
                (
                    f"epsilon_crit={substrate_probe.report.epsilon_crit:.3f} "
                    f"failed={substrate_probe.failed}"
                ),
            ),
            FailureCheck(
                FailureMode.SAFETY_GATE_BYPASS,
                c3.violations > 0,
                f"policy_gate_violations={c3.violations}",
            ),
            FailureCheck(
                FailureMode.DIALOGUE_GATE_BYPASS,
                c4.violations > 0,
                f"dialogue_gate_violations={c4.violations}",
            ),
            FailureCheck(
                FailureMode.OPAQUE_MACRO_ADMISSION,
                c5.violations > 0,
                f"macro_admission_violations={c5.violations}",
            ),
            FailureCheck(
                FailureMode.LINEAR_ACTIVE_MEMORY,
                c5.sublinear_active < c5.admitted,
                f"sublinear_active={c5.sublinear_active} admitted={c5.admitted}",
            ),
        )
    )


__all__ = [
    "UNCOVERED_CRITERION1_CLAUSES",
    "FailureCheck",
    "FailureMode",
    "FailureModeReport",
    "evaluate_failure_modes",
]
