"""Criterion-1 failure-mode detectors for executable VPM-0 slices."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from vpm.evaluation import evaluate_c2, evaluate_c3, evaluate_c4, evaluate_c5
from vpm.evaluation.compute_accounting import evaluate_compute_accounting, hidden_compute_probe
from vpm.evaluation.external_components import (
    dirty_external_component_probe,
    evaluate_external_components,
)
from vpm.retrieval.calibration import dirty_recall_shift_probe, evaluate_recall_shift
from vpm.substrate import encode_task_graph
from vpm.tasks.c0 import arithmetic_task
from vpm.training.probes import edge_deletion_probe
from vpm.training.splits import SplitAssignment
from vpm.verifiers.dependence import dirty_dependence_shift_probe, evaluate_dependence_shift


class FailureMode(StrEnum):
    """Executable subset of Criterion 1 failure clauses."""

    SUPPORT_GUARD_BYPASS = "support_guard_bypass"
    STAGE_SCHEDULER_BYPASS = "stage_scheduler_bypass"
    SUBSTRATE_CRITICAL_EDGE_BYPASS = "substrate_critical_edge_bypass"
    SPLIT_LEAKAGE_BYPASS = "split_leakage_bypass"
    SOURCE_REBUTTAL_SHIFT_BYPASS = "source_rebuttal_shift_bypass"
    DEPENDENCE_RESIDUALIZATION_BYPASS = "dependence_residualization_bypass"
    HIDDEN_COMPUTE_BYPASS = "hidden_compute_bypass"
    EXTERNAL_LLM_DEPENDENCE = "external_llm_dependence"
    SAFETY_GATE_BYPASS = "safety_gate_bypass"
    DIALOGUE_GATE_BYPASS = "dialogue_gate_bypass"
    OPAQUE_MACRO_ADMISSION = "opaque_macro_admission"
    LINEAR_ACTIVE_MEMORY = "linear_active_memory"


UNCOVERED_CRITERION1_CLAUSES = (
    "same-budget external LLM baseline",
    "open-domain context and semantic ambiguity collapse",
    "entailment false-support attacks outside controlled corpus",
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
    recall_shift = evaluate_recall_shift()
    dirty_recall_shift = dirty_recall_shift_probe()
    dependence_shift = evaluate_dependence_shift()
    dirty_dependence_shift = dirty_dependence_shift_probe()
    external_components = evaluate_external_components()
    dirty_external_components = dirty_external_component_probe()
    compute = evaluate_compute_accounting()
    hidden_compute = hidden_compute_probe()
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
                FailureMode.SOURCE_REBUTTAL_SHIFT_BYPASS,
                not recall_shift.passed or dirty_recall_shift.passed,
                (
                    f"source_epsilon={recall_shift.source_epsilon:.3f} "
                    f"rebuttal_epsilon={recall_shift.rebuttal_epsilon:.3f} "
                    f"shifted_epsilon={recall_shift.shifted_epsilon:.3f} "
                    f"dirty_passed={dirty_recall_shift.passed}"
                ),
            ),
            FailureCheck(
                FailureMode.HIDDEN_COMPUTE_BYPASS,
                not compute.passed or hidden_compute.passed,
                (f"compute_passed={compute.passed} hidden_units={hidden_compute.hidden_units:.3f}"),
            ),
            FailureCheck(
                FailureMode.DEPENDENCE_RESIDUALIZATION_BYPASS,
                not dependence_shift.passed or dirty_dependence_shift.passed,
                (
                    f"epsilon_dep={dependence_shift.epsilon_dep:.3f} "
                    f"shifted_epsilon={dependence_shift.shifted_epsilon:.3f} "
                    f"dirty_passed={dirty_dependence_shift.passed}"
                ),
            ),
            FailureCheck(
                FailureMode.EXTERNAL_LLM_DEPENDENCE,
                not external_components.passed or dirty_external_components.passed,
                (
                    f"violations={len(external_components.violations)} "
                    f"external_inference="
                    f"{len(external_components.external_inference_dependencies)} "
                    f"dirty_passed={dirty_external_components.passed}"
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
