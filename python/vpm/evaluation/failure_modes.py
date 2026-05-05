"""Compatibility exports for Criterion-1 failure-mode audits."""

from __future__ import annotations

from vpm.audit.failure_modes import (
    UNCOVERED_CRITERION1_CLAUSES,
    FailureCheck,
    FailureMode,
    FailureModeReport,
    evaluate_failure_modes,
)

__all__ = [
    "UNCOVERED_CRITERION1_CLAUSES",
    "FailureCheck",
    "FailureMode",
    "FailureModeReport",
    "evaluate_failure_modes",
]
