"""Compatibility exports for objective completion audits."""

from __future__ import annotations

from vpm.audit.objective_audit import (
    ObjectiveAuditReport,
    ObjectiveChecklistItem,
    evaluate_objective_completion,
)

__all__ = [
    "ObjectiveAuditReport",
    "ObjectiveChecklistItem",
    "evaluate_objective_completion",
]
