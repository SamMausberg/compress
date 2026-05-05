"""Compatibility exports for release-readiness audits."""

from __future__ import annotations

from vpm.audit.release_audit import (
    ReleaseCriterion,
    ReleaseReadinessReport,
    evaluate_release_readiness,
)

__all__ = [
    "ReleaseCriterion",
    "ReleaseReadinessReport",
    "evaluate_release_readiness",
]
