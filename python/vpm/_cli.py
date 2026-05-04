"""Shared CLI formatting helpers."""

from __future__ import annotations

from typing import Protocol


class _CompressionLike(Protocol):
    @property
    def frontier_delta_vs_enumerative(self) -> float: ...


class PrototypeSummaryReport(Protocol):
    @property
    def solve_rate(self) -> float: ...

    @property
    def operation_accuracy(self) -> float: ...

    @property
    def compression_ratio(self) -> float: ...

    @property
    def compression(self) -> _CompressionLike: ...


class TrainingSummaryReport(Protocol):
    @property
    def heldout(self) -> PrototypeSummaryReport: ...

    @property
    def artifact(self) -> str | None: ...


def prototype_summary(report: PrototypeSummaryReport) -> str:
    """Compact summary for prototype evaluation reports."""
    return (
        f"solve_rate={report.solve_rate:.3f} "
        f"op_acc={report.operation_accuracy:.3f} "
        f"compression={report.compression_ratio:.3f} "
        f"frontier_delta={report.compression.frontier_delta_vs_enumerative:.3f}"
    )


def training_summary(report: TrainingSummaryReport) -> str:
    """Compact summary for training reports."""
    return f"{prototype_summary(report.heldout)} artifact={report.artifact}"


__all__ = [
    "PrototypeSummaryReport",
    "TrainingSummaryReport",
    "prototype_summary",
    "training_summary",
]
