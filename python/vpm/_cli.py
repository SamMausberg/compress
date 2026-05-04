"""Shared CLI formatting helpers."""

from __future__ import annotations


def prototype_summary(report) -> str:
    """Compact summary for prototype evaluation reports."""
    return (
        f"solve_rate={report.solve_rate:.3f} "
        f"op_acc={report.operation_accuracy:.3f} "
        f"compression={report.compression_ratio:.3f} "
        f"frontier_delta={report.compression.frontier_delta_vs_enumerative:.3f}"
    )


def training_summary(report) -> str:
    """Compact summary for training reports."""
    return f"{prototype_summary(report.heldout)} artifact={report.artifact}"


__all__ = ["prototype_summary", "training_summary"]
