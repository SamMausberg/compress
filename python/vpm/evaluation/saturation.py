"""Macro saturation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from vpm.evaluation.compression import CompressionReplayReport, evaluate_c5


@dataclass(frozen=True)
class SaturationReport:
    """Check whether any replay macro still has positive frontier LCB."""

    max_frontier_lcb: float
    positive_macros: int
    macros: int

    @property
    def saturated(self) -> bool:
        """True when no macro has positive lower-confidence utility."""
        return self.max_frontier_lcb <= 0.0

    def to_dict(self) -> dict[str, float | int | bool]:
        """JSON-friendly saturation report."""
        return {
            "saturated": self.saturated,
            "max_frontier_lcb": self.max_frontier_lcb,
            "positive_macros": self.positive_macros,
            "macros": self.macros,
        }


def evaluate_saturation(report: CompressionReplayReport | None = None) -> SaturationReport:
    """Evaluate the saturation condition over C5 replay traces."""
    compression = evaluate_c5() if report is None else report
    lcbs = tuple(trace.frontier_lcb for trace in compression.traces)
    max_lcb = max(lcbs, default=0.0)
    return SaturationReport(
        max_frontier_lcb=max_lcb,
        positive_macros=sum(lcb > 0.0 for lcb in lcbs),
        macros=len(lcbs),
    )


__all__ = ["SaturationReport", "evaluate_saturation"]
