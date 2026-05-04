"""Critical-edge substrate probes."""

from __future__ import annotations

from dataclasses import dataclass

from vpm.substrate.encoder import TypedEventGraph
from vpm.substrate.losses import SubstrateRecallReport, substrate_recall_report


@dataclass(frozen=True)
class CriticalEdgeProbe:
    """Counterfactual critical-edge deletion probe."""

    task_id: str
    deleted_edge: tuple[str, str]
    report: SubstrateRecallReport

    @property
    def failed(self) -> bool:
        """True when deleting the edge violates the recall threshold."""
        return not self.report.passed

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly probe."""
        return {
            "task_id": self.task_id,
            "deleted_edge": self.deleted_edge,
            "failed": self.failed,
            "report": self.report.to_dict(),
        }


def edge_deletion_probe(
    graph: TypedEventGraph,
    deleted_edge: tuple[str, str],
    *,
    threshold: float = 0.0,
) -> CriticalEdgeProbe:
    """Probe substrate recall under one critical-edge deletion."""
    critical_omissions = int(deleted_edge in graph.edges)
    report = substrate_recall_report(
        omitted_edges=critical_omissions,
        total_edges=len(graph.edges),
        critical_omissions=critical_omissions,
        threshold=threshold,
    )
    return CriticalEdgeProbe(graph.task_id, deleted_edge, report)


__all__ = ["CriticalEdgeProbe", "edge_deletion_probe"]
