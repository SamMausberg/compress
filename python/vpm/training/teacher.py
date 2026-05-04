"""Certified-trace teacher posterior."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class TeacherTrace:
    """One candidate trace considered by the teacher posterior."""

    trace_id: str
    candidate: str
    certificate: float
    utility: float
    cost: float
    split_clean: bool = True
    replay_passed: bool = True

    @property
    def score(self) -> float:
        """Teacher energy: certified utility net of execution cost."""
        return self.certificate + self.utility - self.cost

    @property
    def eligible(self) -> bool:
        """Only split-clean replay-passing traces can enter ``p_*``."""
        return self.split_clean and self.replay_passed and self.certificate > 0.0

    def to_dict(self) -> dict[str, float | bool | str]:
        """JSON-friendly trace."""
        return {
            "trace_id": self.trace_id,
            "candidate": self.candidate,
            "certificate": self.certificate,
            "utility": self.utility,
            "cost": self.cost,
            "score": self.score,
            "split_clean": self.split_clean,
            "replay_passed": self.replay_passed,
            "eligible": self.eligible,
        }


@dataclass(frozen=True)
class TeacherPosteriorEntry:
    """One normalized posterior entry."""

    trace: TeacherTrace
    probability: float

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly posterior entry."""
        return {"trace": self.trace.to_dict(), "probability": self.probability}


@dataclass(frozen=True)
class TeacherPosterior:
    """Truncated certified teacher posterior from eq. 141."""

    entries: tuple[TeacherPosteriorEntry, ...]
    rejected: tuple[TeacherTrace, ...]

    @property
    def support(self) -> tuple[str, ...]:
        """Candidate names with non-zero posterior mass."""
        return tuple(entry.trace.candidate for entry in self.entries)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly posterior."""
        return {
            "support": self.support,
            "entries": [entry.to_dict() for entry in self.entries],
            "rejected": [trace.to_dict() for trace in self.rejected],
        }


def teacher_posterior(
    traces: tuple[TeacherTrace, ...],
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> TeacherPosterior:
    """Build a truncated normalized posterior over certified traces."""
    eligible = tuple(trace for trace in traces if trace.eligible)
    rejected = tuple(trace for trace in traces if not trace.eligible)
    ordered = sorted(eligible, key=lambda trace: (trace.score, trace.trace_id), reverse=True)
    truncated = tuple(ordered[:top_k]) if top_k is not None else tuple(ordered)
    if not truncated:
        return TeacherPosterior((), rejected)
    temp = temperature if temperature > 0.0 else 1.0
    max_score = max(trace.score for trace in truncated)
    weights = tuple(exp((trace.score - max_score) / temp) for trace in truncated)
    normalizer = sum(weights)
    entries = tuple(
        TeacherPosteriorEntry(trace, weight / normalizer)
        for trace, weight in zip(truncated, weights, strict=True)
    )
    return TeacherPosterior(entries, rejected)


__all__ = [
    "TeacherPosterior",
    "TeacherPosteriorEntry",
    "TeacherTrace",
    "teacher_posterior",
]
