"""Held-out rate-distortion frontier movement."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from vpm import _native
from vpm._reports import float_field, object_map


@dataclass(frozen=True)
class EmpiricalBernsteinBound:
    """Sequence-valid empirical-Bernstein replay bound."""

    samples: int
    candidates_tested: int
    mean_gain: float
    variance: float
    radius: float
    adapt_loss: float
    drift_loss: float
    leak_loss: float
    selection_loss: float
    lcb: float
    ucb: float

    def to_dict(self) -> dict[str, float | int]:
        """JSON-friendly bound."""
        return {
            "samples": self.samples,
            "candidates_tested": self.candidates_tested,
            "mean_gain": self.mean_gain,
            "variance": self.variance,
            "radius": self.radius,
            "adapt_loss": self.adapt_loss,
            "drift_loss": self.drift_loss,
            "leak_loss": self.leak_loss,
            "selection_loss": self.selection_loss,
            "lcb": self.lcb,
            "ucb": self.ucb,
        }


@dataclass(frozen=True)
class FrontierReport:
    """Replay frontier movement for one macro candidate."""

    certified: int
    replay_tasks: int
    macro_utility: float
    enumerative_utility: float
    frontier_delta: float
    bound: EmpiricalBernsteinBound

    @property
    def positive_lcb(self) -> bool:
        """True when the lower confidence frontier movement is positive."""
        return self.bound.lcb > 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly frontier report."""
        return {
            "certified": self.certified,
            "replay_tasks": self.replay_tasks,
            "macro_utility": self.macro_utility,
            "enumerative_utility": self.enumerative_utility,
            "frontier_delta": self.frontier_delta,
            "positive_lcb": self.positive_lcb,
            "bound": self.bound.to_dict(),
        }


@dataclass(frozen=True)
class OnlineFrontierEstimator:
    """Incremental replay frontier estimator for one macro candidate."""

    macro_key: str
    enumerative_utility: float
    candidates_tested: int = 1
    exact_replay: bool = True
    gains: tuple[float, ...] = ()

    @property
    def observations(self) -> int:
        """Number of replay outcomes observed so far."""
        return len(self.gains)

    @property
    def certified(self) -> int:
        """Number of certified replay outcomes observed so far."""
        return sum(gain > 0.0 for gain in self.gains)

    def observe(self, certified: bool) -> OnlineFrontierEstimator:
        """Return an estimator updated with one replay outcome."""
        utility = 1.0 if certified else 0.0
        gain = utility - self.enumerative_utility
        return OnlineFrontierEstimator(
            macro_key=self.macro_key,
            enumerative_utility=self.enumerative_utility,
            candidates_tested=self.candidates_tested,
            exact_replay=self.exact_replay,
            gains=(*self.gains, gain),
        )

    def report(self) -> FrontierReport:
        """Return the current sequence-valid frontier report."""
        bound = eb_seq_bound(
            self.gains,
            candidates_tested=self.candidates_tested,
            gain_bound=0.0 if self.exact_replay else 1.0,
        )
        macro_utility = self.certified / self.observations if self.observations else 0.0
        return FrontierReport(
            certified=self.certified,
            replay_tasks=self.observations,
            macro_utility=macro_utility,
            enumerative_utility=self.enumerative_utility,
            frontier_delta=macro_utility - self.enumerative_utility,
            bound=bound,
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly online estimator state."""
        return {
            "macro_key": self.macro_key,
            "observations": self.observations,
            "certified": self.certified,
            "enumerative_utility": self.enumerative_utility,
            "candidates_tested": self.candidates_tested,
            "exact_replay": self.exact_replay,
            "gains": self.gains,
            "report": self.report().to_dict(),
        }


def replay_frontier_report(
    certified: int,
    replay_tasks: int,
    *,
    enumerative_utility: float,
    candidates_tested: int = 1,
    exact_replay: bool = True,
) -> FrontierReport:
    """Build a frontier report from certified replay outcomes."""
    macro_utility = certified / replay_tasks if replay_tasks else 0.0
    gains = replay_gains(certified, replay_tasks, enumerative_utility)
    bound = eb_seq_bound(
        gains,
        candidates_tested=candidates_tested,
        gain_bound=0.0 if exact_replay else 1.0,
    )
    return FrontierReport(
        certified=certified,
        replay_tasks=replay_tasks,
        macro_utility=macro_utility,
        enumerative_utility=enumerative_utility,
        frontier_delta=macro_utility - enumerative_utility,
        bound=bound,
    )


def online_replay_frontier_report(
    outcomes: tuple[bool, ...],
    *,
    macro_key: str,
    enumerative_utility: float,
    candidates_tested: int = 1,
    exact_replay: bool = True,
) -> FrontierReport:
    """Build a frontier report by feeding replay outcomes incrementally."""
    estimator = OnlineFrontierEstimator(
        macro_key=macro_key,
        enumerative_utility=enumerative_utility,
        candidates_tested=candidates_tested,
        exact_replay=exact_replay,
    )
    for outcome in outcomes:
        estimator = estimator.observe(outcome)
    return estimator.report()


def replay_gains(
    certified: int,
    replay_tasks: int,
    enumerative_utility: float,
) -> tuple[float, ...]:
    """Convert replay pass/fail counts into bounded utility gains."""
    successes = max(0, min(certified, replay_tasks))
    failures = max(0, replay_tasks - successes)
    return ((1.0 - enumerative_utility),) * successes + ((0.0 - enumerative_utility),) * failures


def eb_seq_bound(
    gains: tuple[float, ...],
    *,
    candidates_tested: int = 1,
    delta: float = 0.05,
    gain_bound: float = 1.0,
    adapt_loss: float = 0.0,
    drift_loss: float = 0.0,
    leak_loss: float = 0.0,
    selection_loss: float = 0.0,
) -> EmpiricalBernsteinBound:
    """Call the Rust empirical-Bernstein bound implementation."""
    payload = object_map(
        json.loads(
            _native.eb_seq_json(
                json.dumps(gains),
                candidates_tested,
                delta,
                gain_bound,
                adapt_loss,
                drift_loss,
                leak_loss,
                selection_loss,
            )
        )
    )
    if payload is None:
        raise ValueError("native empirical-Bernstein bound returned non-object JSON")
    return EmpiricalBernsteinBound(
        samples=int_field(payload, "samples"),
        candidates_tested=int_field(payload, "candidates_tested"),
        mean_gain=float_field(payload, "mean_gain"),
        variance=float_field(payload, "variance"),
        radius=float_field(payload, "radius"),
        adapt_loss=float_field(payload, "adapt_loss"),
        drift_loss=float_field(payload, "drift_loss"),
        leak_loss=float_field(payload, "leak_loss"),
        selection_loss=float_field(payload, "selection_loss"),
        lcb=float_field(payload, "lcb"),
        ucb=float_field(payload, "ucb"),
    )


def int_field(payload: Mapping[str, object], key: str) -> int:
    """Read an integer JSON field without accepting bools."""
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"expected numeric JSON field {key!r}")


__all__ = [
    "EmpiricalBernsteinBound",
    "FrontierReport",
    "OnlineFrontierEstimator",
    "eb_seq_bound",
    "online_replay_frontier_report",
    "replay_frontier_report",
    "replay_gains",
]
