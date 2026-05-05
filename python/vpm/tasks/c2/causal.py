"""Noisy causal-world controls for C2 tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CausalObservation:
    """One intervention sample from a noisy binary causal world."""

    treatment: int
    outcome: int
    noisy: bool = False

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly causal observation."""
        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "noisy": self.noisy,
        }


@dataclass(frozen=True)
class NoisyCausalWorld:
    """Small noisy causal world with an exact intervention target."""

    task_id: str
    treatment_name: str
    outcome_name: str
    observations: tuple[CausalObservation, ...]
    expected_relation: str
    candidates: tuple[str, ...] = ("treatment_causes_outcome", "independent")

    @property
    def observation(self) -> str:
        """Partial causal observation with noise markers hidden from the solver."""
        pairs = ",".join(
            f"{self.treatment_name}={sample.treatment}->{self.outcome_name}={sample.outcome}"
            for sample in self.observations
        )
        return f"causal {pairs}"

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly noisy causal world."""
        return {
            "task_id": self.task_id,
            "treatment_name": self.treatment_name,
            "outcome_name": self.outcome_name,
            "observations": [sample.to_dict() for sample in self.observations],
            "expected_relation": self.expected_relation,
            "candidates": self.candidates,
        }


@dataclass(frozen=True)
class CausalWorldTrace:
    """Noisy causal-world support reduction and intervention certificate."""

    task_id: str
    candidates_before: tuple[str, ...]
    candidates_after: tuple[str, ...]
    expected_relation: str
    selected_relation: str | None
    clean_samples: int
    noisy_samples: int
    noise_rate: float
    noise_threshold: float
    intervention_effect: float
    support_reduced: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly causal-world trace."""
        return {
            "task_id": self.task_id,
            "candidates_before": self.candidates_before,
            "candidates_after": self.candidates_after,
            "expected_relation": self.expected_relation,
            "selected_relation": self.selected_relation,
            "clean_samples": self.clean_samples,
            "noisy_samples": self.noisy_samples,
            "noise_rate": self.noise_rate,
            "noise_threshold": self.noise_threshold,
            "intervention_effect": self.intervention_effect,
            "support_reduced": self.support_reduced,
            "passed": self.passed,
        }


def causal_world_curriculum() -> tuple[NoisyCausalWorld, ...]:
    """Build deterministic noisy causal-world intervention probes."""
    return (
        NoisyCausalWorld(
            task_id="c2-causal-sprinkler-wet",
            treatment_name="sprinkler",
            outcome_name="wet_grass",
            observations=(
                CausalObservation(0, 0),
                CausalObservation(0, 1, noisy=True),
                CausalObservation(1, 1),
                CausalObservation(1, 1),
            ),
            expected_relation="treatment_causes_outcome",
        ),
        NoisyCausalWorld(
            task_id="c2-causal-lamp-shadow",
            treatment_name="lamp",
            outcome_name="shadow",
            observations=(
                CausalObservation(0, 1),
                CausalObservation(0, 0, noisy=True),
                CausalObservation(1, 1),
                CausalObservation(1, 1),
            ),
            expected_relation="independent",
        ),
    )


def identify_causal_world(
    world: NoisyCausalWorld,
    *,
    noise_threshold: float = 0.25,
) -> CausalWorldTrace:
    """Identify a noisy binary causal relation from clean interventions."""
    clean = tuple(sample for sample in world.observations if not sample.noisy)
    low = mean_outcome(clean, treatment=0)
    high = mean_outcome(clean, treatment=1)
    effect = high - low
    if effect > 0.5:
        selected = "treatment_causes_outcome"
    elif abs(effect) <= 0.25:
        selected = "independent"
    else:
        selected = None
    candidates_after = (selected,) if selected else world.candidates
    noisy_samples = len(world.observations) - len(clean)
    noise_rate = noisy_samples / len(world.observations) if world.observations else 1.0
    support_reduced = len(candidates_after) < len(world.candidates)
    passed = (
        selected == world.expected_relation and support_reduced and noise_rate <= noise_threshold
    )
    return CausalWorldTrace(
        task_id=world.task_id,
        candidates_before=world.candidates,
        candidates_after=candidates_after,
        expected_relation=world.expected_relation,
        selected_relation=selected,
        clean_samples=len(clean),
        noisy_samples=noisy_samples,
        noise_rate=noise_rate,
        noise_threshold=noise_threshold,
        intervention_effect=effect,
        support_reduced=support_reduced,
        passed=passed,
    )


def mean_outcome(samples: tuple[CausalObservation, ...], *, treatment: int) -> float:
    """Return mean outcome for one intervention value."""
    matches = tuple(sample.outcome for sample in samples if sample.treatment == treatment)
    return sum(matches) / len(matches) if matches else 0.0


__all__ = [
    "CausalObservation",
    "CausalWorldTrace",
    "NoisyCausalWorld",
    "causal_world_curriculum",
    "identify_causal_world",
    "mean_outcome",
]
