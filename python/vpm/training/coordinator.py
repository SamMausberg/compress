"""Block-coordinate training schedule."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingBlock:
    """One hard-gated block-coordinate update."""

    name: str
    losses: tuple[str, ...]
    frozen_inputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly training block."""
        return {
            "name": self.name,
            "losses": self.losses,
            "frozen_inputs": self.frozen_inputs,
        }


def block_coordinate_plan() -> tuple[TrainingBlock, ...]:
    """Return the eqs. 171-174 block-coordinate plan."""
    return (
        TrainingBlock("language", ("ctx", "sem"), ("audit_labels", "hard_gates")),
        TrainingBlock(
            "evidence",
            ("src", "rebut", "ent", "real", "dep", "ver", "cal", "split"),
            ("audit_labels", "candidate_generator"),
        ),
        TrainingBlock(
            "substrate_compiler",
            ("free_energy", "base", "cmp", "trace", "value", "halt", "supp", "probe"),
            ("verifier_gates", "split_policy"),
        ),
        TrainingBlock(
            "render_memory",
            ("render", "mem", "safe", "front"),
            ("certified_atoms", "risk_gate"),
        ),
    )


__all__ = ["TrainingBlock", "block_coordinate_plan"]
