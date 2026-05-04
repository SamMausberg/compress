"""Curriculum stage ``C_5`` — continual compression and replay.

From §8 of ``docs/architecture/08-training-system.md``:

> ``C_5``: continual compression with replay-safe macro admission.

Exit gate: a compression phase transition is observed (§9, last
paragraph): solved-task cost falls, active memory grows sublinearly,
held-out certified utility rises, rate-distortion frontier decreases,
and support / context / semantic / source / rebuttal / realization /
dependency / substrate losses remain bounded. Vector-risk budgets stay
valid and false-pass bounds do not rise.
"""

from __future__ import annotations

from dataclasses import dataclass

from vpm.tasks.c0 import C0Task, arithmetic_task, concat_task, equality_task
from vpm.tasks.spec import StageSpec


@dataclass(frozen=True)
class C5MacroCandidate:
    """Replay-bounded macro candidate for continual compression."""

    task_id: str
    macro_key: str
    operation: str
    replay_tasks: tuple[C0Task, ...]
    expected_admitted: bool

    @property
    def observation(self) -> str:
        """Compact macro observation for audit logs."""
        replay = ",".join(task.task_id for task in self.replay_tasks)
        return f"macro {self.macro_key} op={self.operation} replay={replay}"

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly macro candidate."""
        return {
            "task_id": self.task_id,
            "macro_key": self.macro_key,
            "operation": self.operation,
            "replay_tasks": [task.task_id for task in self.replay_tasks],
            "expected_admitted": self.expected_admitted,
        }


def macro_replay_curriculum() -> list[C5MacroCandidate]:
    """Build replay-safe and replay-unsafe macro admission probes."""
    return [
        C5MacroCandidate(
            task_id="c5-admit-add-macro",
            macro_key="macro:add",
            operation="add",
            replay_tasks=(
                arithmetic_task("add", 1, 2),
                arithmetic_task("add", -3, 5),
                arithmetic_task("add", 8, 13),
            ),
            expected_admitted=True,
        ),
        C5MacroCandidate(
            task_id="c5-admit-concat-macro",
            macro_key="macro:concat",
            operation="concat",
            replay_tasks=(concat_task("a", "b"), concat_task("left", "right")),
            expected_admitted=True,
        ),
        C5MacroCandidate(
            task_id="c5-admit-eq-macro",
            macro_key="macro:eq",
            operation="eq",
            replay_tasks=(equality_task(5, 5), equality_task("x", "y")),
            expected_admitted=True,
        ),
        C5MacroCandidate(
            task_id="c5-demote-wrong-mul-macro",
            macro_key="macro:add-as-mul",
            operation="add",
            replay_tasks=(arithmetic_task("mul", 2, 5), arithmetic_task("mul", 3, 4)),
            expected_admitted=False,
        ),
    ]


def stage_spec() -> StageSpec:
    """Runtime metadata for the C5 curriculum stage."""
    return StageSpec(
        name="C5",
        summary="continual compression and replay-safe macro admission",
        executable=True,
        implemented_components=(
            "memory-library",
            "canonicalization-witnesses",
            "replay-safe-macro-admission",
            "macro-demotion-replay",
            "frontier-movement-probe",
        ),
        blockers=("online frontier estimator", "cross-stage replay scheduler"),
    )


__all__ = ["C5MacroCandidate", "macro_replay_curriculum", "stage_spec"]
