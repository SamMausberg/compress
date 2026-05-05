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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from vpm.tasks.c0 import C0Task, arithmetic_task, concat_task, curriculum, equality_task
from vpm.tasks.c1 import hidden_schema_curriculum
from vpm.tasks.c2 import active_curriculum, active_test
from vpm.tasks.c3 import policy_probe_curriculum
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
    def target_operation(self) -> str:
        """Operation family covered by the replay set."""
        if not self.replay_tasks:
            return self.operation
        first = self.replay_tasks[0].operation
        if all(task.operation == first for task in self.replay_tasks):
            return first
        return self.operation

    @property
    def observation(self) -> str:
        """Compact macro observation for audit logs."""
        replay = ",".join(task.task_id for task in self.replay_tasks)
        return (
            f"macro {self.macro_key} op={self.operation}"
            f" target={self.target_operation} replay={replay}"
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly macro candidate."""
        return {
            "task_id": self.task_id,
            "macro_key": self.macro_key,
            "operation": self.operation,
            "target_operation": self.target_operation,
            "replay_tasks": [task.task_id for task in self.replay_tasks],
            "expected_admitted": self.expected_admitted,
        }


@dataclass(frozen=True)
class CrossStageReplayTask:
    """One scheduled replay task with its expected gate behavior."""

    stage: str
    task: C0Task
    expected_gate_passed: bool = True
    labels: tuple[str, ...] = ("data",)
    risk_items: tuple[tuple[str, float], ...] = ()

    @property
    def risk(self) -> dict[str, float]:
        """Risk map passed to the verifier gate."""
        return dict(self.risk_items)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly scheduled replay task."""
        return {
            "stage": self.stage,
            "task_id": self.task.task_id,
            "operation": self.task.operation,
            "expected_gate_passed": self.expected_gate_passed,
            "labels": self.labels,
            "risk": self.risk,
        }


@dataclass(frozen=True)
class CrossStageReplayBatch:
    """Replay tasks sourced from one curriculum stage."""

    stage: str
    tasks: tuple[CrossStageReplayTask, ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly replay batch."""
        return {
            "stage": self.stage,
            "tasks": [task.to_dict() for task in self.tasks],
        }


@dataclass(frozen=True)
class CrossStageReplayPlan:
    """Deterministic C5 replay schedule across earlier curriculum stages."""

    macro_key: str
    implementation_operation: str
    target_operation: str
    batches: tuple[CrossStageReplayBatch, ...]

    @property
    def scheduled_tasks(self) -> tuple[CrossStageReplayTask, ...]:
        """Flattened replay tasks in execution order."""
        return tuple(task for batch in self.batches for task in batch.tasks)

    @property
    def curriculum_stages(self) -> tuple[str, ...]:
        """Earlier curriculum stages that contribute replay tasks."""
        return tuple(
            batch.stage for batch in self.batches if batch.stage != "C5-seed" and batch.tasks
        )

    @property
    def cross_stage_covered(self) -> bool:
        """True when replay spans at least two earlier curriculum stages."""
        return len(self.curriculum_stages) >= 2

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly replay plan."""
        return {
            "macro_key": self.macro_key,
            "implementation_operation": self.implementation_operation,
            "target_operation": self.target_operation,
            "replay_tasks": len(self.scheduled_tasks),
            "curriculum_stages": self.curriculum_stages,
            "cross_stage_covered": self.cross_stage_covered,
            "batches": [batch.to_dict() for batch in self.batches],
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


def cross_stage_replay_plan(
    candidate: C5MacroCandidate,
    *,
    per_stage_limit: int = 2,
) -> CrossStageReplayPlan:
    """Schedule seed and cross-stage replay tasks for one macro candidate."""
    target_operation = candidate.target_operation
    batches = [
        CrossStageReplayBatch(
            "C5-seed",
            tuple(
                CrossStageReplayTask(
                    stage="C5-seed",
                    task=task,
                    expected_gate_passed=True,
                )
                for task in candidate.replay_tasks
            ),
        )
    ]
    batches.extend(cross_stage_replay_batches(target_operation, per_stage_limit=per_stage_limit))
    return CrossStageReplayPlan(
        macro_key=candidate.macro_key,
        implementation_operation=candidate.operation,
        target_operation=target_operation,
        batches=tuple(batch for batch in batches if batch.tasks),
    )


def cross_stage_replay_batches(
    target_operation: str,
    *,
    per_stage_limit: int = 2,
) -> tuple[CrossStageReplayBatch, ...]:
    """Build deterministic replay batches from executable earlier stages."""
    return (
        CrossStageReplayBatch(
            "C0",
            replay_tasks_from_c0("C0", curriculum(), target_operation, per_stage_limit),
        ),
        CrossStageReplayBatch(
            "C1",
            replay_tasks_from_c0(
                "C1",
                (task.to_c0_task() for task in hidden_schema_curriculum(limit=1)),
                target_operation,
                per_stage_limit,
            ),
        ),
        CrossStageReplayBatch(
            "C2",
            replay_tasks_from_c0(
                "C2",
                c2_replay_tasks(target_operation),
                target_operation,
                per_stage_limit,
            ),
        ),
        CrossStageReplayBatch(
            "C3",
            c3_replay_tasks(target_operation, per_stage_limit),
        ),
    )


def replay_tasks_from_c0(
    stage: str,
    tasks: Iterable[C0Task],
    target_operation: str,
    limit: int,
) -> tuple[CrossStageReplayTask, ...]:
    """Select a bounded, deterministic verifier-backed replay set."""
    selected: list[CrossStageReplayTask] = []
    seen: set[str] = set()
    for task in tasks:
        if task.operation != target_operation or task.task_id in seen:
            continue
        selected.append(CrossStageReplayTask(stage=stage, task=task))
        seen.add(task.task_id)
        if len(selected) >= limit:
            break
    return tuple(selected)


def c2_replay_tasks(target_operation: str) -> tuple[C0Task, ...]:
    """Expose C2 active-test tasks only after the support set narrows."""
    tasks: list[C0Task] = []
    for task in active_curriculum(limit=1):
        trace = active_test(task)
        if task.operation == target_operation and trace.selected_operation == target_operation:
            tasks.append(task.to_c0_task())
    return tuple(tasks)


def c3_replay_tasks(
    target_operation: str,
    limit: int,
) -> tuple[CrossStageReplayTask, ...]:
    """Replay C3 policy controls that share the target operation."""
    selected: list[CrossStageReplayTask] = []
    for probe in policy_probe_curriculum():
        if probe.task.operation != target_operation:
            continue
        selected.append(
            CrossStageReplayTask(
                stage="C3",
                task=probe.task,
                expected_gate_passed=probe.expected_pass,
                labels=probe.labels,
                risk_items=risk_items(probe.risk),
            )
        )
        if len(selected) >= limit:
            break
    return tuple(selected)


def risk_items(risk: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    """Stable immutable representation of a risk map."""
    return tuple(sorted(risk.items()))


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
            "online-frontier-estimator",
            "cross-stage-replay-scheduler",
        ),
        blockers=(),
    )


__all__ = [
    "C5MacroCandidate",
    "CrossStageReplayBatch",
    "CrossStageReplayPlan",
    "CrossStageReplayTask",
    "cross_stage_replay_batches",
    "cross_stage_replay_plan",
    "macro_replay_curriculum",
    "stage_spec",
]
