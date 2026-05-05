"""Curriculum stage ``C_2`` — partially observed and active worlds.

From §8 of ``docs/architecture/08-training-system.md``:

> ``C_2``: noisy/partial observations, active tests, small causal worlds,
> and verifiable planning.

Exit gate: support guard (§5 eqs. 92–95) fires at the calibrated rate;
test-selection ``e^*`` (§5 eq. 100) reduces certified posterior cost on
held-out trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass

from vpm.tasks.c0 import C0Task, C0Value, value_token
from vpm.tasks.spec import StageSpec

C2_OPERATIONS = ("add", "mul", "concat", "eq")
GridPoint = tuple[int, int]


@dataclass(frozen=True)
class C2Task:
    """Partially observed active-test task backed by the C0 verifier."""

    task_id: str
    left: C0Value
    right: C0Value
    expected: C0Value
    operation: str
    candidates: tuple[str, ...]

    @property
    def observation(self) -> str:
        """Partial observation before the active test reveals expected output."""
        return (
            f"{value_token(self.left)} {value_token(self.right)}"
            f" candidates={','.join(self.candidates)}"
        )

    def to_c0_task(self, operation: str | None = None) -> C0Task:
        """Expose a verifier-backed C0 task after active-test selection."""
        return C0Task(
            task_id=self.task_id,
            left=self.left,
            right=self.right,
            expected=self.expected,
            operation=operation or self.operation,
        )


@dataclass(frozen=True)
class ActiveTestTrace:
    """Support-set reduction produced by an active test."""

    task_id: str
    selected_operation: str | None
    candidates_before: tuple[str, ...]
    candidates_after: tuple[str, ...]
    expected: C0Value
    support_reduced: bool

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly active-test trace."""
        return {
            "task_id": self.task_id,
            "selected_operation": self.selected_operation,
            "candidates_before": self.candidates_before,
            "candidates_after": self.candidates_after,
            "expected": self.expected,
            "support_reduced": self.support_reduced,
        }


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


@dataclass(frozen=True)
class GridPlanStep:
    """One action in a bounded grid plan."""

    action: str
    state: GridPoint

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly plan step."""
        return {"action": self.action, "state": self.state}


@dataclass(frozen=True)
class GridPlanningTask:
    """Small verifiable multi-step planning task."""

    task_id: str
    width: int
    height: int
    start: GridPoint
    goal: GridPoint
    blocked: tuple[GridPoint, ...]
    expected_min_steps: int

    @property
    def observation(self) -> str:
        """Compact partial planning observation."""
        blocked = ",".join(f"{x}:{y}" for x, y in self.blocked)
        return (
            f"grid {self.width}x{self.height} start={self.start} goal={self.goal} blocked={blocked}"
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly planning task."""
        return {
            "task_id": self.task_id,
            "width": self.width,
            "height": self.height,
            "start": self.start,
            "goal": self.goal,
            "blocked": self.blocked,
            "expected_min_steps": self.expected_min_steps,
        }


@dataclass(frozen=True)
class PlanningTrace:
    """Bounded multi-step planning certificate."""

    task_id: str
    start: GridPoint
    goal: GridPoint
    actions: tuple[str, ...]
    states: tuple[GridPoint, ...]
    expected_min_steps: int
    reached_goal: bool
    avoided_blocked: bool
    multi_step: bool
    passed: bool

    @property
    def plan_length(self) -> int:
        """Number of actions in the plan."""
        return len(self.actions)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly planning trace."""
        return {
            "task_id": self.task_id,
            "start": self.start,
            "goal": self.goal,
            "actions": self.actions,
            "states": self.states,
            "plan_length": self.plan_length,
            "expected_min_steps": self.expected_min_steps,
            "reached_goal": self.reached_goal,
            "avoided_blocked": self.avoided_blocked,
            "multi_step": self.multi_step,
            "passed": self.passed,
        }


def active_test(task: C2Task) -> ActiveTestTrace:
    """Reveal expected output and narrow executable candidates by typed equality."""
    outputs = candidate_outputs(task.left, task.right, task.candidates)
    narrowed = tuple(
        operation for operation, value in outputs if same_typed_value(value, task.expected)
    )
    return ActiveTestTrace(
        task_id=task.task_id,
        selected_operation=narrowed[0] if len(narrowed) == 1 else None,
        candidates_before=tuple(operation for operation, _ in outputs),
        candidates_after=narrowed,
        expected=task.expected,
        support_reduced=len(narrowed) < len(outputs),
    )


def active_curriculum(limit: int = 3) -> list[C2Task]:
    """Build deterministic C2 partial-observation tasks."""
    numbers = list(range(-limit, limit + 1))
    text_values = [f"p{index}" for index in range(0, (limit * 2) + 1)]
    tasks: list[C2Task] = []
    for left in numbers:
        for right in numbers:
            if left + right != left * right:
                tasks.append(active_task(left, right, left + right, "add"))
                tasks.append(active_task(left, right, left * right, "mul"))
            tasks.append(active_task(left, right, left == right, "eq"))
    for left in text_values:
        for right in text_values:
            tasks.append(active_task(left, right, left + right, "concat"))
            tasks.append(active_task(left, right, left == right, "eq"))
    return tasks


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


def planning_curriculum() -> tuple[GridPlanningTask, ...]:
    """Build bounded multi-step planning probes."""
    return (
        GridPlanningTask(
            task_id="c2-plan-around-wall",
            width=3,
            height=3,
            start=(0, 0),
            goal=(2, 1),
            blocked=((1, 0),),
            expected_min_steps=3,
        ),
        GridPlanningTask(
            task_id="c2-plan-corridor",
            width=4,
            height=3,
            start=(0, 0),
            goal=(3, 2),
            blocked=((1, 0), (1, 1)),
            expected_min_steps=5,
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


def plan_grid_task(task: GridPlanningTask) -> PlanningTrace:
    """Find and verify a shortest bounded grid plan."""
    blocked = set(task.blocked)
    queue: list[tuple[GridPoint, tuple[GridPlanStep, ...]]] = [(task.start, ())]
    seen = {task.start}
    plan: tuple[GridPlanStep, ...] = ()
    while queue:
        state, steps = queue.pop(0)
        if state == task.goal:
            plan = steps
            break
        for action, neighbor in grid_neighbors(task, state):
            if neighbor in seen or neighbor in blocked:
                continue
            seen.add(neighbor)
            queue.append((neighbor, (*steps, GridPlanStep(action, neighbor))))

    states = tuple(step.state for step in plan)
    actions = tuple(step.action for step in plan)
    reached_goal = bool(states) and states[-1] == task.goal
    avoided_blocked = all(state not in blocked for state in states)
    multi_step = len(actions) > 1
    passed = (
        reached_goal and avoided_blocked and multi_step and len(actions) == task.expected_min_steps
    )
    return PlanningTrace(
        task_id=task.task_id,
        start=task.start,
        goal=task.goal,
        actions=actions,
        states=states,
        expected_min_steps=task.expected_min_steps,
        reached_goal=reached_goal,
        avoided_blocked=avoided_blocked,
        multi_step=multi_step,
        passed=passed,
    )


def active_task(
    left: C0Value,
    right: C0Value,
    expected: C0Value,
    operation: str,
) -> C2Task:
    """Construct one partial-observation active-test task."""
    candidates = tuple(
        operation for operation in C2_OPERATIONS if operation_applicable(operation, left, right)
    )
    return C2Task(
        task_id=f"c2-active-{value_token(left)}-{value_token(right)}-{value_token(expected)}",
        left=left,
        right=right,
        expected=expected,
        operation=operation,
        candidates=candidates,
    )


def candidate_outputs(
    left: C0Value,
    right: C0Value,
    candidates: tuple[str, ...],
) -> tuple[tuple[str, C0Value], ...]:
    """Evaluate supported candidate operations without assigning authority."""
    outputs: list[tuple[str, C0Value]] = []
    for operation in candidates:
        if not operation_applicable(operation, left, right):
            continue
        if operation == "add":
            outputs.append((operation, int(left) + int(right)))
        elif operation == "mul":
            outputs.append((operation, int(left) * int(right)))
        elif operation == "concat":
            outputs.append((operation, str(left) + str(right)))
        elif operation == "eq":
            outputs.append((operation, left == right))
    return tuple(outputs)


def operation_applicable(operation: str, left: C0Value, right: C0Value) -> bool:
    """Return true when an operation is type-valid for the observed operands."""
    if operation in {"add", "mul"}:
        return (
            isinstance(left, int)
            and isinstance(right, int)
            and not isinstance(left, bool)
            and not isinstance(right, bool)
        )
    if operation == "concat":
        return isinstance(left, str) and isinstance(right, str)
    return operation == "eq"


def same_typed_value(left: C0Value, right: C0Value) -> bool:
    """Compare values without allowing bool/int equality coercion."""
    return type(left) is type(right) and left == right


def mean_outcome(samples: tuple[CausalObservation, ...], *, treatment: int) -> float:
    """Return mean outcome for one intervention value."""
    matches = tuple(sample.outcome for sample in samples if sample.treatment == treatment)
    return sum(matches) / len(matches) if matches else 0.0


def grid_neighbors(
    task: GridPlanningTask,
    state: GridPoint,
) -> tuple[tuple[str, GridPoint], ...]:
    """Return in-bounds grid neighbors in deterministic action order."""
    x, y = state
    candidates = (
        ("right", (x + 1, y)),
        ("down", (x, y + 1)),
        ("left", (x - 1, y)),
        ("up", (x, y - 1)),
    )
    return tuple(
        (action, point)
        for action, point in candidates
        if 0 <= point[0] < task.width and 0 <= point[1] < task.height
    )


def stage_spec() -> StageSpec:
    """Runtime metadata for the C2 curriculum stage."""
    return StageSpec(
        name="C2",
        summary="partial observations, active tests, causal worlds",
        executable=True,
        implemented_components=(
            "partial-observation-generators",
            "active-test-selector",
            "uncertainty-action-scoring",
            "evc-halt-rule",
            "support-set-reduction",
            "c0-verifier-bridge",
            "noisy-causal-worlds",
            "multi-step-planning",
        ),
        blockers=(),
    )


__all__ = [
    "ActiveTestTrace",
    "C2Task",
    "CausalObservation",
    "CausalWorldTrace",
    "GridPlanStep",
    "GridPlanningTask",
    "NoisyCausalWorld",
    "PlanningTrace",
    "active_curriculum",
    "active_task",
    "active_test",
    "candidate_outputs",
    "causal_world_curriculum",
    "identify_causal_world",
    "plan_grid_task",
    "planning_curriculum",
    "same_typed_value",
    "stage_spec",
]
