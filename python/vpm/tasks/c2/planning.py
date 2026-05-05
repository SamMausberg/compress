"""Grid-planning controls for C2 tasks."""

from __future__ import annotations

from dataclasses import dataclass

GridPoint = tuple[int, int]


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


__all__ = [
    "GridPlanStep",
    "GridPlanningTask",
    "GridPoint",
    "PlanningTrace",
    "grid_neighbors",
    "plan_grid_task",
    "planning_curriculum",
]
