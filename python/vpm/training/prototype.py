"""End-to-end train/evaluate loop for the C0 trainable prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from vpm.infer import InferenceResult, run_task_candidate
from vpm.memory import MemoryLibrary
from vpm.substrate.prototype import (
    ArithmeticProposalNet,
    OperationProposal,
    batch_tensors,
    load_prototype,
    predict_operation,
    resolve_device,
    save_prototype,
)
from vpm.tasks.c0 import C0Task, arithmetic_task, concat_task, equality_task
from vpm.tasks.c1 import as_c0_tasks, schema_split
from vpm.training.prototype_metrics import (
    BaselineMetrics,
    CompressionMetrics,
    compression_frontier_metrics,
    matched_baselines,
)
from vpm.verifiers import RiskMap, gate_passed


@dataclass(frozen=True)
class TrainingConfig:
    """Local training configuration for the trainable C0 prototype."""

    limit: int = 8
    epochs: int = 80
    hidden_dim: int = 32
    learning_rate: float = 0.03
    seed: int = 7
    device: str = "auto"
    artifact: Path | None = None


@dataclass(frozen=True)
class PrototypeTrace:
    """One held-out proposal with verifier and memory outcome."""

    task_id: str
    expected_operation: str
    proposed_operation: str
    passed: bool
    rendered: str
    memory_key: str | None
    gate_reasons: tuple[str, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly audit trace."""
        return {
            "task_id": self.task_id,
            "expected_operation": self.expected_operation,
            "proposed_operation": self.proposed_operation,
            "passed": self.passed,
            "rendered": self.rendered,
            "memory_key": self.memory_key,
            "gate_reasons": self.gate_reasons,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class PrototypeEvalReport:
    """Held-out report for learned proposals and verifier-gated outcomes."""

    tasks: int
    certified: int
    rejected: int
    operation_accuracy: float
    mean_confidence: float
    macro_memory_active: int
    compression_ratio: float
    compression: CompressionMetrics
    baselines: tuple[BaselineMetrics, ...]
    traces: tuple[PrototypeTrace, ...]

    @property
    def solve_rate(self) -> float:
        """Verifier-certified solve rate."""
        return self.certified / self.tasks if self.tasks else 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly evaluation report."""
        return {
            "tasks": self.tasks,
            "certified": self.certified,
            "rejected": self.rejected,
            "solve_rate": self.solve_rate,
            "operation_accuracy": self.operation_accuracy,
            "mean_confidence": self.mean_confidence,
            "macro_memory_active": self.macro_memory_active,
            "compression_ratio": self.compression_ratio,
            "compression": self.compression.to_dict(),
            "baselines": [baseline.to_dict() for baseline in self.baselines],
            "traces": [trace.to_dict() for trace in self.traces],
        }


@dataclass(frozen=True)
class TrainingReport:
    """Training result plus held-out verifier/evaluation evidence."""

    epochs: int
    device: str
    train_tasks: int
    heldout_tasks: int
    initial_loss: float
    final_loss: float
    initial_accuracy: float
    final_accuracy: float
    heldout: PrototypeEvalReport
    artifact: str | None

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly training report."""
        return {
            "epochs": self.epochs,
            "device": self.device,
            "train_tasks": self.train_tasks,
            "heldout_tasks": self.heldout_tasks,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "initial_accuracy": self.initial_accuracy,
            "final_accuracy": self.final_accuracy,
            "heldout": self.heldout.to_dict(),
            "artifact": self.artifact,
        }


@dataclass(frozen=True)
class PrototypeInference:
    """One learned proposal routed through the verifier-native inference loop."""

    proposal: OperationProposal
    result: InferenceResult

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly inference payload."""
        return {"proposal": self.proposal.to_dict(), "result": self.result.to_dict()}


def train_c0_prototype(
    config: TrainingConfig | None = None,
) -> tuple[ArithmeticProposalNet, TrainingReport]:
    """Train a recurrent substrate proposal model on C0 curriculum tasks."""
    cfg = config or TrainingConfig()
    train_tasks, heldout_tasks = curriculum_split(cfg.limit)
    return train_prototype_split(cfg, train_tasks, heldout_tasks, scale=cfg.limit)


def train_c1_prototype(
    config: TrainingConfig | None = None,
) -> tuple[ArithmeticProposalNet, TrainingReport]:
    """Train the proposal model on executable C1 hidden-schema tasks."""
    cfg = config or TrainingConfig()
    train_tasks, heldout_tasks = schema_split(cfg.limit)
    return train_prototype_split(
        cfg,
        as_c0_tasks(train_tasks),
        as_c0_tasks(heldout_tasks),
        scale=cfg.limit,
    )


def train_prototype_split(
    cfg: TrainingConfig,
    train_tasks: list[C0Task],
    heldout_tasks: list[C0Task],
    scale: float,
) -> tuple[ArithmeticProposalNet, TrainingReport]:
    """Train a recurrent substrate proposal model on a prepared split."""
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = resolve_device(cfg.device)
    model = ArithmeticProposalNet(cfg.hidden_dim, scale=scale).to(device)
    events, labels = batch_tensors(train_tasks, model.scale, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    initial_loss = float(loss_fn(model(events), labels).item())
    initial_accuracy = operation_accuracy(model, train_tasks, device)
    final_loss = initial_loss
    for _ in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(events), labels)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    final_accuracy = operation_accuracy(model, train_tasks, device)
    artifact = str(save_prototype(model, cfg.artifact)) if cfg.artifact else None
    heldout = evaluate_prototype(model, train_tasks, heldout_tasks, device)
    return model, TrainingReport(
        epochs=cfg.epochs,
        device=str(device),
        train_tasks=len(train_tasks),
        heldout_tasks=len(heldout_tasks),
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_accuracy=initial_accuracy,
        final_accuracy=final_accuracy,
        heldout=heldout,
        artifact=artifact,
    )


def evaluate_saved_prototype(
    path: Path, limit: int = 8, device: str = "auto"
) -> PrototypeEvalReport:
    """Load and evaluate a saved prototype artifact."""
    model = load_prototype(path, device)
    train_tasks, heldout_tasks = curriculum_split(limit)
    return evaluate_prototype(model, train_tasks, heldout_tasks, resolve_device(device))


def evaluate_saved_c1_prototype(
    path: Path, limit: int = 8, device: str = "auto"
) -> PrototypeEvalReport:
    """Load and evaluate a saved prototype artifact on C1 hidden-schema tasks."""
    model = load_prototype(path, device)
    train_tasks, heldout_tasks = schema_split(limit)
    return evaluate_prototype(
        model,
        as_c0_tasks(train_tasks),
        as_c0_tasks(heldout_tasks),
        resolve_device(device),
    )


def run_learned_task(
    model: ArithmeticProposalNet,
    task: C0Task,
    labels: tuple[str, ...] = ("data",),
    risk: RiskMap | None = None,
) -> PrototypeInference:
    """Run one task using the learned operation proposal."""
    proposal = predict_operation(model, task)
    return PrototypeInference(
        proposal,
        run_task_candidate(task, proposal.operation, labels=labels, risk=risk),
    )


def evaluate_prototype(
    model: ArithmeticProposalNet,
    train_tasks: list[C0Task],
    heldout_tasks: list[C0Task],
    device: torch.device,
) -> PrototypeEvalReport:
    """Evaluate learned proposals against verifier-gated held-out tasks."""
    macro_memory = MemoryLibrary()
    certified = 0
    correct_ops = 0
    confidence_sum = 0.0
    traces: list[PrototypeTrace] = []
    for task in heldout_tasks:
        proposal = predict_operation(model, task, device)
        result = run_task_candidate(task, proposal.operation)
        passed = gate_passed(result.native_report)
        certified += int(passed)
        correct_ops += int(proposal.operation == task.operation)
        confidence_sum += proposal.confidence
        memory_key = None
        if passed:
            memory_key = f"macro:{proposal.operation}"
            macro_memory.admit(memory_key, proposal.operation, result.native_report)
        traces.append(
            PrototypeTrace(
                task_id=task.task_id,
                expected_operation=task.operation,
                proposed_operation=proposal.operation,
                passed=passed,
                rendered=result.rendered,
                memory_key=memory_key,
                gate_reasons=gate_reasons(result.native_report),
                errors=tuple(result.errors),
            )
        )

    tasks = len(heldout_tasks)
    baselines = matched_baselines(train_tasks, heldout_tasks)
    macro_active = len(macro_memory.active)
    compression = certified / macro_active if macro_active else 0.0
    compression_metrics = compression_frontier_metrics(tasks, certified, macro_active, baselines)
    return PrototypeEvalReport(
        tasks=tasks,
        certified=certified,
        rejected=tasks - certified,
        operation_accuracy=correct_ops / tasks if tasks else 0.0,
        mean_confidence=confidence_sum / tasks if tasks else 0.0,
        macro_memory_active=macro_active,
        compression_ratio=compression,
        compression=compression_metrics,
        baselines=baselines,
        traces=tuple(traces),
    )


def gate_reasons(report: dict[str, object]) -> tuple[str, ...]:
    """Extract gate reasons from a native report."""
    gate = report.get("gate")
    if not isinstance(gate, dict):
        return ()
    reasons = gate.get("reasons")
    if not isinstance(reasons, list):
        return ()
    return tuple(reason for reason in reasons if isinstance(reason, str))


def operation_accuracy(
    model: ArithmeticProposalNet,
    tasks: list[C0Task],
    device: torch.device,
) -> float:
    """Return operation prediction accuracy."""
    if not tasks:
        return 0.0
    return sum(
        predict_operation(model, task, device).operation == task.operation for task in tasks
    ) / len(tasks)


def curriculum_split(limit: int) -> tuple[list[C0Task], list[C0Task]]:
    """Build deterministic train/held-out C0 typed splits."""
    numbers = list(range(-limit, limit + 1))
    text_values = [f"t{index}" for index in range(0, (limit * 2) + 1)]
    tasks = typed_curriculum(numbers, text_values)
    train: list[C0Task] = []
    heldout: list[C0Task] = []
    for task in tasks:
        split_key = stable_task_key(task)
        (heldout if split_key % 5 == 0 else train).append(task)
    return train, heldout


def typed_curriculum(numbers: list[int], text_values: list[str]) -> list[C0Task]:
    """Generate the small typed C0 curriculum used by the prototype."""
    tasks: list[C0Task] = []
    for left in numbers:
        for right in numbers:
            if left + right != left * right:
                tasks.append(arithmetic_task("add", left, right))
                tasks.append(arithmetic_task("mul", left, right))
            tasks.append(equality_task(left, right))
    for left in text_values:
        for right in text_values:
            tasks.append(concat_task(left, right))
            tasks.append(equality_task(left, right))
    tasks.extend(equality_task(left, right) for left in (False, True) for right in (False, True))
    return tasks


def stable_task_key(task: C0Task) -> int:
    """Stable split key independent of Python hash randomization."""
    raw = f"{task.operation}:{task.left}:{task.right}"
    return sum((index + 1) * ord(char) for index, char in enumerate(raw))


__all__ = [
    "BaselineMetrics",
    "CompressionMetrics",
    "PrototypeEvalReport",
    "PrototypeInference",
    "TrainingConfig",
    "TrainingReport",
    "curriculum_split",
    "evaluate_prototype",
    "evaluate_saved_c1_prototype",
    "evaluate_saved_prototype",
    "run_learned_task",
    "train_c0_prototype",
    "train_c1_prototype",
]
