"""Local matched neural baselines for C1 splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from torch import nn

from vpm.infer import run_task_candidate
from vpm.substrate.prototype import (
    FEATURES,
    OPERATIONS,
    OperationProposal,
    batch_tensors,
    event_tensor,
)
from vpm.tasks.c0 import C0Task
from vpm.verifiers import gate_passed

type TorchDevice = Any
type TorchTensor = Any
_TORCH = cast(Any, torch)


class OperationModel(Protocol):
    """Protocol for local operation-proposal baselines."""

    scale: float

    def parameters(self) -> Any:
        """Return model parameters."""
        ...

    def train(self, mode: bool = True) -> Any:
        """Switch training mode."""
        ...

    def __call__(self, events: TorchTensor) -> TorchTensor:
        """Return operation logits."""
        ...


@dataclass(frozen=True)
class NeuralBaselineConfig:
    """Small local baseline training configuration."""

    epochs: int = 2
    hidden_dim: int = 8
    learning_rate: float = 0.03
    seed: int = 17
    device: str = "auto"


@dataclass(frozen=True)
class NeuralBaselineReport:
    """One local neural baseline report."""

    name: str
    solve_rate: float
    operation_accuracy: float
    mean_candidates: float
    compute_units: float
    train_tasks: int
    heldout_tasks: int


class TransformerOperationBaseline(nn.Module):
    """Tiny self-attention operation baseline."""

    def __init__(self, hidden_dim: int = 16, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = max(float(scale), 1.0)
        self.input_proj = nn.Linear(FEATURES, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.op_head = nn.Linear(hidden_dim, len(OPERATIONS))

    def forward(self, events: TorchTensor) -> TorchTensor:
        """Return operation logits."""
        encoded = _TORCH.tanh(self.input_proj(events))
        attended, _ = self.attention(encoded, encoded, encoded, need_weights=False)
        pooled = self.norm(encoded + attended).mean(dim=1)
        return self.op_head(pooled)


class SSMOperationBaseline(nn.Module):
    """Tiny selective state-space-style operation baseline."""

    def __init__(self, hidden_dim: int = 16, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = max(float(scale), 1.0)
        self.input_proj = nn.Linear(FEATURES, hidden_dim)
        self.decay = nn.Parameter(_TORCH.zeros(hidden_dim))
        self.gate = nn.Linear(FEATURES, hidden_dim)
        self.op_head = nn.Linear(hidden_dim, len(OPERATIONS))

    def forward(self, events: TorchTensor) -> TorchTensor:
        """Return operation logits."""
        hidden = events.new_zeros((events.shape[0], self.decay.shape[0]))
        decay = _TORCH.sigmoid(self.decay).unsqueeze(0)
        for step in range(events.shape[1]):
            row = events[:, step, :]
            update = _TORCH.tanh(self.input_proj(row))
            gate = _TORCH.sigmoid(self.gate(row))
            hidden = (decay * hidden) + (gate * update)
        return self.op_head(hidden)


def train_local_neural_baselines(
    train_tasks: list[C0Task],
    heldout_tasks: list[C0Task],
    *,
    scale: float,
    config: NeuralBaselineConfig | None = None,
) -> tuple[NeuralBaselineReport, ...]:
    """Train local transformer and SSM baselines on the same split."""
    cfg = config or NeuralBaselineConfig()
    return (
        train_one_baseline(
            "local-transformer-c1",
            TransformerOperationBaseline(cfg.hidden_dim, scale),
            train_tasks,
            heldout_tasks,
            cfg,
        ),
        train_one_baseline(
            "local-ssm-c1",
            SSMOperationBaseline(cfg.hidden_dim, scale),
            train_tasks,
            heldout_tasks,
            cfg,
        ),
    )


def train_one_baseline(
    name: str,
    model: OperationModel,
    train_tasks: list[C0Task],
    heldout_tasks: list[C0Task],
    cfg: NeuralBaselineConfig,
) -> NeuralBaselineReport:
    """Train and evaluate one local neural baseline."""
    _TORCH.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model = cast(Any, model).to(device)
    events, labels = batch_tensors(train_tasks, model.scale, device)
    optimizer = _TORCH.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(events), labels)
        loss.backward()
        optimizer.step()
    accuracy = local_operation_accuracy(model, heldout_tasks, device)
    certified = sum(
        gate_passed(
            run_task_candidate(
                task, predict_local_operation(model, task, device).operation
            ).native_report
        )
        for task in heldout_tasks
    )
    tasks = len(heldout_tasks)
    return NeuralBaselineReport(
        name=name,
        solve_rate=certified / tasks if tasks else 0.0,
        operation_accuracy=accuracy,
        mean_candidates=1.0,
        compute_units=float((len(train_tasks) * cfg.epochs) + tasks),
        train_tasks=len(train_tasks),
        heldout_tasks=tasks,
    )


def predict_local_operation(
    model: OperationModel,
    task: C0Task,
    device: TorchDevice,
) -> OperationProposal:
    """Predict one operation from a local baseline model."""
    cast(Any, model).eval()
    with _TORCH.no_grad():
        events = event_tensor(task, model.scale, device).unsqueeze(0)
        probs = _TORCH.softmax(model(events), dim=-1).squeeze(0)
        confidence, index = probs.max(dim=0)
    return OperationProposal(OPERATIONS[int(index.item())], float(confidence.item()))


def local_operation_accuracy(
    model: OperationModel,
    tasks: list[C0Task],
    device: TorchDevice,
) -> float:
    """Return operation accuracy for a local baseline model."""
    if not tasks:
        return 0.0
    correct = sum(
        predict_local_operation(model, task, device).operation == task.operation for task in tasks
    )
    return correct / len(tasks)


def resolve_device(requested: str) -> TorchDevice:
    """Resolve a PyTorch device for local baselines."""
    if requested == "auto":
        return _TORCH.device("cuda" if _TORCH.cuda.is_available() else "cpu")
    return _TORCH.device(requested)


__all__ = [
    "NeuralBaselineConfig",
    "NeuralBaselineReport",
    "SSMOperationBaseline",
    "TransformerOperationBaseline",
    "predict_local_operation",
    "train_local_neural_baselines",
]
