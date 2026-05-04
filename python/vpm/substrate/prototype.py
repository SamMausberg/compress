"""Trainable non-transformer substrate prototype for C0 arithmetic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vpm.tasks.c0 import C0Task

OPERATIONS = ("add", "mul")
FEATURES = len(OPERATIONS) + 2


@dataclass(frozen=True)
class OperationProposal:
    """Operation proposed by the substrate, with calibrated-soft confidence."""

    operation: str
    confidence: float

    def to_dict(self) -> dict[str, float | str]:
        """JSON-friendly proposal."""
        return {"operation": self.operation, "confidence": self.confidence}


class ArithmeticProposalNet(nn.Module):
    """Tiny recurrent typed-event encoder with an operation head."""

    def __init__(self, hidden_dim: int = 32, scale: float = 1.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = max(float(scale), 1.0)
        self.input_proj = nn.Linear(FEATURES, hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.slot_gate = nn.Linear(hidden_dim, hidden_dim)
        self.op_head = nn.Linear(hidden_dim, len(OPERATIONS))

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """Return operation logits from typed event sequences."""
        hidden = events.new_zeros((events.shape[0], self.hidden_dim))
        for step in range(events.shape[1]):
            encoded = torch.tanh(self.input_proj(events[:, step, :]))
            hidden = self.cell(encoded, hidden)
        slots = torch.tanh(self.slot_gate(hidden)) * hidden
        return self.op_head(slots)


def event_tensor(task: C0Task, scale: float, device: torch.device) -> torch.Tensor:
    """Encode one typed arithmetic task as operation/argument events."""
    events = torch.zeros((3, FEATURES), dtype=torch.float32, device=device)
    events[0, operation_index(task.operation)] = 1.0
    events[1, len(OPERATIONS)] = float(task.left) / scale
    events[1, len(OPERATIONS) + 1] = -1.0 if task.left < 0 else 1.0
    events[2, len(OPERATIONS)] = float(task.right) / scale
    events[2, len(OPERATIONS) + 1] = -1.0 if task.right < 0 else 1.0
    return events


def batch_tensors(
    tasks: list[C0Task],
    scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build event and target tensors for a task batch."""
    events = torch.stack([event_tensor(task, scale, device) for task in tasks])
    targets = torch.tensor(
        [operation_index(task.operation) for task in tasks],
        dtype=torch.long,
        device=device,
    )
    return events, targets


def predict_operation(
    model: ArithmeticProposalNet,
    task: C0Task,
    device: torch.device | None = None,
) -> OperationProposal:
    """Predict one operation without assigning certificate authority."""
    model_device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        events = event_tensor(task, model.scale, model_device).unsqueeze(0)
        probs = torch.softmax(model(events), dim=-1).squeeze(0)
        confidence, index = probs.max(dim=0)
    return OperationProposal(OPERATIONS[int(index.item())], float(confidence.item()))


def operation_index(operation: str) -> int:
    """Return the operation class index."""
    if operation not in OPERATIONS:
        raise ValueError(f"unsupported C0 arithmetic operation: {operation}")
    return OPERATIONS.index(operation)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a training/inference device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def save_prototype(model: ArithmeticProposalNet, path: Path) -> Path:
    """Save model weights without using pickle-based torch serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({"hidden_dim": model.hidden_dim, "scale": model.scale})
    arrays: dict[str, np.ndarray] = {"__metadata__": np.array(metadata)}
    for key, tensor in model.state_dict().items():
        arrays[key] = tensor.detach().cpu().numpy()
    np.savez(path, **arrays)
    return path


def load_prototype(path: Path, device: str = "auto") -> ArithmeticProposalNet:
    """Load a saved prototype model."""
    target = resolve_device(device)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["__metadata__"].item()))
        model = ArithmeticProposalNet(
            hidden_dim=int(metadata["hidden_dim"]),
            scale=float(metadata["scale"]),
        )
        state = {key: torch.as_tensor(data[key]) for key in data.files if key != "__metadata__"}
    model.load_state_dict(state)
    return model.to(target)


__all__ = [
    "OPERATIONS",
    "ArithmeticProposalNet",
    "OperationProposal",
    "batch_tensors",
    "load_prototype",
    "predict_operation",
    "resolve_device",
    "save_prototype",
]
