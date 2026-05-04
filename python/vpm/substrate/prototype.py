"""Trainable non-transformer substrate prototype for C0 typed tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from vpm.tasks.c0 import C0Task, C0Value

type TorchDevice = Any
type TorchTensor = Any
_TORCH = cast(Any, torch)
_NP_SAVEZ = cast(Any, np.savez)

OPERATIONS = ("add", "mul", "concat", "eq")
VALUE_KINDS = ("int", "text", "bool")
FEATURES = len(VALUE_KINDS) + 3


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

    def forward(self, events: TorchTensor) -> TorchTensor:
        """Return operation logits from typed event sequences."""
        hidden = events.new_zeros((events.shape[0], self.hidden_dim))
        for step in range(events.shape[1]):
            encoded = _TORCH.tanh(self.input_proj(events[:, step, :]))
            hidden = self.cell(encoded, hidden)
        slots = _TORCH.tanh(self.slot_gate(hidden)) * hidden
        return self.op_head(slots)


def event_tensor(task: C0Task, scale: float, device: TorchDevice) -> TorchTensor:
    """Encode one typed C0 task without leaking the operation label."""
    events = _TORCH.zeros((3, FEATURES), dtype=_TORCH.float32, device=device)
    fill_value_features(events[0], task.left, scale)
    fill_value_features(events[1], task.right, scale)
    fill_value_features(events[2], task.expected, scale)
    return events


def fill_value_features(row: TorchTensor, value: C0Value, scale: float) -> None:
    """Fill one operand row with type and compact value features."""
    scalar_offset = len(VALUE_KINDS)
    if isinstance(value, bool):
        row[VALUE_KINDS.index("bool")] = 1.0
        row[scalar_offset] = 1.0 if value else 0.0
        return
    if isinstance(value, int):
        row[VALUE_KINDS.index("int")] = 1.0
        row[scalar_offset] = abs(float(value)) / scale
        row[scalar_offset + 1] = -1.0 if value < 0 else 1.0
        return
    row[VALUE_KINDS.index("text")] = 1.0
    row[scalar_offset] = len(value) / scale
    row[scalar_offset + 2] = stable_text_feature(value)


def stable_text_feature(value: str) -> float:
    """Return a deterministic bounded text feature."""
    if not value:
        return 0.0
    total = sum((index + 1) * ord(char) for index, char in enumerate(value))
    return (total % 997) / 997.0


def batch_tensors(
    tasks: list[C0Task],
    scale: float,
    device: TorchDevice,
) -> tuple[TorchTensor, TorchTensor]:
    """Build event and target tensors for a task batch."""
    events = _TORCH.stack([event_tensor(task, scale, device) for task in tasks])
    targets = _TORCH.tensor(
        [operation_index(task.operation) for task in tasks],
        dtype=_TORCH.long,
        device=device,
    )
    return events, targets


def predict_operation(
    model: ArithmeticProposalNet,
    task: C0Task,
    device: TorchDevice | None = None,
) -> OperationProposal:
    """Predict one operation without assigning certificate authority."""
    model_device = device or next(model.parameters()).device
    model.eval()
    with _TORCH.no_grad():
        events = event_tensor(task, model.scale, model_device).unsqueeze(0)
        probs = _TORCH.softmax(model(events), dim=-1).squeeze(0)
        confidence, index = probs.max(dim=0)
    return OperationProposal(OPERATIONS[int(index.item())], float(confidence.item()))


def operation_index(operation: str) -> int:
    """Return the operation class index."""
    if operation not in OPERATIONS:
        raise ValueError(f"unsupported C0 typed operation: {operation}")
    return OPERATIONS.index(operation)


def resolve_device(requested: str = "auto") -> TorchDevice:
    """Resolve a training/inference device."""
    if requested == "auto":
        return _TORCH.device("cuda" if _TORCH.cuda.is_available() else "cpu")
    return _TORCH.device(requested)


def save_prototype(model: ArithmeticProposalNet, path: Path) -> Path:
    """Save model weights without using pickle-based torch serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({"hidden_dim": model.hidden_dim, "scale": model.scale})
    arrays: dict[str, np.ndarray] = {"__metadata__": np.array(metadata)}
    for key, tensor in model.state_dict().items():
        arrays[key] = tensor.detach().cpu().numpy()
    _NP_SAVEZ(path, **arrays)
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
        state = {key: _TORCH.as_tensor(data[key]) for key in data.files if key != "__metadata__"}
    model.load_state_dict(state)
    return model.to(target)


__all__ = [
    "OPERATIONS",
    "VALUE_KINDS",
    "ArithmeticProposalNet",
    "OperationProposal",
    "batch_tensors",
    "load_prototype",
    "predict_operation",
    "resolve_device",
    "save_prototype",
]
