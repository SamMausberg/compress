"""Local installation and accelerator diagnostics."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any, cast

import torch

_TORCH = cast(Any, torch)


@dataclass(frozen=True)
class DiagnosticReport:
    """Runtime diagnostic payload for local VPM installs."""

    python: str
    torch: str
    torch_cuda: str | None
    cuda_available: bool
    cuda_device: str | None
    cuda_probe_ok: bool
    native_extension_ok: bool

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly diagnostics."""
        return asdict(self)


def collect_diagnostics() -> DiagnosticReport:
    """Inspect Python, PyTorch/CUDA, and the Rust native extension."""
    cuda_available = bool(_TORCH.cuda.is_available())
    cuda_device = str(_TORCH.cuda.get_device_name(0)) if cuda_available else None
    return DiagnosticReport(
        python=platform.python_version(),
        torch=str(_TORCH.__version__),
        torch_cuda=_TORCH.version.cuda,
        cuda_available=cuda_available,
        cuda_device=cuda_device,
        cuda_probe_ok=cuda_probe_ok(cuda_available),
        native_extension_ok=native_extension_ok(),
    )


def cuda_probe_ok(cuda_available: bool) -> bool:
    """Run a tiny CUDA matmul when a CUDA device is visible."""
    if not cuda_available:
        return False
    try:
        values = _TORCH.ones((2, 2), device="cuda")
        return float((values @ values).sum().item()) == 8.0
    except RuntimeError:
        return False


def native_extension_ok() -> bool:
    """Return true when the maturin-built Rust extension imports."""
    try:
        import_module("vpm._native")
    except ImportError:
        return False
    return True


__all__ = ["DiagnosticReport", "collect_diagnostics"]
