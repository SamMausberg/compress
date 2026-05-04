"""VPM-5.3 — Verifier-Native Compressive Predictive Mechanisms.

Reference implementation. See ``docs/architecture/`` for the spec this
package realises.

The package is organised by architecture section:

- :mod:`vpm.substrate`  — §3, the neural substrate ``VNS_θ``.
- :mod:`vpm.compiler`   — §5, the conversation-normal-form compiler ``q_φ``.
- :mod:`vpm.language`   — §4, context / semantic / source / rebuttal /
  realization gates.
- :mod:`vpm.retrieval`  — source and rebuttal retrievers.
- :mod:`vpm.verifiers`  — Python-side verifier implementations and the
  thin wrapper around :mod:`vpm._native.verify`.
- :mod:`vpm.memory`     — §7, two-tier active / archival library.
- :mod:`vpm.training`   — §8, losses, dual-price budget, curriculum.
- :mod:`vpm.evaluation` — §9, metrics, ablations, failure-mode
  detectors.
- :mod:`vpm.tasks`      — curriculum tasks ``C_0`` … ``C_5``.
- :mod:`vpm.infer`      — Appendix A, the compact inference loop.

The Rust core is exposed as :mod:`vpm._native`; the Python-side modules
above are thin, idiomatic wrappers around it.
"""

from __future__ import annotations

__version__ = "0.0.0"

from vpm.infer import InferenceResult, run_c0_add, run_task
from vpm.tasks import (
    C0Task,
    addition_task,
    arithmetic_task,
    curriculum,
    multiplication_task,
    stages,
)
from vpm.training import TrainingConfig, TrainingReport, train_c0_prototype

__all__ = [
    "C0Task",
    "InferenceResult",
    "TrainingConfig",
    "TrainingReport",
    "__version__",
    "addition_task",
    "arithmetic_task",
    "curriculum",
    "multiplication_task",
    "run_c0_add",
    "run_task",
    "stages",
    "train_c0_prototype",
]
