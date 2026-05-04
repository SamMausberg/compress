"""Curriculum tasks ``C_0`` … ``C_5``.

See the curriculum paragraph at the end of §8 of
``docs/architecture/08-training-system.md`` and the VPM-0 minimal
implementation list at the end of §9.

Layout:

- ``c0/`` — deterministic grids, strings, FSMs, small graphs,
  arithmetic, finite-state tasks, data transformations.
- ``c1/`` — hidden-schema splits, equality saturation, typed program
  synthesis, theorem-proving fragments.
- ``c2/`` — noisy/partial observations, active tests, small causal
  worlds, verifiable planning.
- ``c3/`` — formal tools, agent workflows, tool use, authority,
  declassification, prompt-injection, rollback, influence-risk,
  conflict.
- ``c4/`` — controlled dialogue/artifacts with context normal forms,
  semantic atom extraction, source-grounded QA, contradiction search,
  entailment checking, round-trip realization checking, calibrated
  uncertainty, intent-entropy gates.
- ``c5/`` — continual compression with replay-safe macro admission.

Advancement requires: held-out certified utility, calibrated anytime
false-pass bounds, calibrated dependence residuals, positive frontier
movement, sublinear active-memory growth, bounded support, context,
semantic, source, rebuttal, realization, dependency, and substrate
loss, vector-risk budget validity, and zero critical gate violations
in adversarial suites.
"""

from __future__ import annotations

from vpm.tasks.c0 import C0Task, addition_task, curriculum

__all__ = ["C0Task", "addition_task", "curriculum"]
