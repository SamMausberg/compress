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

from vpm.tasks.c0 import (
    C0Task,
    C0Value,
    addition_task,
    arithmetic_task,
    concat_task,
    curriculum,
    equality_task,
    hidden_task,
    multiplication_task,
    typed_hidden_task,
    typed_task,
)
from vpm.tasks.c0 import stage_spec as c0_stage
from vpm.tasks.c1 import C1Task, as_c0_tasks, hidden_schema_curriculum, schema_split
from vpm.tasks.c1 import stage_spec as c1_stage
from vpm.tasks.c2 import ActiveTestTrace, C2Task, active_curriculum, active_test
from vpm.tasks.c2 import stage_spec as c2_stage
from vpm.tasks.c3 import C3PolicyProbe, policy_probe_curriculum
from vpm.tasks.c3 import stage_spec as c3_stage
from vpm.tasks.c4 import C4DialogueTask, dialogue_curriculum, gate_dialogue
from vpm.tasks.c4 import stage_spec as c4_stage
from vpm.tasks.c5 import C5MacroCandidate, macro_replay_curriculum
from vpm.tasks.c5 import stage_spec as c5_stage
from vpm.tasks.hard_domains import HardDomain, HardDomainTask, hard_domain_curriculum
from vpm.tasks.spec import StageSpec


def stages() -> tuple[StageSpec, ...]:
    """Return runtime-visible metadata for every curriculum package."""
    return (c0_stage(), c1_stage(), c2_stage(), c3_stage(), c4_stage(), c5_stage())


__all__ = [
    "ActiveTestTrace",
    "C0Task",
    "C0Value",
    "C1Task",
    "C2Task",
    "C3PolicyProbe",
    "C4DialogueTask",
    "C5MacroCandidate",
    "HardDomain",
    "HardDomainTask",
    "StageSpec",
    "active_curriculum",
    "active_test",
    "addition_task",
    "arithmetic_task",
    "as_c0_tasks",
    "concat_task",
    "curriculum",
    "dialogue_curriculum",
    "equality_task",
    "gate_dialogue",
    "hard_domain_curriculum",
    "hidden_schema_curriculum",
    "hidden_task",
    "macro_replay_curriculum",
    "multiplication_task",
    "policy_probe_curriculum",
    "schema_split",
    "stages",
    "typed_hidden_task",
    "typed_task",
]
