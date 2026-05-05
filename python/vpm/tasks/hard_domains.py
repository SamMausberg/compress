"""Held-out hard-domain tasks for M6 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from vpm.tasks.spec import StageSpec


class HardDomain(StrEnum):
    """Hard-domain suite categories."""

    RESEARCH_MATH = "research_math"
    FORMAL = "formal"
    TOOL_USE = "tool_use"
    SOURCE_GROUNDED = "source_grounded"


@dataclass(frozen=True)
class HardDomainTask:
    """One verifier-backed hard-domain probe."""

    task_id: str
    domain: HardDomain
    prompt: str
    expected: str
    evidence: tuple[str, ...]
    baseline_answer: str

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly task."""
        return {
            "task_id": self.task_id,
            "domain": self.domain.value,
            "prompt": self.prompt,
            "expected": self.expected,
            "evidence": self.evidence,
            "baseline_answer": self.baseline_answer,
        }


def hard_domain_curriculum() -> tuple[HardDomainTask, ...]:
    """Return a small held-out hard-domain suite."""
    return (
        HardDomainTask(
            task_id="hard-math-gauss-20",
            domain=HardDomain.RESEARCH_MATH,
            prompt="Find the closed-form value of 1 + ... + 20.",
            expected="210",
            evidence=("formula:sum_1_to_n=n*(n+1)/2", "n=20"),
            baseline_answer="20",
        ),
        HardDomainTask(
            task_id="hard-formal-modus-ponens",
            domain=HardDomain.FORMAL,
            prompt="Given p and p->q, derive the certified conclusion.",
            expected="q",
            evidence=("premise:p", "premise:p_implies_q"),
            baseline_answer="p",
        ),
        HardDomainTask(
            task_id="hard-tool-square-12",
            domain=HardDomain.TOOL_USE,
            prompt="Use the audited calculator result square(12)=144.",
            expected="144",
            evidence=("tool:calculator:square(12)=144", "authority:data"),
            baseline_answer="12",
        ),
        HardDomainTask(
            task_id="hard-source-water",
            domain=HardDomain.SOURCE_GROUNDED,
            prompt="Answer from the audited chemistry corpus: formula(water).",
            expected="H2O",
            evidence=("audit:chem:water:formula=H2O",),
            baseline_answer="water",
        ),
    )


def stage_spec() -> StageSpec:
    """Runtime metadata for the M6 adversarial/release probe suite."""
    return StageSpec(
        name="M6",
        summary="red-team replay, ablations, and held-out hard-domain probes",
        executable=True,
        implemented_components=(
            "criterion1-failure-suite",
            "control-ablation-replay",
            "hard-domain-probes",
            "external-llm-task-export",
        ),
    )


__all__ = ["HardDomain", "HardDomainTask", "hard_domain_curriculum", "stage_spec"]
