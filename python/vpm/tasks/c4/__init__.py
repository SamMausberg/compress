"""Curriculum stage ``C_4`` — controlled dialogue and rendered text.

From §8 of ``docs/architecture/08-training-system.md``:

> ``C_4``: controlled dialogue/artifacts with context normal forms,
> semantic atom extraction, source-grounded QA, contradiction search,
> entailment checking, round-trip realization checking, calibrated
> uncertainty, and intent-entropy gates.

Exit gate: ``Gate_NL`` (§4 eqs. 80–81; Proposition 2) holds on every
rendered transcript; no certified-mode atom is rendered without
entailment + source + rebuttal + round-trip witnesses.
"""

from __future__ import annotations

from dataclasses import dataclass

from vpm.tasks.spec import StageSpec


@dataclass(frozen=True)
class C4DialogueTask:
    """Controlled source-grounded dialogue task."""

    task_id: str
    question: str
    atom: str
    answer: str
    sources: tuple[str, ...]
    rebuttals: tuple[str, ...] = ()

    @property
    def observation(self) -> str:
        """Compact controlled-dialogue observation."""
        return f"ask {self.atom}"

    @property
    def evidence_text(self) -> str:
        """Text visible to the simple entailment witness."""
        return " ".join((self.atom, self.answer, *self.sources))


@dataclass(frozen=True)
class DialogueGateTrace:
    """Source/rebuttal/entailment/realization gate trace."""

    task_id: str
    rendered: str
    source_ok: bool
    rebuttal_ok: bool
    entailment_ok: bool
    realization_ok: bool
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly gate trace."""
        return {
            "task_id": self.task_id,
            "rendered": self.rendered,
            "source_ok": self.source_ok,
            "rebuttal_ok": self.rebuttal_ok,
            "entailment_ok": self.entailment_ok,
            "realization_ok": self.realization_ok,
            "passed": self.passed,
            "reasons": self.reasons,
        }


def dialogue_curriculum() -> list[C4DialogueTask]:
    """Build a small audited controlled-dialogue curriculum."""
    return [
        C4DialogueTask(
            task_id="c4-capital-france",
            question="What is the capital of France?",
            atom="capital(france)",
            answer="Paris",
            sources=("audit:geo:france:capital=Paris",),
        ),
        C4DialogueTask(
            task_id="c4-symbol-water",
            question="What is the chemical formula for water?",
            atom="formula(water)",
            answer="H2O",
            sources=("audit:chem:water:formula=H2O",),
        ),
        C4DialogueTask(
            task_id="c4-contradicted-sky",
            question="What color is the daytime clear sky in this toy corpus?",
            atom="color(clear_day_sky)",
            answer="blue",
            sources=("audit:toy:sky:color=blue",),
            rebuttals=("audit:toy:sky:defeater=color=green",),
        ),
    ]


def gate_dialogue(task: C4DialogueTask) -> DialogueGateTrace:
    """Gate a controlled answer under source/rebuttal/realization checks."""
    source_ok = any(source_entails_answer(source, task.answer) for source in task.sources)
    rebuttal_ok = not any(
        source_entails_other(rebuttal, task.answer) for rebuttal in task.rebuttals
    )
    entailment_ok = source_ok and task.answer in task.evidence_text
    rendered_candidate = render_dialogue_answer(task)
    realization_ok = recover_answer(rendered_candidate) == task.answer
    reasons = dialogue_reasons(source_ok, rebuttal_ok, entailment_ok, realization_ok)
    passed = not reasons
    return DialogueGateTrace(
        task_id=task.task_id,
        rendered=rendered_candidate if passed else "refusal",
        source_ok=source_ok,
        rebuttal_ok=rebuttal_ok,
        entailment_ok=entailment_ok,
        realization_ok=realization_ok,
        passed=passed,
        reasons=reasons,
    )


def render_dialogue_answer(task: C4DialogueTask) -> str:
    """Render a controlled answer without adding unbacked atoms."""
    source = task.sources[0] if task.sources else "missing-source"
    return f"{task.answer} [source={source}]"


def recover_answer(rendered: str) -> str:
    """Round-trip the answer span out of the renderer output."""
    return rendered.split(" [source=", maxsplit=1)[0]


def source_entails_answer(source: str, answer: str) -> bool:
    """Return true when an audited source states the exact answer."""
    return source.endswith(f"={answer}")


def source_entails_other(source: str, answer: str) -> bool:
    """Return true when a rebuttal states a different exact answer."""
    return "=" in source and not source.endswith(f"={answer}")


def dialogue_reasons(
    source_ok: bool,
    rebuttal_ok: bool,
    entailment_ok: bool,
    realization_ok: bool,
) -> tuple[str, ...]:
    """Collect failed dialogue-gate reasons."""
    reasons = []
    if not source_ok:
        reasons.append("source witness missing")
    if not rebuttal_ok:
        reasons.append("material rebuttal present")
    if not entailment_ok:
        reasons.append("entailment witness failed")
    if not realization_ok:
        reasons.append("round-trip realization failed")
    return tuple(reasons)


def stage_spec() -> StageSpec:
    """Runtime metadata for the C4 curriculum stage."""
    return StageSpec(
        name="C4",
        summary="controlled dialogue, grounded QA, realization gates",
        executable=True,
        implemented_components=(
            "controlled-source-corpus",
            "entailment-checker",
            "rebuttal-checker",
            "round-trip-checker",
            "dialogue-renderer",
        ),
        blockers=("open-domain retrieval", "calibrated uncertainty"),
    )


__all__ = [
    "C4DialogueTask",
    "DialogueGateTrace",
    "dialogue_curriculum",
    "gate_dialogue",
    "recover_answer",
    "render_dialogue_answer",
    "stage_spec",
]
