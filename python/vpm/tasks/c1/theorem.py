"""Propositional theorem-fragment controls for C1 tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TheoremFragmentTask:
    """Small propositional theorem-proving fragment."""

    task_id: str
    premises: tuple[str, ...]
    goal: str
    expected_steps: int

    @property
    def observation(self) -> str:
        """Compact theorem fragment observation."""
        return f"prove {self.goal} from {','.join(self.premises)}"

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly theorem task."""
        return {
            "task_id": self.task_id,
            "premises": self.premises,
            "goal": self.goal,
            "expected_steps": self.expected_steps,
        }


@dataclass(frozen=True)
class ProofStep:
    """One checked propositional proof step."""

    rule: str
    antecedent: str
    implication: str
    conclusion: str

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly proof step."""
        return {
            "rule": self.rule,
            "antecedent": self.antecedent,
            "implication": self.implication,
            "conclusion": self.conclusion,
        }


@dataclass(frozen=True)
class TheoremProofTrace:
    """Proof trace for one theorem fragment."""

    task_id: str
    goal: str
    derived: tuple[str, ...]
    proof_steps: tuple[ProofStep, ...]
    expected_steps: int

    @property
    def proven(self) -> bool:
        """True when the goal was derived."""
        return self.goal in self.derived

    @property
    def passed(self) -> bool:
        """True when the proof derives the goal in the expected step budget."""
        return self.proven and len(self.proof_steps) == self.expected_steps

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly theorem proof trace."""
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "derived": self.derived,
            "proof_steps": [step.to_dict() for step in self.proof_steps],
            "expected_steps": self.expected_steps,
            "proven": self.proven,
            "passed": self.passed,
        }


def theorem_fragment_curriculum() -> tuple[TheoremFragmentTask, ...]:
    """Build small propositional theorem-proving fragments."""
    return (
        TheoremFragmentTask(
            task_id="c1-proof-modus-ponens",
            premises=("p", "p->q"),
            goal="q",
            expected_steps=1,
        ),
        TheoremFragmentTask(
            task_id="c1-proof-two-hop-chain",
            premises=("rain", "rain->wet", "wet->slippery"),
            goal="slippery",
            expected_steps=2,
        ),
    )


def prove_theorem_fragment(task: TheoremFragmentTask) -> TheoremProofTrace:
    """Forward-chain a propositional theorem fragment under modus ponens."""
    known = {premise for premise in task.premises if parse_implication(premise) is None}
    implications = tuple(
        (premise, parsed)
        for premise in task.premises
        if (parsed := parse_implication(premise)) is not None
    )
    steps: list[ProofStep] = []
    changed = True
    while changed and task.goal not in known:
        changed = False
        for implication, (antecedent, conclusion) in implications:
            if antecedent in known and conclusion not in known:
                known.add(conclusion)
                steps.append(
                    ProofStep(
                        rule="modus_ponens",
                        antecedent=antecedent,
                        implication=implication,
                        conclusion=conclusion,
                    )
                )
                changed = True
                if conclusion == task.goal:
                    break
    return TheoremProofTrace(
        task_id=task.task_id,
        goal=task.goal,
        derived=tuple(sorted(known)),
        proof_steps=tuple(steps),
        expected_steps=task.expected_steps,
    )


def parse_implication(claim: str) -> tuple[str, str] | None:
    """Parse a compact propositional implication."""
    if "->" not in claim:
        return None
    left, right = claim.split("->", maxsplit=1)
    antecedent = left.strip()
    conclusion = right.strip()
    if not antecedent or not conclusion:
        return None
    return antecedent, conclusion


__all__ = [
    "ProofStep",
    "TheoremFragmentTask",
    "TheoremProofTrace",
    "parse_implication",
    "prove_theorem_fragment",
    "theorem_fragment_curriculum",
]
