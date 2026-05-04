"""Scoped entailment checks and held-out false-support attacks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceClaim:
    """Parsed controlled evidence claim."""

    atom: str
    value: str
    negated: bool = False


@dataclass(frozen=True)
class EntailmentProbe:
    """One held-out entailment or false-support probe."""

    probe_id: str
    atom: str
    answer: str
    evidence: str
    expected_entails: bool


type EntailmentFn = Callable[[str, str, str], bool]


@dataclass(frozen=True)
class EntailmentAttackTrace:
    """Outcome for one held-out entailment attack."""

    probe_id: str
    atom: str
    answer: str
    evidence: str
    expected_entails: bool
    entailed: bool
    naive_answer_support: bool

    @property
    def failed(self) -> bool:
        """True when the checker disagrees with the held-out label."""
        return self.entailed != self.expected_entails

    @property
    def false_support_attack(self) -> bool:
        """True when answer-string support would pass a negative probe."""
        return self.naive_answer_support and not self.expected_entails

    @property
    def false_support_caught(self) -> bool:
        """True when a naive false support pass was rejected."""
        return self.false_support_attack and not self.entailed

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly entailment attack trace."""
        return {
            "probe_id": self.probe_id,
            "atom": self.atom,
            "answer": self.answer,
            "evidence": self.evidence,
            "expected_entails": self.expected_entails,
            "entailed": self.entailed,
            "naive_answer_support": self.naive_answer_support,
            "failed": self.failed,
            "false_support_attack": self.false_support_attack,
            "false_support_caught": self.false_support_caught,
        }


@dataclass(frozen=True)
class EntailmentAttackReport:
    """Held-out entailment false-support attack report."""

    traces: tuple[EntailmentAttackTrace, ...]

    @property
    def failures(self) -> tuple[EntailmentAttackTrace, ...]:
        """Traces where the checker disagreed with the held-out label."""
        return tuple(trace for trace in self.traces if trace.failed)

    @property
    def false_support_attacks(self) -> tuple[EntailmentAttackTrace, ...]:
        """Negative probes that naive answer-string support would accept."""
        return tuple(trace for trace in self.traces if trace.false_support_attack)

    @property
    def caught_false_support(self) -> tuple[EntailmentAttackTrace, ...]:
        """False-support attacks rejected by the entailment checker."""
        return tuple(trace for trace in self.traces if trace.false_support_caught)

    @property
    def passed(self) -> bool:
        """True when labels are correct and every naive false support is caught."""
        return not self.failures and len(self.caught_false_support) == len(
            self.false_support_attacks
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly attack report."""
        return {
            "passed": self.passed,
            "failures": [trace.to_dict() for trace in self.failures],
            "false_support_attacks": [trace.to_dict() for trace in self.false_support_attacks],
            "caught_false_support": [trace.to_dict() for trace in self.caught_false_support],
            "traces": [trace.to_dict() for trace in self.traces],
        }


def entailment_attack_curriculum() -> tuple[EntailmentProbe, ...]:
    """Build held-out entailment probes outside the C4 dialogue corpus."""
    return (
        EntailmentProbe(
            probe_id="heldout-positive-explicit",
            atom="capital(italy)",
            answer="Rome",
            evidence="entails:capital(italy)=Rome",
            expected_entails=True,
        ),
        EntailmentProbe(
            probe_id="mention-not-support",
            atom="capital(france)",
            answer="Paris",
            evidence="mentions: Paris appears; entails:capital(france)=London",
            expected_entails=False,
        ),
        EntailmentProbe(
            probe_id="entity-swap-support",
            atom="capital(france)",
            answer="Paris",
            evidence="entails:capital(texas)=Paris",
            expected_entails=False,
        ),
        EntailmentProbe(
            probe_id="negated-support",
            atom="capital(france)",
            answer="Paris",
            evidence="not:capital(france)=Paris",
            expected_entails=False,
        ),
        EntailmentProbe(
            probe_id="quoted-support",
            atom="formula(water)",
            answer="H2O",
            evidence="quote:formula(water)=H2O",
            expected_entails=False,
        ),
    )


def evaluate_entailment_attacks(
    probes: Iterable[EntailmentProbe] | None = None,
    *,
    entailment_fn: EntailmentFn | None = None,
) -> EntailmentAttackReport:
    """Evaluate held-out false-support probes with an entailment checker."""
    cases = tuple(entailment_attack_curriculum() if probes is None else probes)
    checker = entails_atom if entailment_fn is None else entailment_fn
    return EntailmentAttackReport(
        tuple(
            EntailmentAttackTrace(
                probe_id=probe.probe_id,
                atom=probe.atom,
                answer=probe.answer,
                evidence=probe.evidence,
                expected_entails=probe.expected_entails,
                entailed=checker(probe.atom, probe.answer, probe.evidence),
                naive_answer_support=naive_entailment(probe.atom, probe.answer, probe.evidence),
            )
            for probe in cases
        )
    )


def dirty_entailment_attack_probe() -> EntailmentAttackReport:
    """Run held-out attacks through naive answer-string support."""
    return evaluate_entailment_attacks(entailment_fn=naive_entailment)


def entails_atom(atom: str, answer: str, evidence: str) -> bool:
    """Return true when evidence entails the exact atom/value pair."""
    claim = parse_evidence_claim(evidence)
    if claim is None or claim.negated:
        return False
    return claim.atom == normalize_atom(atom) and claim.value == normalize_answer(answer)


def naive_entailment(_atom: str, answer: str, evidence: str) -> bool:
    """Naive answer-string support used as an adversarial baseline."""
    return normalize_answer(answer) in normalize_free_text(evidence)


def parse_evidence_claim(evidence: str) -> EvidenceClaim | None:
    """Parse explicit or audit-corpus evidence into an atom/value claim."""
    stripped = evidence.strip()
    negated = stripped.startswith(("not:", "deny:"))
    if stripped.startswith(("quote:", "mentions:")):
        return None
    for prefix in ("entails:", "not:", "deny:"):
        if stripped.startswith(prefix):
            stripped = stripped.removeprefix(prefix)
            break
    if "=" not in stripped:
        return None
    left, value = stripped.rsplit("=", maxsplit=1)
    atom = atom_from_left_side(left)
    if atom is None:
        return None
    return EvidenceClaim(atom=atom, value=normalize_answer(value), negated=negated)


def atom_from_left_side(left: str) -> str | None:
    """Extract an atom from explicit ``rel(arg)`` or audit ``...:arg:rel`` syntax."""
    candidate = left.strip()
    if "(" in candidate and candidate.endswith(")"):
        return normalize_atom(candidate)
    parts = candidate.split(":")
    if len(parts) < 3:
        return None
    relation = parts[-1].strip()
    entity = parts[-2].strip()
    if not relation or not entity:
        return None
    return normalize_atom(f"{relation}({entity})")


def normalize_atom(atom: str) -> str:
    """Canonicalize a controlled atom string."""
    return "".join(character.casefold() for character in atom if not character.isspace())


def normalize_answer(value: str) -> str:
    """Canonicalize a controlled answer value."""
    return "".join(character.casefold() for character in value.strip() if character.isalnum())


def normalize_free_text(value: str) -> str:
    """Canonicalize free text for naive answer-string matching."""
    return "".join(character.casefold() for character in value if character.isalnum())


__all__ = [
    "EntailmentAttackReport",
    "EntailmentAttackTrace",
    "EntailmentFn",
    "EntailmentProbe",
    "EvidenceClaim",
    "atom_from_left_side",
    "dirty_entailment_attack_probe",
    "entailment_attack_curriculum",
    "entails_atom",
    "evaluate_entailment_attacks",
    "naive_entailment",
    "normalize_answer",
    "normalize_atom",
    "normalize_free_text",
    "parse_evidence_claim",
]
