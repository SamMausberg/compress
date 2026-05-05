"""Open-domain context and semantic ambiguity guards."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AmbiguityAction(StrEnum):
    """Allowed routing actions for open-domain language prompts."""

    CERTIFY = "certify"
    ASK = "ask"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class AmbiguityDecision:
    """Routing decision for an open-domain prompt."""

    prompt: str
    action: AmbiguityAction
    certified_atoms: tuple[str, ...]
    context_loss: float
    semantic_loss: float
    reasons: tuple[str, ...]

    @property
    def certified(self) -> bool:
        """True when the prompt produced certified semantic atoms."""
        return self.action is AmbiguityAction.CERTIFY and bool(self.certified_atoms)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly ambiguity decision."""
        return {
            "prompt": self.prompt,
            "action": self.action.value,
            "certified_atoms": self.certified_atoms,
            "context_loss": self.context_loss,
            "semantic_loss": self.semantic_loss,
            "reasons": self.reasons,
            "certified": self.certified,
        }


def guard_open_domain_prompt(prompt: str) -> AmbiguityDecision:
    """Route open-domain prompts without collapsing unresolved meaning."""
    reasons = ambiguity_reasons(prompt)
    if reasons:
        action = (
            AmbiguityAction.ASK
            if any("unresolved" in reason for reason in reasons)
            else AmbiguityAction.ABSTAIN
        )
        return AmbiguityDecision(
            prompt=prompt,
            action=action,
            certified_atoms=(),
            context_loss=1.0
            if any("context" in reason or "reference" in reason for reason in reasons)
            else 0.0,
            semantic_loss=1.0
            if any("semantic" in reason or "vague" in reason for reason in reasons)
            else 0.0,
            reasons=reasons,
        )
    atom = audited_atom(prompt)
    if atom is not None:
        return AmbiguityDecision(
            prompt=prompt,
            action=AmbiguityAction.CERTIFY,
            certified_atoms=(atom,),
            context_loss=0.0,
            semantic_loss=0.0,
            reasons=(),
        )
    return AmbiguityDecision(
        prompt=prompt,
        action=AmbiguityAction.ABSTAIN,
        certified_atoms=(),
        context_loss=1.0,
        semantic_loss=1.0,
        reasons=("open-domain source unavailable",),
    )


def naive_open_domain_collapse(prompt: str) -> AmbiguityDecision:
    """Naive adversarial router that certifies the first plausible atom."""
    atom = next(iter(candidate_atoms(prompt)), "open_domain(answer)")
    return AmbiguityDecision(
        prompt=prompt,
        action=AmbiguityAction.CERTIFY,
        certified_atoms=(atom,),
        context_loss=0.0,
        semantic_loss=0.0,
        reasons=(),
    )


def ambiguity_reasons(prompt: str) -> tuple[str, ...]:
    """Detect unresolved context and semantic ambiguity in open-domain prompts."""
    text = normalized_text(prompt)
    reasons: list[str] = []
    if any(token in text for token in (" this ", " that ", " it ", " they ", " he ", " she ")):
        reasons.append("reference unresolved")
    if any(token in text for token in (" near me", " today", " current", " latest", " president")):
        reasons.append("context unresolved")
    if any(token in text for token in (" best ", " safest ", " safe ", " good ", " recent ")):
        reasons.append("vague predicate unresolved")
    if len(candidate_atoms(prompt)) > 1:
        reasons.append("semantic ambiguity unresolved")
    return tuple(reasons)


def candidate_atoms(prompt: str) -> tuple[str, ...]:
    """Return plausible semantic atoms for known ambiguous open-domain names."""
    text = normalized_text(prompt)
    candidates: list[str] = []
    if " jordan" in text:
        candidates.extend(("person(jordan)", "country(jordan)"))
    if " apple" in text:
        candidates.extend(("company(apple)", "food(apple)"))
    if " mercury" in text:
        candidates.extend(("planet(mercury)", "element(mercury)"))
    if " java" in text:
        candidates.extend(("language(java)", "island(java)", "coffee(java)"))
    return tuple(candidates)


def audited_atom(prompt: str) -> str | None:
    """Parse explicit audited atom prompts that are allowed to certify."""
    stripped = prompt.strip()
    if not stripped.startswith("audited:") or "=" not in stripped:
        return None
    left, _value = stripped.removeprefix("audited:").rsplit("=", maxsplit=1)
    if "(" not in left or not left.endswith(")"):
        return None
    return "".join(character.casefold() for character in left if not character.isspace())


def normalized_text(prompt: str) -> str:
    """Normalize prompt text for conservative lexical guards."""
    return f" {prompt.casefold().strip()} "


__all__ = [
    "AmbiguityAction",
    "AmbiguityDecision",
    "ambiguity_reasons",
    "audited_atom",
    "candidate_atoms",
    "guard_open_domain_prompt",
    "naive_open_domain_collapse",
    "normalized_text",
]
