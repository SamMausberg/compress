"""Open-domain ambiguity-collapse evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vpm.language.ambiguity import (
    AmbiguityAction,
    AmbiguityDecision,
    guard_open_domain_prompt,
    naive_open_domain_collapse,
)

type OpenDomainRouter = Callable[[str], AmbiguityDecision]


@dataclass(frozen=True)
class OpenDomainProbe:
    """One open-domain context or semantic ambiguity probe."""

    probe_id: str
    prompt: str
    expected_action: AmbiguityAction


@dataclass(frozen=True)
class OpenDomainTrace:
    """Observed route for one open-domain ambiguity probe."""

    probe_id: str
    prompt: str
    expected_action: AmbiguityAction
    decision: AmbiguityDecision

    @property
    def action_ok(self) -> bool:
        """True when the router selected the expected action."""
        return self.decision.action is self.expected_action

    @property
    def collapsed(self) -> bool:
        """True when unresolved open-domain meaning was certified."""
        return self.expected_action is not AmbiguityAction.CERTIFY and self.decision.certified

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly open-domain trace."""
        return {
            "probe_id": self.probe_id,
            "prompt": self.prompt,
            "expected_action": self.expected_action.value,
            "decision": self.decision.to_dict(),
            "action_ok": self.action_ok,
            "collapsed": self.collapsed,
        }


@dataclass(frozen=True)
class OpenDomainAmbiguityReport:
    """Report for open-domain ambiguity collapse checks."""

    traces: tuple[OpenDomainTrace, ...]

    @property
    def collapses(self) -> tuple[OpenDomainTrace, ...]:
        """Traces that certified unresolved open-domain meaning."""
        return tuple(trace for trace in self.traces if trace.collapsed)

    @property
    def failures(self) -> tuple[OpenDomainTrace, ...]:
        """Traces that did not match the expected route."""
        return tuple(trace for trace in self.traces if not trace.action_ok)

    @property
    def passed(self) -> bool:
        """True when every ambiguous prompt narrows or abstains."""
        return not self.collapses and not self.failures

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly open-domain ambiguity report."""
        return {
            "passed": self.passed,
            "collapses": [trace.to_dict() for trace in self.collapses],
            "failures": [trace.to_dict() for trace in self.failures],
            "traces": [trace.to_dict() for trace in self.traces],
        }


def open_domain_ambiguity_curriculum() -> tuple[OpenDomainProbe, ...]:
    """Build held-out open-domain ambiguity probes."""
    return (
        OpenDomainProbe(
            probe_id="deictic-reference",
            prompt="Is it safe to publish this?",
            expected_action=AmbiguityAction.ASK,
        ),
        OpenDomainProbe(
            probe_id="local-current-context",
            prompt="What is the best bank near me today?",
            expected_action=AmbiguityAction.ASK,
        ),
        OpenDomainProbe(
            probe_id="ambiguous-entity",
            prompt="Tell me about Jordan.",
            expected_action=AmbiguityAction.ASK,
        ),
        OpenDomainProbe(
            probe_id="current-office-context",
            prompt="Who is the current president?",
            expected_action=AmbiguityAction.ASK,
        ),
        OpenDomainProbe(
            probe_id="audited-explicit-atom",
            prompt="audited:capital(france)=Paris",
            expected_action=AmbiguityAction.CERTIFY,
        ),
    )


def evaluate_open_domain_ambiguity(
    probes: Iterable[OpenDomainProbe] | None = None,
    *,
    router: OpenDomainRouter = guard_open_domain_prompt,
) -> OpenDomainAmbiguityReport:
    """Evaluate open-domain context and semantic ambiguity routing."""
    cases = tuple(open_domain_ambiguity_curriculum() if probes is None else probes)
    return OpenDomainAmbiguityReport(
        tuple(
            OpenDomainTrace(
                probe_id=probe.probe_id,
                prompt=probe.prompt,
                expected_action=probe.expected_action,
                decision=router(probe.prompt),
            )
            for probe in cases
        )
    )


def dirty_open_domain_collapse_probe() -> OpenDomainAmbiguityReport:
    """Run ambiguous prompts through a naive certifying router."""
    ambiguous = tuple(
        probe
        for probe in open_domain_ambiguity_curriculum()
        if probe.expected_action is not AmbiguityAction.CERTIFY
    )
    return evaluate_open_domain_ambiguity(
        ambiguous,
        router=naive_open_domain_collapse,
    )


__all__ = [
    "OpenDomainAmbiguityReport",
    "OpenDomainProbe",
    "OpenDomainRouter",
    "OpenDomainTrace",
    "dirty_open_domain_collapse_probe",
    "evaluate_open_domain_ambiguity",
    "open_domain_ambiguity_curriculum",
]
