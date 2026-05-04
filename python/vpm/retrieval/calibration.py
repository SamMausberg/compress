"""Controlled source/rebuttal recall calibration under corpus shift."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

type RecallChannel = Literal["source", "rebuttal"]


@dataclass(frozen=True)
class RecallShiftProbe:
    """One controlled source or rebuttal recall probe."""

    probe_id: str
    channel: RecallChannel
    atom: str
    answer: str
    evidence: tuple[str, ...]
    should_recall: bool
    shifted: bool = False


type RecallFn = Callable[[RecallShiftProbe], bool]


@dataclass(frozen=True)
class RecallShiftTrace:
    """Observed recall outcome for one source/rebuttal shift probe."""

    probe_id: str
    channel: RecallChannel
    atom: str
    answer: str
    shifted: bool
    expected: bool
    recalled: bool
    evidence: tuple[str, ...]

    @property
    def miss(self) -> bool:
        """True when an expected source/rebuttal witness was not recalled."""
        return self.expected and not self.recalled

    @property
    def false_recall(self) -> bool:
        """True when the retriever recalled evidence that should not support the channel."""
        return self.recalled and not self.expected

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly recall trace."""
        return {
            "probe_id": self.probe_id,
            "channel": self.channel,
            "atom": self.atom,
            "answer": self.answer,
            "shifted": self.shifted,
            "expected": self.expected,
            "recalled": self.recalled,
            "miss": self.miss,
            "false_recall": self.false_recall,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class RecallShiftReport:
    """Controlled source/rebuttal recall calibration report."""

    traces: tuple[RecallShiftTrace, ...]
    source_epsilon: float
    rebuttal_epsilon: float
    shifted_epsilon: float
    false_recall_rate: float
    threshold: float = 0.0

    @property
    def misses(self) -> tuple[RecallShiftTrace, ...]:
        """Recall-miss traces."""
        return tuple(trace for trace in self.traces if trace.miss)

    @property
    def false_recalls(self) -> tuple[RecallShiftTrace, ...]:
        """False-recall traces."""
        return tuple(trace for trace in self.traces if trace.false_recall)

    @property
    def passed(self) -> bool:
        """True when the controlled shift calibration has no excess loss."""
        return (
            self.source_epsilon <= self.threshold
            and self.rebuttal_epsilon <= self.threshold
            and self.shifted_epsilon <= self.threshold
            and self.false_recall_rate <= self.threshold
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly recall-shift report."""
        return {
            "passed": self.passed,
            "source_epsilon": self.source_epsilon,
            "rebuttal_epsilon": self.rebuttal_epsilon,
            "shifted_epsilon": self.shifted_epsilon,
            "false_recall_rate": self.false_recall_rate,
            "threshold": self.threshold,
            "misses": [trace.to_dict() for trace in self.misses],
            "false_recalls": [trace.to_dict() for trace in self.false_recalls],
            "traces": [trace.to_dict() for trace in self.traces],
        }


def recall_shift_curriculum() -> tuple[RecallShiftProbe, ...]:
    """Build controlled in-domain and shifted source/rebuttal recall probes."""
    return (
        RecallShiftProbe(
            probe_id="source-exact-capital",
            channel="source",
            atom="capital(france)",
            answer="Paris",
            evidence=("audit:geo:france:capital=Paris",),
            should_recall=True,
        ),
        RecallShiftProbe(
            probe_id="source-shifted-whitespace",
            channel="source",
            atom="formula(water)",
            answer="H2O",
            evidence=("audit:chem:water:formula = H2O",),
            should_recall=True,
            shifted=True,
        ),
        RecallShiftProbe(
            probe_id="source-shifted-colon",
            channel="source",
            atom="sum(two,two)",
            answer="4",
            evidence=("audit:math:two-plus-two:sum: 4",),
            should_recall=True,
            shifted=True,
        ),
        RecallShiftProbe(
            probe_id="source-wrong-answer",
            channel="source",
            atom="capital(france)",
            answer="Paris",
            evidence=("audit:geo:france:capital=London",),
            should_recall=False,
        ),
        RecallShiftProbe(
            probe_id="rebuttal-exact-defeater",
            channel="rebuttal",
            atom="color(clear_day_sky)",
            answer="blue",
            evidence=("audit:toy:sky:defeater=color=green",),
            should_recall=True,
        ),
        RecallShiftProbe(
            probe_id="rebuttal-shifted-stale",
            channel="rebuttal",
            atom="formula(water)",
            answer="H2O",
            evidence=("audit:chem:water:stale formula: HO",),
            should_recall=True,
            shifted=True,
        ),
        RecallShiftProbe(
            probe_id="rebuttal-same-answer-clear",
            channel="rebuttal",
            atom="capital(france)",
            answer="Paris",
            evidence=("audit:geo:france:capital=Paris",),
            should_recall=False,
        ),
    )


def evaluate_recall_shift(
    probes: Iterable[RecallShiftProbe] | None = None,
    *,
    recall_fn: RecallFn | None = None,
    threshold: float = 0.0,
) -> RecallShiftReport:
    """Evaluate source/rebuttal recall against controlled shifted evidence."""
    cases = tuple(recall_shift_curriculum() if probes is None else probes)
    recall = calibrated_recall if recall_fn is None else recall_fn
    traces = tuple(
        RecallShiftTrace(
            probe_id=probe.probe_id,
            channel=probe.channel,
            atom=probe.atom,
            answer=probe.answer,
            shifted=probe.shifted,
            expected=probe.should_recall,
            recalled=recall(probe),
            evidence=probe.evidence,
        )
        for probe in cases
    )
    return RecallShiftReport(
        traces=traces,
        source_epsilon=miss_rate(trace for trace in traces if trace.channel == "source"),
        rebuttal_epsilon=miss_rate(trace for trace in traces if trace.channel == "rebuttal"),
        shifted_epsilon=miss_rate(trace for trace in traces if trace.shifted),
        false_recall_rate=false_recall_rate(traces),
        threshold=threshold,
    )


def dirty_recall_shift_probe() -> RecallShiftReport:
    """Run shifted positives through the legacy exact-suffix recall rule."""
    shifted_positives = (
        probe for probe in recall_shift_curriculum() if probe.shifted and probe.should_recall
    )
    return evaluate_recall_shift(
        shifted_positives,
        recall_fn=exact_suffix_recall,
        threshold=0.0,
    )


def calibrated_recall(probe: RecallShiftProbe) -> bool:
    """Recall controlled source/rebuttal evidence after normal-form shifts."""
    return any(evidence_recalls(probe.channel, item, probe.answer) for item in probe.evidence)


def exact_suffix_recall(probe: RecallShiftProbe) -> bool:
    """Legacy exact-suffix rule used as an adversarial calibration bypass."""
    if probe.channel == "source":
        return any(item.endswith(f"={probe.answer}") for item in probe.evidence)
    return any("=" in item and not item.endswith(f"={probe.answer}") for item in probe.evidence)


def evidence_recalls(channel: RecallChannel, evidence: str, answer: str) -> bool:
    """Return whether one controlled evidence string recalls the requested channel."""
    value = normalized_claim_value(evidence)
    if value is None:
        return False
    expected = normalize_answer(answer)
    if channel == "source":
        return value == expected
    return value not in ("", expected)


def normalized_claim_value(evidence: str) -> str | None:
    """Extract a normalized claim value from controlled audit-corpus syntax."""
    text = " ".join(evidence.strip().split())
    for separator in ("=>", "->", "=", ":"):
        if separator in text:
            return normalize_answer(text.rsplit(separator, maxsplit=1)[1])
    return None


def normalize_answer(value: str) -> str:
    """Canonicalize controlled answer spans for exact-audit comparisons."""
    trimmed = value.strip()
    for delimiter in (" [", " (", ";", ","):
        trimmed = trimmed.split(delimiter, maxsplit=1)[0]
    return "".join(character.casefold() for character in trimmed if character.isalnum())


def miss_rate(traces: Iterable[RecallShiftTrace]) -> float:
    """Return expected-positive recall miss rate."""
    expected = tuple(trace for trace in traces if trace.expected)
    if not expected:
        return 0.0
    return sum(trace.miss for trace in expected) / len(expected)


def false_recall_rate(traces: Iterable[RecallShiftTrace]) -> float:
    """Return false-recall rate over expected negatives."""
    negatives = tuple(trace for trace in traces if not trace.expected)
    if not negatives:
        return 0.0
    return sum(trace.false_recall for trace in negatives) / len(negatives)


__all__ = [
    "RecallChannel",
    "RecallFn",
    "RecallShiftProbe",
    "RecallShiftReport",
    "RecallShiftTrace",
    "calibrated_recall",
    "dirty_recall_shift_probe",
    "evaluate_recall_shift",
    "evidence_recalls",
    "exact_suffix_recall",
    "false_recall_rate",
    "miss_rate",
    "normalize_answer",
    "normalized_claim_value",
    "recall_shift_curriculum",
]
