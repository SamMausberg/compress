"""Dependence-block residualization checks under calibration shift."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class DependenceSignal:
    """One verifier pass with fields that induce a dependence block."""

    signal_id: str
    claim_id: str
    verifier_family: str
    generator_id: str
    data_split: str
    retrieval_index: str
    training_run: str
    calibration_source: str
    prompt_id: str
    tool_id: str
    failure_mode: str
    passed: bool = True
    false_pass_upper: float = 0.01
    residualized: bool = False
    residual_shift: float = 0.0

    @property
    def block_key(self) -> tuple[str, ...]:
        """Fields that make verifier passes dependent for one claim."""
        return (
            self.claim_id,
            self.verifier_family,
            self.generator_id,
            self.data_split,
            self.retrieval_index,
            self.training_run,
            self.calibration_source,
            self.prompt_id,
            self.tool_id,
            self.failure_mode,
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly dependence signal."""
        return {
            "signal_id": self.signal_id,
            "claim_id": self.claim_id,
            "verifier_family": self.verifier_family,
            "generator_id": self.generator_id,
            "data_split": self.data_split,
            "retrieval_index": self.retrieval_index,
            "training_run": self.training_run,
            "calibration_source": self.calibration_source,
            "prompt_id": self.prompt_id,
            "tool_id": self.tool_id,
            "failure_mode": self.failure_mode,
            "passed": self.passed,
            "false_pass_upper": self.false_pass_upper,
            "residualized": self.residualized,
            "residual_shift": self.residual_shift,
        }


@dataclass(frozen=True)
class DependenceBlockTrace:
    """Residualization decision for one dependence block."""

    block_key: tuple[str, ...]
    signal_ids: tuple[str, ...]
    contributing_signal_ids: tuple[str, ...]
    blocked_signal_ids: tuple[str, ...]
    opportunities: int
    leaks: int
    shifted_opportunities: int
    shifted_leaks: int
    residualized: bool
    residual_shift: float
    e_value: float

    @property
    def leaked_independence(self) -> bool:
        """True when dependent passes were treated as independent."""
        return self.leaks > 0

    @property
    def shifted(self) -> bool:
        """True when the block contains residual calibration shift."""
        return self.residual_shift > 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly dependence-block trace."""
        return {
            "block_key": self.block_key,
            "signal_ids": self.signal_ids,
            "contributing_signal_ids": self.contributing_signal_ids,
            "blocked_signal_ids": self.blocked_signal_ids,
            "opportunities": self.opportunities,
            "leaks": self.leaks,
            "shifted_opportunities": self.shifted_opportunities,
            "shifted_leaks": self.shifted_leaks,
            "residualized": self.residualized,
            "residual_shift": self.residual_shift,
            "e_value": self.e_value,
            "leaked_independence": self.leaked_independence,
            "shifted": self.shifted,
        }


@dataclass(frozen=True)
class DependenceCalibrationReport:
    """Report for dependence residualization under shift."""

    traces: tuple[DependenceBlockTrace, ...]
    epsilon_dep: float
    shifted_epsilon: float
    shift_threshold: float

    @property
    def leaks(self) -> tuple[DependenceBlockTrace, ...]:
        """Blocks that leaked dependent passes as independent."""
        return tuple(trace for trace in self.traces if trace.leaked_independence)

    @property
    def passed(self) -> bool:
        """True when no dependence leak exceeds the calibration threshold."""
        return self.epsilon_dep <= 0.0 and self.shifted_epsilon <= 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly dependence calibration report."""
        return {
            "passed": self.passed,
            "epsilon_dep": self.epsilon_dep,
            "shifted_epsilon": self.shifted_epsilon,
            "shift_threshold": self.shift_threshold,
            "leaks": [trace.to_dict() for trace in self.leaks],
            "traces": [trace.to_dict() for trace in self.traces],
        }


def dependence_shift_curriculum() -> tuple[DependenceSignal, ...]:
    """Build controlled verifier dependence blocks with shifted residual probes."""
    return (
        DependenceSignal(
            signal_id="arith-independent-a",
            claim_id="claim-add",
            verifier_family="exact-arithmetic",
            generator_id="gen-a",
            data_split="audit-a",
            retrieval_index="none",
            training_run="run-a",
            calibration_source="cal-a",
            prompt_id="none",
            tool_id="python-exec",
            failure_mode="arithmetic",
        ),
        DependenceSignal(
            signal_id="dup-unresid-a",
            claim_id="claim-dup",
            verifier_family="exact-arithmetic",
            generator_id="gen-dup",
            data_split="audit-dup",
            retrieval_index="none",
            training_run="run-dup",
            calibration_source="cal-dup",
            prompt_id="none",
            tool_id="python-exec",
            failure_mode="arithmetic",
            false_pass_upper=0.02,
        ),
        DependenceSignal(
            signal_id="dup-unresid-b",
            claim_id="claim-dup",
            verifier_family="exact-arithmetic",
            generator_id="gen-dup",
            data_split="audit-dup",
            retrieval_index="none",
            training_run="run-dup",
            calibration_source="cal-dup",
            prompt_id="none",
            tool_id="python-exec",
            failure_mode="arithmetic",
            false_pass_upper=0.01,
        ),
        DependenceSignal(
            signal_id="resid-clean-a",
            claim_id="claim-clean-resid",
            verifier_family="source-checker",
            generator_id="gen-clean",
            data_split="audit-clean",
            retrieval_index="idx-clean",
            training_run="run-clean",
            calibration_source="cal-clean",
            prompt_id="prompt-clean",
            tool_id="retriever",
            failure_mode="source-recall",
            residualized=True,
            residual_shift=0.0,
            false_pass_upper=0.05,
        ),
        DependenceSignal(
            signal_id="resid-clean-b",
            claim_id="claim-clean-resid",
            verifier_family="source-checker",
            generator_id="gen-clean",
            data_split="audit-clean",
            retrieval_index="idx-clean",
            training_run="run-clean",
            calibration_source="cal-clean",
            prompt_id="prompt-clean",
            tool_id="retriever",
            failure_mode="source-recall",
            residualized=True,
            residual_shift=0.0,
            false_pass_upper=0.04,
        ),
        DependenceSignal(
            signal_id="resid-shifted-a",
            claim_id="claim-shifted-resid",
            verifier_family="source-checker",
            generator_id="gen-shift",
            data_split="audit-shift",
            retrieval_index="idx-shift",
            training_run="run-shift",
            calibration_source="cal-shift",
            prompt_id="prompt-shift",
            tool_id="retriever",
            failure_mode="source-recall",
            residualized=True,
            residual_shift=0.35,
            false_pass_upper=0.05,
        ),
        DependenceSignal(
            signal_id="resid-shifted-b",
            claim_id="claim-shifted-resid",
            verifier_family="source-checker",
            generator_id="gen-shift",
            data_split="audit-shift",
            retrieval_index="idx-shift",
            training_run="run-shift",
            calibration_source="cal-shift",
            prompt_id="prompt-shift",
            tool_id="retriever",
            failure_mode="source-recall",
            residualized=True,
            residual_shift=0.35,
            false_pass_upper=0.04,
        ),
    )


def evaluate_dependence_shift(
    signals: Iterable[DependenceSignal] | None = None,
    *,
    shift_threshold: float = 0.1,
    treat_uncalibrated_as_independent: bool = False,
) -> DependenceCalibrationReport:
    """Evaluate dependence-block residualization and shifted calibration gates."""
    cases = tuple(dependence_shift_curriculum() if signals is None else signals)
    traces = tuple(
        block_trace(
            block,
            shift_threshold=shift_threshold,
            treat_uncalibrated_as_independent=treat_uncalibrated_as_independent,
        )
        for block in group_blocks(cases)
    )
    opportunities = sum(trace.opportunities for trace in traces)
    shifted_opportunities = sum(trace.shifted_opportunities for trace in traces)
    leaks = sum(trace.leaks for trace in traces)
    shifted_leaks = sum(trace.shifted_leaks for trace in traces)
    return DependenceCalibrationReport(
        traces=traces,
        epsilon_dep=leaks / opportunities if opportunities else 0.0,
        shifted_epsilon=shifted_leaks / shifted_opportunities if shifted_opportunities else 0.0,
        shift_threshold=shift_threshold,
    )


def dirty_dependence_shift_probe() -> DependenceCalibrationReport:
    """Treat shifted residual duplicates as independent to verify calibration catches it."""
    shifted = tuple(
        signal
        for signal in dependence_shift_curriculum()
        if signal.claim_id == "claim-shifted-resid"
    )
    return evaluate_dependence_shift(
        shifted,
        shift_threshold=0.1,
        treat_uncalibrated_as_independent=True,
    )


def group_blocks(signals: Iterable[DependenceSignal]) -> tuple[tuple[DependenceSignal, ...], ...]:
    """Group signals by dependence-block key."""
    blocks: dict[tuple[str, ...], list[DependenceSignal]] = {}
    for signal in signals:
        blocks.setdefault(signal.block_key, []).append(signal)
    return tuple(tuple(block) for block in blocks.values())


def block_trace(
    block: tuple[DependenceSignal, ...],
    *,
    shift_threshold: float,
    treat_uncalibrated_as_independent: bool,
) -> DependenceBlockTrace:
    """Build one dependence-block trace."""
    ordered = strongest_first(block)
    passed = tuple(signal for signal in ordered if signal.passed)
    opportunities = max(len(passed) - 1, 0)
    residual_shift = max((signal.residual_shift for signal in block), default=0.0)
    residualized = bool(passed) and all(signal.residualized for signal in passed)
    eligible = residualized and residual_shift <= shift_threshold
    treated_independent = eligible or (treat_uncalibrated_as_independent and opportunities > 0)
    contributing = passed if treated_independent else passed[:1]
    blocked = () if treated_independent else passed[1:]
    leaked = opportunities if treat_uncalibrated_as_independent and not eligible else 0
    shifted_opportunities = opportunities if residual_shift > shift_threshold else 0
    shifted_leaks = leaked if shifted_opportunities else 0
    return DependenceBlockTrace(
        block_key=block[0].block_key if block else (),
        signal_ids=tuple(signal.signal_id for signal in ordered),
        contributing_signal_ids=tuple(signal.signal_id for signal in contributing),
        blocked_signal_ids=tuple(signal.signal_id for signal in blocked),
        opportunities=opportunities,
        leaks=leaked,
        shifted_opportunities=shifted_opportunities,
        shifted_leaks=shifted_leaks,
        residualized=eligible,
        residual_shift=residual_shift,
        e_value=block_e_value(contributing),
    )


def strongest_first(signals: Iterable[DependenceSignal]) -> tuple[DependenceSignal, ...]:
    """Order verifier passes by strongest calibrated false-pass bound."""
    return tuple(sorted(signals, key=lambda signal: (signal.false_pass_upper, signal.signal_id)))


def block_e_value(signals: Iterable[DependenceSignal]) -> float:
    """Return the dependence-block e-value from contributing verifier passes."""
    value = 1.0
    for signal in signals:
        if signal.passed:
            value *= 1.0 / clamp_probability(signal.false_pass_upper)
    return value


def clamp_probability(value: float) -> float:
    """Clamp a false-pass upper bound to the valid e-value range."""
    return min(max(value, 1.0e-12), 1.0)


__all__ = [
    "DependenceBlockTrace",
    "DependenceCalibrationReport",
    "DependenceSignal",
    "block_e_value",
    "block_trace",
    "clamp_probability",
    "dependence_shift_curriculum",
    "dirty_dependence_shift_probe",
    "evaluate_dependence_shift",
    "group_blocks",
    "strongest_first",
]
