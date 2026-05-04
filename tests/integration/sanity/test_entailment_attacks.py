"""Held-out entailment false-support attack regressions."""

from __future__ import annotations

import pytest

from vpm.tasks.c4 import dialogue_curriculum, gate_dialogue
from vpm.verifiers.entailment import (
    dirty_entailment_attack_probe,
    entails_atom,
    evaluate_entailment_attacks,
)

pytestmark = pytest.mark.sanity


def test_entailment_checker_rejects_false_support_attacks() -> None:
    report = evaluate_entailment_attacks()
    assert report.passed is True
    assert report.failures == ()
    assert len(report.false_support_attacks) == 4
    assert len(report.caught_false_support) == 4
    assert entails_atom("capital(italy)", "Rome", "entails:capital(italy)=Rome")
    assert not entails_atom("capital(france)", "Paris", "entails:capital(texas)=Paris")


def test_dirty_entailment_attack_probe_is_rejected() -> None:
    report = dirty_entailment_attack_probe()
    assert report.passed is False
    assert len(report.failures) == 4
    assert len(report.caught_false_support) == 0


def test_c4_dialogue_uses_scoped_entailment_witness() -> None:
    trace = gate_dialogue(dialogue_curriculum()[0])
    assert trace.passed is True
    assert trace.entailment_ok is True
