"""Executable training split and teacher controls."""

from __future__ import annotations

import pytest

from vpm.training import SplitAssignment, TeacherTrace, split_report, teacher_posterior

pytestmark = pytest.mark.sanity


def test_split_policy_rejects_generator_witness_and_audit_leakage() -> None:
    clean = SplitAssignment(
        generator=frozenset({"gen-a"}),
        context=frozenset({"ctx-a"}),
        verifier_train=frozenset({"train-a"}),
        verifier_calibration=frozenset({"cal-a"}),
        audit=frozenset({"audit-a"}),
    )
    assert clean.clean is True

    dirty = SplitAssignment(
        generator=frozenset({"shared"}),
        verifier_train=frozenset({"shared", "train-a"}),
        verifier_calibration=frozenset({"train-a"}),
        audit=frozenset({"shared"}),
    )
    assert dirty.clean is False
    assert any("generator/witness overlap" in violation for violation in dirty.violations())
    assert any("train/calibration overlap" in violation for violation in dirty.violations())
    assert any("audit leakage" in violation for violation in dirty.violations())
    assert split_report((clean, dirty))["clean"] is False


def test_teacher_posterior_truncates_to_certified_split_clean_replay() -> None:
    posterior = teacher_posterior(
        (
            TeacherTrace(
                "dirty", "shortcut", certificate=10.0, utility=10.0, cost=0.1, split_clean=False
            ),
            TeacherTrace(
                "failed", "bad", certificate=10.0, utility=10.0, cost=0.1, replay_passed=False
            ),
            TeacherTrace("weak", "safe-weak", certificate=1.0, utility=0.2, cost=0.1),
            TeacherTrace("strong", "safe-strong", certificate=2.0, utility=0.2, cost=0.1),
        ),
        top_k=1,
    )
    assert posterior.support == ("safe-strong",)
    assert posterior.entries[0].probability == 1.0
    assert {trace.candidate for trace in posterior.rejected} == {"shortcut", "bad"}
