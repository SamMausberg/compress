"""Compiler posterior and score-head controls."""

from __future__ import annotations

import pytest

from vpm.compiler import compile_candidate, compiler_posterior, proposal_score
from vpm.tasks import arithmetic_task, concat_task

pytestmark = pytest.mark.sanity


def test_compiler_posterior_selects_expected_matching_operation() -> None:
    posterior = compiler_posterior(arithmetic_task("mul", 3, 4))
    assert posterior.selected is not None
    assert posterior.selected.operation == "mul"
    assert posterior.selected.value_matches is True
    assert 0.0 <= posterior.support_loss < 1.0
    assert sum(alternative.probability for alternative in posterior.alternatives) == pytest.approx(
        1.0
    )


def test_compiler_posterior_filters_type_invalid_operations() -> None:
    posterior = compiler_posterior(concat_task("a", "b"))
    operations = {alternative.operation for alternative in posterior.alternatives}
    assert operations == {"concat", "eq"}
    assert posterior.selected is not None
    assert posterior.selected.operation == "concat"


def test_compile_candidate_carries_candidate_support_loss() -> None:
    compiled = compile_candidate(arithmetic_task("add", 2, 5), "add")
    assert compiled.support_loss == 0.0
    assert compiled.normal_form.ok is True


def test_score_head_rewards_valid_matching_candidates() -> None:
    matching = proposal_score(type_valid=True, value_matches=True)
    invalid = proposal_score(type_valid=False, value_matches=False)
    lossy = proposal_score(type_valid=True, value_matches=True, parser_loss=1.0)
    assert matching > invalid
    assert matching > lossy
