"""External cognitive-component authority regressions."""

from __future__ import annotations

import pytest

from vpm.evaluation.external_components import (
    CognitiveComponent,
    ComponentFamily,
    ComponentRole,
    dirty_external_component_probe,
    evaluate_external_components,
)

pytestmark = pytest.mark.sanity


def test_external_components_allow_only_logged_non_authoritative_llm_roles() -> None:
    report = evaluate_external_components(
        (
            CognitiveComponent(
                name="logged-llm-teacher",
                family=ComponentFamily.EXTERNAL_LLM,
                role=ComponentRole.TEACHER,
                budget_logged=True,
                certificate_authority=False,
            ),
            CognitiveComponent(
                name="logged-llm-baseline",
                family=ComponentFamily.EXTERNAL_LLM,
                role=ComponentRole.BASELINE,
                budget_logged=True,
                certificate_authority=False,
            ),
        )
    )
    assert report.passed is True
    assert report.violations == ()
    assert report.external_inference_dependencies == ()


def test_dirty_external_component_probe_is_rejected() -> None:
    report = dirty_external_component_probe()
    assert report.passed is False
    assert len(report.violations) == 1
    assert len(report.external_inference_dependencies) == 1
    assert report.violations[0].name == "unlogged-llm-inference"
