"""Executable sanity checks for the MVP C0 kernel."""

from __future__ import annotations

import json

import pytest

from vpm import _native
from vpm.infer import run_c0_add


pytestmark = pytest.mark.sanity


def test_domain_route_solves_cheap_verifier_task() -> None:
    result = run_c0_add(8, 13)
    assert result.route == "solve"
    assert result.native_report["gate"]["passed"] is True


def test_failed_verifier_does_not_render_certified_answer() -> None:
    result = run_c0_add(8, 13, expected=22)
    assert result.rendered == "refusal"
    assert result.native_report["verification"]["passed"] is False


def test_componentwise_risk_policy_is_visible_in_contract() -> None:
    contract = json.loads(_native.c0_contract_json())
    risk_budget = contract["risk_budget"]
    assert risk_budget["privacy"] == 0.0
    assert "capability" not in contract["allowed_authorities"]
