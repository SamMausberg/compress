"""Held-out hard-domain evaluation checks."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from vpm.evaluation.hard_domains import evaluate_hard_domains
from vpm.tasks import HardDomain, hard_domain_curriculum

pytestmark = pytest.mark.sanity


def test_hard_domain_curriculum_covers_named_goal_domains() -> None:
    tasks = hard_domain_curriculum()
    assert {task.domain for task in tasks} == set(HardDomain)
    assert all(task.evidence for task in tasks)


def test_hard_domain_evaluation_certifies_all_shipped_probes() -> None:
    report = evaluate_hard_domains()
    assert report.solve_rate == 1.0
    assert report.baseline_solve_rate == 0.0
    assert [domain.tasks for domain in report.domains] == [1, 1, 1, 1]
    assert all(trace.certified for trace in report.traces)


def test_cli_runs_hard_domain_suite() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "eval-hard-domains", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["solve_rate"] == 1.0
    assert {domain["domain"] for domain in payload["domains"]} == {
        "research_math",
        "formal",
        "tool_use",
        "source_grounded",
    }
