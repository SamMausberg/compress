"""Evaluation CLI command registration."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import json

import typer

from vpm.evaluation import (
    evaluate_c0,
    evaluate_c1,
    evaluate_c2,
    evaluate_c3,
    evaluate_c4,
    evaluate_c5,
)
from vpm.evaluation.ablations import evaluate_ablations
from vpm.evaluation.baselines import evaluate_baseline_suite
from vpm.evaluation.compute_accounting import evaluate_compute_accounting
from vpm.evaluation.external_components import evaluate_external_components
from vpm.evaluation.failure_modes import evaluate_failure_modes
from vpm.evaluation.hard_domains import evaluate_hard_domains
from vpm.evaluation.phase_transition import evaluate_phase_transition
from vpm.evaluation.red_team import red_team_replay
from vpm.evaluation.saturation import evaluate_saturation
from vpm.retrieval.calibration import evaluate_recall_shift
from vpm.verifiers.dependence import evaluate_dependence_shift
from vpm.verifiers.entailment import evaluate_entailment_attacks


def register_eval_commands(app: typer.Typer) -> None:
    """Register evaluation commands on the top-level Typer app."""
    register_curriculum_eval_commands(app)
    register_meta_eval_commands(app)
    register_audit_eval_commands(app)


def register_curriculum_eval_commands(app: typer.Typer) -> None:
    """Register C0-C5 evaluation commands."""

    @app.command("eval-c0")
    def eval_c0_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run the bundled C0 curriculum and summarize MVP metrics."""
        report = evaluate_c0()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(f"solve_rate={report.solve_rate:.3f} tasks={report.tasks}")

    @app.command("eval-c1")
    def eval_c1_command(
        limit: int = typer.Option(3, help="Absolute integer limit used for C1 generation."),
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run the executable C1 hidden-schema subset and summarize metrics."""
        report = evaluate_c1(limit=limit)
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(f"solve_rate={report.solve_rate:.3f} tasks={report.tasks}")

    @app.command("eval-c2")
    def eval_c2_command(
        limit: int = typer.Option(3, help="Absolute integer limit used for C2 generation."),
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run C2 active-test selection and verifier-gated execution."""
        report = evaluate_c2(limit=limit)
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"solve_rate={report.solve_rate:.3f} "
                f"support_reduction={report.support_reduction_rate:.3f} "
                f"tasks={report.tasks}"
            )

    @app.command("eval-c3")
    def eval_c3_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run C3 adversarial authority/risk policy probes."""
        report = evaluate_c3()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"violation_rate={report.violation_rate:.3f} "
                f"rejected={report.rejected} "
                f"probes={report.probes}"
            )

    @app.command("eval-c4")
    def eval_c4_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run C4 controlled dialogue source/rebuttal/realization gates."""
        report = evaluate_c4()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"render_rate={report.render_rate:.3f} "
                f"refusal_rate={report.refusal_rate:.3f} "
                f"violations={report.violations}"
            )

    @app.command("eval-c5")
    def eval_c5_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run C5 replay-safe macro admission and demotion probes."""
        report = evaluate_c5()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"admission_rate={report.admission_rate:.3f} "
                f"demoted={report.demoted} "
                f"violations={report.violations}"
            )


def register_meta_eval_commands(app: typer.Typer) -> None:
    """Register failure, ablation, and red-team evaluation commands."""
    register_failure_eval_commands(app)
    register_calibration_eval_commands(app)
    register_red_team_eval_commands(app)


def register_failure_eval_commands(app: typer.Typer) -> None:
    """Register failure-mode and ablation evaluation commands."""

    @app.command("eval-failures")
    def eval_failures_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run executable Criterion-1 failure-mode checks."""
        report = evaluate_failure_modes()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"passed={report.passed} "
                f"failures={len(report.failures)} "
                f"uncovered={len(report.uncovered_clauses)}"
            )

    @app.command("eval-ablations")
    def eval_ablations_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run executable control ablations."""
        report = evaluate_ablations()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            regressions = sum(result.expected_regression for result in report.results)
            typer.echo(f"passed={report.passed} regressions={regressions}")


def register_calibration_eval_commands(app: typer.Typer) -> None:
    """Register calibrated guard evaluation commands."""

    @app.command("eval-recall-shift")
    def eval_recall_shift_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run controlled source/rebuttal recall calibration under shift."""
        report = evaluate_recall_shift()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"passed={report.passed} "
                f"source_epsilon={report.source_epsilon:.3f} "
                f"rebuttal_epsilon={report.rebuttal_epsilon:.3f} "
                f"shifted_epsilon={report.shifted_epsilon:.3f}"
            )

    @app.command("eval-dependence-shift")
    def eval_dependence_shift_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run dependence residualization calibration under shift."""
        report = evaluate_dependence_shift()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"passed={report.passed} "
                f"epsilon_dep={report.epsilon_dep:.3f} "
                f"shifted_epsilon={report.shifted_epsilon:.3f}"
            )

    @app.command("eval-external-components")
    def eval_external_components_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run external cognitive-component authority checks."""
        report = evaluate_external_components()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"passed={report.passed} "
                f"violations={len(report.violations)} "
                f"external_inference={len(report.external_inference_dependencies)}"
            )

    @app.command("eval-entailment-attacks")
    def eval_entailment_attacks_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run held-out entailment false-support attacks."""
        report = evaluate_entailment_attacks()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"passed={report.passed} "
                f"false_support={len(report.false_support_attacks)} "
                f"caught={len(report.caught_false_support)}"
            )


def register_red_team_eval_commands(app: typer.Typer) -> None:
    """Register red-team replay evaluation commands."""

    @app.command("eval-red-team")
    def eval_red_team_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run M6 failure-mode and ablation replay."""
        report = red_team_replay()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"passed={report.passed} "
                f"failures={len(report.failures.failures)} "
                f"ablations={len(report.ablations.results)} "
                f"hard_domains={report.hard_domains.tasks}"
            )


def register_audit_eval_commands(app: typer.Typer) -> None:
    """Register baseline, phase, hard-domain, and compute audit commands."""

    @app.command("eval-baselines")
    def eval_baselines_command(
        limit: int = typer.Option(2, help="Absolute integer limit used for C1 splits."),
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Audit matched baseline availability and executable results."""
        report = evaluate_baseline_suite(limit=limit)
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"ready_for_claims={report.ready_for_claims} "
                f"missing={','.join(report.missing_families)}"
            )

    @app.command("eval-phase")
    def eval_phase_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run compression phase-transition and saturation diagnostics."""
        phase = evaluate_phase_transition()
        saturation = evaluate_saturation(phase.compression)
        payload = {
            "phase_transition": phase.to_dict(),
            "saturation": saturation.to_dict(),
        }
        if as_json:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(
                f"phase_observed={phase.observed} "
                f"saturated={saturation.saturated} "
                f"positive_macros={saturation.positive_macros}"
            )

    @app.command("eval-hard-domains")
    def eval_hard_domains_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run held-out hard-domain probes."""
        report = evaluate_hard_domains()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"solve_rate={report.solve_rate:.3f} "
                f"baseline_solve_rate={report.baseline_solve_rate:.3f} "
                f"tasks={report.tasks}"
            )

    @app.command("eval-compute")
    def eval_compute_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run matched compute-accounting checks."""
        report = evaluate_compute_accounting()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"passed={report.passed} "
                f"total_units={report.total_units:.3f} "
                f"budget={report.budget:.3f}"
            )


__all__ = ["register_eval_commands"]
