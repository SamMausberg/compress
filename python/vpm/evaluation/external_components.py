"""External cognitive-component authority checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ComponentFamily(StrEnum):
    """Families of cognitive components that can influence VPM reports."""

    INTERNAL = "internal"
    EXTERNAL_LLM = "external_llm"


class ComponentRole(StrEnum):
    """Allowed roles for cognitive components."""

    INFERENCE = "inference"
    BASELINE = "baseline"
    TEACHER = "teacher"


@dataclass(frozen=True)
class CognitiveComponent:
    """One component that may propose, score, or certify a claim."""

    name: str
    family: ComponentFamily
    role: ComponentRole
    budget_logged: bool
    certificate_authority: bool
    configured: bool = True

    @property
    def allowed(self) -> bool:
        """True when the component is allowed by the external-LLM policy."""
        if not self.configured:
            return True
        if self.family is ComponentFamily.INTERNAL:
            return True
        return (
            self.role in {ComponentRole.BASELINE, ComponentRole.TEACHER}
            and self.budget_logged
            and not self.certificate_authority
        )

    @property
    def external_inference_dependency(self) -> bool:
        """True when an external LLM participates in inference."""
        return (
            self.configured
            and self.family is ComponentFamily.EXTERNAL_LLM
            and self.role is ComponentRole.INFERENCE
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly component record."""
        return {
            "name": self.name,
            "family": self.family.value,
            "role": self.role.value,
            "budget_logged": self.budget_logged,
            "certificate_authority": self.certificate_authority,
            "configured": self.configured,
            "allowed": self.allowed,
            "external_inference_dependency": self.external_inference_dependency,
        }


@dataclass(frozen=True)
class ExternalComponentReport:
    """Audit report for external cognitive-component authority."""

    components: tuple[CognitiveComponent, ...]

    @property
    def violations(self) -> tuple[CognitiveComponent, ...]:
        """Components that violate the external cognitive-component policy."""
        return tuple(component for component in self.components if not component.allowed)

    @property
    def external_inference_dependencies(self) -> tuple[CognitiveComponent, ...]:
        """External LLM components in the inference path."""
        return tuple(
            component for component in self.components if component.external_inference_dependency
        )

    @property
    def passed(self) -> bool:
        """True when no external LLM can certify inference-time cognition."""
        return not self.violations and not self.external_inference_dependencies

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly external-component report."""
        return {
            "passed": self.passed,
            "components": [component.to_dict() for component in self.components],
            "violations": [component.to_dict() for component in self.violations],
            "external_inference_dependencies": [
                component.to_dict() for component in self.external_inference_dependencies
            ],
        }


def component_inventory() -> tuple[CognitiveComponent, ...]:
    """Return the default VPM-0 cognitive components."""
    return (
        CognitiveComponent(
            name="compiler-dsl",
            family=ComponentFamily.INTERNAL,
            role=ComponentRole.INFERENCE,
            budget_logged=True,
            certificate_authority=False,
        ),
        CognitiveComponent(
            name="exact-verifier",
            family=ComponentFamily.INTERNAL,
            role=ComponentRole.INFERENCE,
            budget_logged=True,
            certificate_authority=True,
        ),
        CognitiveComponent(
            name="local-neural-baselines",
            family=ComponentFamily.INTERNAL,
            role=ComponentRole.BASELINE,
            budget_logged=True,
            certificate_authority=False,
        ),
        CognitiveComponent(
            name="external-llm-teacher-placeholder",
            family=ComponentFamily.EXTERNAL_LLM,
            role=ComponentRole.TEACHER,
            budget_logged=True,
            certificate_authority=False,
            configured=False,
        ),
    )


def evaluate_external_components(
    components: tuple[CognitiveComponent, ...] | None = None,
) -> ExternalComponentReport:
    """Evaluate cognitive components against the external-LLM authority policy."""
    return ExternalComponentReport(component_inventory() if components is None else components)


def dirty_external_component_probe() -> ExternalComponentReport:
    """Inject an unlogged external LLM inference component."""
    return evaluate_external_components(
        (
            CognitiveComponent(
                name="unlogged-llm-inference",
                family=ComponentFamily.EXTERNAL_LLM,
                role=ComponentRole.INFERENCE,
                budget_logged=False,
                certificate_authority=True,
            ),
        )
    )


__all__ = [
    "CognitiveComponent",
    "ComponentFamily",
    "ComponentRole",
    "ExternalComponentReport",
    "component_inventory",
    "dirty_external_component_probe",
    "evaluate_external_components",
]
