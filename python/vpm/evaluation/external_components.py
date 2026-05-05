"""Compatibility exports for external component audits."""

from __future__ import annotations

from vpm.audit.external_components import (
    CognitiveComponent,
    ComponentFamily,
    ComponentRole,
    ExternalComponentReport,
    component_inventory,
    dirty_external_component_probe,
    evaluate_external_components,
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
