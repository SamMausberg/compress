"""Minimal smoke test — keeps `pytest -q` from exiting 5 (no tests collected)
during the placeholder phase, and verifies the Rust↔Python boundary
imports cleanly. Replace with real coverage as M0 modules land.
"""

from __future__ import annotations

import vpm
from vpm._native import authority, contract, dsl, egraph, ledger, verify
from vpm.infer import run_c0_add


def test_version() -> None:
    assert vpm.__version__ == "0.0.0"


def test_native_boundary() -> None:
    # Six submodules registered by `crates/vpm-py/src/lib.rs`.
    for mod in (contract, ledger, dsl, egraph, authority, verify):
        assert mod.__doc__ is not None


def test_public_vertical_slice() -> None:
    result = run_c0_add(2, 3)
    assert result.rendered.startswith("5 ")
    assert result.route == "solve"
