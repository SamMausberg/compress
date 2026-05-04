"""Type stubs for the Rust extension module ``vpm._native``.

This file is hand-maintained alongside the PyO3 bindings in
``crates/vpm-py``. When you add a new ``#[pyclass]`` or ``#[pyfunction]``
on the Rust side, mirror the public signature here so Pyright /
``import vpm`` users see accurate types.
"""

from __future__ import annotations

from types import ModuleType

__version__: str
__doc__: str

contract: ModuleType
ledger: ModuleType
dsl: ModuleType
egraph: ModuleType
authority: ModuleType
verify: ModuleType

def c0_contract_json() -> str: ...
def run_c0_add_json(left: int, right: int, expected: int) -> str: ...
def run_c0_arith_json(operation: str, left: int, right: int, expected: int) -> str: ...
def run_c0_typed_json(
    operation: str,
    left_json: str,
    right_json: str,
    expected_json: str,
) -> str: ...
def run_c0_typed_policy_json(
    operation: str,
    left_json: str,
    right_json: str,
    expected_json: str,
    labels_json: str,
    risk_json: str,
) -> str: ...
