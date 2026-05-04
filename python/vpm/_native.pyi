"""Type stubs for the Rust extension module ``vpm._native``.

This file is hand-maintained alongside the PyO3 bindings in
``crates/vpm-py``. When you add a new ``#[pyclass]`` or ``#[pyfunction]``
on the Rust side, mirror the public signature here so Pyright /
``import vpm`` users see accurate types.
"""

from __future__ import annotations

__version__: str
__doc__: str
