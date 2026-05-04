//! `vpm-py`: PyO3 bindings exposing the VPM-5.3 Rust core to Python as
//! `vpm._native`.
//!
//! This crate is the **only** Rust → Python boundary. The Python package
//! `vpm` imports a single extension module, `vpm._native`, and re-exports
//! the wrapped Rust types from idiomatic Python modules under
//! `python/vpm/`.
//!
//! Module map (Python ↔ Rust):
//!
//! - `vpm._native.contract` ↔ [`vpm_core::contract`]
//! - `vpm._native.ledger`   ↔ [`vpm_ledger`]
//! - `vpm._native.dsl`      ↔ [`vpm_dsl`]
//! - `vpm._native.egraph`   ↔ [`vpm_egraph`]
//! - `vpm._native.authority`↔ [`vpm_authority`]
//! - `vpm._native.verify`   ↔ [`vpm_verify`]
//!
//! The exposed API is intentionally **narrow**: only the public boundary
//! types (`Contract`, `Ledger`, `TraceDag`, `EClass`, `AuthLabel`,
//! `Cert`, `Verifier`, `Gate*`) cross the Python boundary. Internal
//! state (slot bindings, e-graph hash-cons, dependence blocks) stays in
//! Rust.

use pyo3::prelude::*;

/// The `vpm._native` Python extension module.
///
/// The `#[pyo3(name = "_native")]` override ensures the generated
/// `PyInit__native` symbol matches the file `vpm/_native.so` that
/// `maturin` produces (see `[tool.maturin]` in `pyproject.toml`).
#[pymodule]
#[pyo3(name = "_native")]
fn vpm_py_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "VPM-5.3 native core (Rust). See docs/architecture/.")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
