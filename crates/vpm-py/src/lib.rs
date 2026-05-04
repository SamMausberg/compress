//! `vpm-py`: `PyO3` bindings exposing the VPM-5.3 Rust core to Python as
//! `vpm._native`.
//!
//! This crate is the **only** Rust → Python boundary. The Python package
//! `vpm` imports a single extension module, `vpm._native`, and re-exports
//! the wrapped Rust types from idiomatic Python modules under
//! `python/vpm/`.
//!
//! Module map (Python ↔ Rust):
//!
//! - `vpm._native.contract` ↔ `vpm_core::contract`
//! - `vpm._native.ledger`   ↔ `vpm_ledger`
//! - `vpm._native.dsl`      ↔ `vpm_dsl`
//! - `vpm._native.egraph`   ↔ `vpm_egraph`
//! - `vpm._native.authority`↔ `vpm_authority`
//! - `vpm._native.verify`   ↔ `vpm_verify`
//!
//! The exposed API is intentionally **narrow**: only the public boundary
//! types (`Contract`, `Ledger`, `TraceDag`, `EClass`, `AuthLabel`,
//! `Cert`, `Verifier`, `Gate*`) cross the Python boundary. Internal
//! state (slot bindings, e-graph hash-cons, dependence blocks) stays in
//! Rust.

use pyo3::{exceptions::PyValueError, prelude::*};
use vpm_core::{Contract, Value};
use vpm_dsl::{c0_add_program, c0_mul_program};
use vpm_verify::run_program;

/// The `vpm._native` Python extension module.
///
/// The `#[pyo3(name = "_native")]` override ensures the generated
/// `PyInit__native` symbol matches the file `vpm/_native.so` that
/// `maturin` produces (see `[tool.maturin]` in `pyproject.toml`).
#[pymodule]
#[pyo3(name = "_native")]
fn vpm_py_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add(
        "__doc__",
        "VPM-5.3 native core (Rust). See docs/architecture/.",
    )?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(c0_contract_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_c0_add_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_c0_arith_json, m)?)?;

    register_submodule(
        m,
        py,
        "contract",
        "Typed atoms, modes, contracts Γ — see vpm_core::contract.",
    )?;
    register_submodule(
        m,
        py,
        "ledger",
        "Append-only ledger Λ and trace DAG — see vpm_ledger.",
    )?;
    register_submodule(
        m,
        py,
        "dsl",
        "Typed bytecode and deterministic executor — see vpm_dsl.",
    )?;
    register_submodule(
        m,
        py,
        "egraph",
        "E-graph / equality-saturation wrapper — see vpm_egraph.",
    )?;
    register_submodule(
        m,
        py,
        "authority",
        "Authority lattice and declassification — see vpm_authority.",
    )?;
    register_submodule(
        m,
        py,
        "verify",
        "Verifier registry, e-values, falsifier, gates — see vpm_verify.",
    )?;
    Ok(())
}

/// Return the default C0 contract as JSON.
#[pyfunction]
fn c0_contract_json() -> PyResult<String> {
    serde_json::to_string_pretty(&Contract::c0()).map_err(json_error)
}

/// Run the Rust C0 add → canonicalize → execute → verify → gate flow.
#[pyfunction]
fn run_c0_add_json(left: i64, right: i64, expected: i64) -> PyResult<String> {
    let program = c0_add_program(left, right);
    let report = run_program(&program, Value::Int(expected)).map_err(PyValueError::new_err)?;
    serde_json::to_string_pretty(&report).map_err(json_error)
}

/// Run a Rust C0 arithmetic candidate through canonicalize → execute → verify → gate.
#[pyfunction]
fn run_c0_arith_json(operation: &str, left: i64, right: i64, expected: i64) -> PyResult<String> {
    let program = match operation {
        "add" => c0_add_program(left, right),
        "mul" => c0_mul_program(left, right),
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported C0 arithmetic operation: {operation}"
            )));
        }
    };
    let report = run_program(&program, Value::Int(expected)).map_err(PyValueError::new_err)?;
    serde_json::to_string_pretty(&report).map_err(json_error)
}

fn json_error(error: serde_json::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn register_submodule(
    parent: &Bound<'_, PyModule>,
    py: Python<'_>,
    name: &str,
    doc: &str,
) -> PyResult<()> {
    let submod = PyModule::new(py, name)?;
    submod.add("__doc__", doc)?;
    parent.add_submodule(&submod)
}
