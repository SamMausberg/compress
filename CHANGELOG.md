# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial repository scaffold for the VPM-5.3 architecture.
- Architecture documentation translated from `vpm_architecture_v5_3_improved.pdf`.
- Python + Rust hybrid project skeleton (PyTorch substrate, Rust execution / e-graph / ledger / verifier core).
- Build system: `uv` for Python, Cargo workspace for Rust, `maturin` for the PyO3 binding.
- CI pipeline (`ruff`, `pyright`, `pytest`, `cargo fmt`, `cargo clippy`, `cargo test`).

[Unreleased]: https://github.com/SamMausberg/compress/compare/HEAD...HEAD
