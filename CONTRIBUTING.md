# Contributing to `compress`

Thanks for your interest. This project is a research prototype of the
**VPM-5.3** architecture (see [`docs/architecture/`](./docs/architecture/)
and the root [`vpm_architecture_v5_3_improved.pdf`](./vpm_architecture_v5_3_improved.pdf)).
It is currently maintained by a single author; contributions are welcome
but the bar is "does this make the system more clearly correct,
faster, or better calibrated."

## Before you start

Read, in order:

1. The [README](./README.md) — the project thesis.
2. [`docs/architecture/README.md`](./docs/architecture/README.md) — the
   spec sections and which crate / Python module realises each.
3. [`docs/implementation/README.md`](./docs/implementation/README.md) —
   the milestone plan (M0…M6). Work that doesn't move us forward on a
   milestone or close a Criterion 1 (§9) failure clause is unlikely
   to land.

The PDF is the source of truth. When the markdown disagrees with the
PDF, fix the markdown.

## Local setup

```bash
# Rust toolchain (pinned by rust-toolchain.toml)
rustup show

# Python + uv
uv sync --locked --group dev --group docs

# Build the maturin extension
uv sync --locked --group dev    # implicitly builds vpm._native

# Pre-commit hooks
uv run pre-commit install
```

## The local sanity loop

Run before every commit; CI runs the same checks plus security scanners:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo nextest run --workspace --locked          # or: cargo test --workspace --all-targets --locked
cargo doc --workspace --no-deps --locked

uv run ruff check .
uv run ruff format --check .
uv run pyright python
uv run pytest -q
uv run pip-audit
uv run deptry python
```

## Pull requests

- One milestone slice per PR. M0.3 ("typed bytecode") is one PR; don't
  bundle it with M0.4 ("e-graph wrapper").
- The crate or module you touch should have a docstring quoting the
  spec equation it realises. Equations are referenced as
  `§5 eq. 88` or `(eq. 88)`. Keep the convention consistent with the
  existing crates.
- `unsafe` is `forbid`ed at workspace level. If you need to relax that
  for a specific crate (e.g. an FFI shim), justify the relaxation in
  the PR description.
- A claim that something is "certified" or "verified" in code, docs,
  or commit message must point to the corresponding mathematical
  certificate in the spec. The architecture is *very* explicit about
  what "evidence" means; loose use of those words confuses everyone.
- Tests live next to the change. Behavior-level invariants from
  Appendix B go in `tests/integration/sanity/`; Criterion 1 regression
  tests in `tests/integration/failure_modes/`.

## Commit message style

Short imperative subject; explain *why* in the body. Conventional
Commits prefixes (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`,
`test:`) are used but not enforced.

## Reporting issues

- Bugs / unexpected behavior → GitHub issue using the bug report
  template.
- Suspected security issues → see [`SECURITY.md`](./SECURITY.md). Do
  *not* open a public issue.
- Spec disagreements → file an issue with a quote of the PDF and the
  conflicting code/markdown.

## Conduct

See [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md). The short version:
disagree about ideas freely; don't make it about people.
