---
name: Bug report
about: A reproducible defect — wrong behavior, panic, crash, or failing test.
title: "[bug] "
labels: ["bug"]
---

## Summary

<!-- One sentence: what's broken. -->

## Reproduction

```text
Steps to reproduce, ideally as a minimal command or test:

1. uv sync
2. uv run python -c "..."
3. ...
```

## Expected vs actual

- **Expected:** ...
- **Actual:** ...

## Environment

- OS:
- Python: `python --version`
- Rust: `rustc --version`
- `uv --version`:
- `cargo tree` excerpt for the relevant crate, if relevant:

## Architecture cross-reference

If the bug is a violation of an architecture invariant, cite the section
and equation number from `docs/architecture/` (e.g. "Invariant 3 of §1",
"eq. 95 of §5"). This makes review and regression tests faster.

## Logs / traceback

```text
<paste here>
```
