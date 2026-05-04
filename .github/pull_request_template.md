<!--
Thanks for contributing to compress / VPM-5.3. Please keep PRs focused
and link the architecture section / equation numbers you're realising
or modifying.
-->

## Summary

<!-- One paragraph: what changes and why. -->

## Architecture cross-reference

- Section(s):
- Equation(s):
- Implementation milestone (M0..M6, see `docs/implementation/README.md`):

## Checks

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] `uv run ruff check . && uv run ruff format --check .`
- [ ] `uv run pyright python`
- [ ] `uv run pytest -q`
- [ ] If this PR touches the spec semantics: relevant
      `docs/architecture/` page updated and the verbatim wording from
      `vpm_architecture_v5_3_improved.pdf` is preserved.
- [ ] If this PR adds or modifies a verifier / retriever / capsule:
      property tests under `tests/integration/sanity/` and (where
      applicable) a failure-mode entry under
      `tests/integration/failure_modes/`.

## Notes for reviewers

<!-- Anything non-obvious, or specific files you want eyes on. -->
