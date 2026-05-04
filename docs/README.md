# Documentation

This directory holds the canonical documentation for `compress`, the reference
implementation of **VPM-5.3: Verifier-Native Compressive Predictive Mechanisms**.

## Layout

- [`architecture/`](./architecture/) — the architecture specification, translated
  verbatim from `vpm_architecture_v5_3_improved.pdf` at repo root. One markdown
  file per section / appendix; equations are preserved as LaTeX rendered with
  MathJax (see `mkdocs.yml`).
- [`glossary.md`](./glossary.md) — symbols, types, abbreviations.
- [`references.md`](./references.md) — the 14 papers cited in the architecture.
- [`implementation/`](./implementation/) — implementation roadmap (VPM-0 minimal
  prototype, milestones, curriculum scheduling).

## Reading order for new contributors

1. Start with the [project README](https://github.com/SamMausberg/compress#readme) for the thesis.
2. Read the [architecture overview](./architecture/README.md) and the
   [Abstract](./architecture/00-abstract.md).
3. Skim §1 invariants and §2 mechanism definitions — these constrain everything else.
4. Drop into the section relevant to the module you intend to touch
   (e.g. §3 for the neural substrate, §6 for verification & authority).
5. Keep the [glossary](./glossary.md) open while reading.

## Building the docs site

```bash
uv sync --group docs
uv run mkdocs serve         # local preview at http://127.0.0.1:8000
uv run mkdocs build --strict
```

The `mkdocs.yml` at the repo root drives a Material-themed site with KaTeX math.
