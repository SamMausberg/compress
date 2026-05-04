# VPM-5.3 Architecture

This directory holds a faithful transcription of
`vpm_architecture_v5_3_improved.pdf` (architecture draft, May 2026), broken into
one markdown file per top-level section so that each implementation module can
quote the spec it is realising.

> **Source of truth.** When this transcription disagrees with the PDF, the PDF
> wins. Fix the markdown.

## Sections

| # | Title | File |
|---|-------|------|
| — | Title and Abstract | [`00-abstract.md`](./00-abstract.md) |
| 1 | Contract, ledger, and non-negotiable invariants | [`01-contract-ledger-invariants.md`](./01-contract-ledger-invariants.md) |
| 2 | Typed mechanisms and calibrated evidence | [`02-typed-mechanisms-evidence.md`](./02-typed-mechanisms-evidence.md) |
| 3 | Verifier-native neural substrate | [`03-neural-substrate.md`](./03-neural-substrate.md) |
| 4 | Natural-language contractization, rebuttal, and semantic realization | [`04-natural-language.md`](./04-natural-language.md) |
| 5 | Compiler, posterior, and support-preserving inference | [`05-compiler-posterior.md`](./05-compiler-posterior.md) |
| 6 | Verification, falsification, authority, and rendering | [`06-verification-authority.md`](./06-verification-authority.md) |
| 7 | Compression, memory, and capacity limits | [`07-compression-memory.md`](./07-compression-memory.md) |
| 8 | Training system and inefficiency controls | [`08-training-system.md`](./08-training-system.md) |
| 9 | Evaluation, failure, and minimal implementation | [`09-evaluation-failure.md`](./09-evaluation-failure.md) |
| A | Compact inference procedure | [`A-inference-procedure.md`](./A-inference-procedure.md) |
| B | Executable sanity conditions | [`B-sanity-conditions.md`](./B-sanity-conditions.md) |

## Style conventions

- **Verbatim prose.** Every definition, proposition, theorem, proof, invariant,
  and criterion is reproduced exactly as in the PDF.
- **Equations** are preserved as LaTeX inside `$$…$$` (display) or `$…$`
  (inline). Equation numbers are kept in the form `(n)` matching the PDF.
- **Section headings** follow the PDF.
- Each section ends with an **Implementation pointers** block (added, not part
  of the original document) listing the Rust crate(s) and Python module(s) that
  realise that section. This is the only material added to the original spec
  and is clearly delimited.
