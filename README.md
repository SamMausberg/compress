# compress

An experimental departure from LLMs and transformer-based reasoning.

## What this is

`compress` is a research project where I am trying to build a different kind of AI system from the ground up.

The basic idea stems from the rather simple observation that while current frontier models are extremely intelligent, they suffer from inefficiency. To get state-of-the-art intelligence today, you generally need massive datasets, massive compute, massive models, and a training process that is completely out of reach for an individual researcher or small team. Furthermore, models struggle on basic tasks while being able to solve quantum physics problems, and general approaches to fix these basic errors require substantial RL on top as part of post-training.

I do not think this has to be the final form of machine intelligence. While it may be enough to reach AGI, I cannot help but be interested in other paradigms. 

This repo is my attempt to explore whether a much smaller, more structured, more efficient architecture can learn and reason in a way that is not just a scaled-down LLM.

## Core thesis

The project is based on a few assumptions I want to test seriously:

1. Intelligence should not require internet-scale token prediction.
2. A model should be able to compress useful structure, not just memorize statistical patterns.
3. Reasoning should be more explicit, modular, and inspectable than it is in current LLMs.
4. Continual learning should be a basic property of the system.
5. A good architecture should be trainable and testable on limited compute.

The goal is not to make a toy chatbot.

The goal is to investigate whether a non-LLM, non-transformer architecture can become genuinely useful, general, and efficient.

## What I am trying to build

I am exploring an AI system built around:

- compressed internal representations
- structured memory
- non-autoregressive inference
- modular reasoning components
- explicit world-model construction
- self-evaluation and verification
- continual learning without catastrophic forgetting

The hope is that by changing the architecture, not just scaling the same architecture harder, we can get better intelligence per parameter, per token, and per watt.

## Why not just use transformers?

Transformers work extremely well. This project is not based on pretending they are bad.

The problem is that their success has made the field extremely dependent on scale. Bigger models, more data, more GPUs, more RL, more inference-time compute. That path will probably keep working, but it is not accessible, and it may not be the most elegant or efficient route to intelligence.

This repo is an attempt to ask a different question:

> What would an AI architecture look like if efficiency, compression, continual learning, and structured reasoning were treated as first principles?

## Goals

The long-term goals are:

- build a working prototype of a non-transformer AI system
- make the system trainable on limited local compute
- test whether it can learn useful abstractions from much less data
- support conversation and tool use eventually
- make reasoning more inspectable than in standard LLMs
- compare it honestly against small language models and other baselines

The immediate goal is much smaller:

Build the simplest serious version of the architecture, test the core ideas, and avoid adding unnecessary complexity before the base system works.

## Non-goals

This is not:

- an LLM wrapper
- a prompt engineering project
- a chatbot skin
- a small transformer clone
- a vague AGI manifesto
- a benchmark-chasing repo with no architecture behind it

This is an architecture experiment.

It might fail. The point is to make the idea concrete enough that it can actually be tested.

## Current status

Early research and implementation.

The focus right now is on designing the base architecture, defining the training process, and building the first minimal prototype.

## MVP vertical slice

The repo includes a runnable VPM-0 kernel for cheap-verifier C0 tasks. It
connects the Rust core model, ledger, DSL executor, canonicalization witness,
authority/risk gate, verifier kernel, Python compiler/retrieval/substrate
proposal/memory/inference/evaluation modules, CLI, example, and tests.

```bash
cargo test --workspace
uv run maturin develop
uv run vpm run-c0-add 2 3
uv run vpm run-c0-add 2 3 --json
uv run vpm run-c0 mul 6 7 --json
uv run vpm run-c0 concat ab cd --json
uv run vpm run-c0 eq 5 5 --json
uv run vpm run-c0 add 2 3 --authority capability --json
uv run vpm run-c0 add 2 3 --risk-privacy 0.1 --json
uv run vpm eval-c0 --json
uv run vpm eval-c1 --json
uv run vpm train-c0 --epochs 80 --json
uv run vpm eval-prototype --json
uv run vpm infer-c0 mul 6 7 --json
uv run vpm infer-c0-auto ab cd abcd --json
uv run vpm stages
uv run python examples/vpm0/run.py
uv run python examples/vpm0/train.py
uv run pytest -q
```

The implemented C0 path is intentionally small: `add left right` or
`mul left right` is compiled
to typed bytecode, canonicalized with an auditable witness stack, executed into
the ledger/trace DAG, verified by an exact primitive verifier, gated by
certificate/route/authority/risk checks, rendered only after the gate passes,
and admitted to active memory with its report.

The trainable prototype adds a small recurrent, typed-event substrate on top
of the same verifier-native path. It learns to propose C0 arithmetic
operations, writes an `artifacts/vpm_c0_prototype.npz` weight file, runs held-out
tasks through the native executor/verifier/gate, reports rejected proposals,
and compares solve rate against one-candidate majority/add-only baselines plus
an exact enumerative upper bound. Learned proposals never certify themselves.
`eval-prototype --json` includes per-task traces with expected/proposed
operations, pass/refusal status, gate reasons, errors, and admitted memory
keys, plus compression/frontier metrics for learned one-shot utility,
active-memory growth, and movement against the enumerative baseline.
The substrate input is operation-hidden: it sees typed operands plus the
expected value, proposes an operation, and the native executor/verifier/gate
either certifies or rejects that proposal.
The `run-c0` command also exposes authority and componentwise-risk probes;
even an exactly verified value is rendered as `refusal` and skipped by memory
admission when the authority or risk gate fails.
The C1 executable subset adds hidden-schema tasks whose observations carry
only typed operands plus expected values; evaluation bridges them back through
the same native C0 executor/verifier/gate.
`eval-c0 --json` and `eval-c1 --json` summarize source coverage, rebuttal
clearance, and realization-loss rates in their `evidence` blocks.

Python API:

```python
from pathlib import Path

from vpm.tasks import typed_hidden_task
from vpm.training import TrainingConfig, run_learned_task, train_c0_prototype

model, report = train_c0_prototype(
    TrainingConfig(epochs=80, artifact=Path("artifacts/vpm_c0_prototype.npz"))
)
print(
    report.heldout.solve_rate,
    report.heldout.compression.frontier_delta_vs_enumerative,
)

inference = run_learned_task(model, typed_hidden_task("ab", "cd", "abcd"))
print(inference.proposal.to_dict())
print(inference.result.to_dict()["native_report"]["gate"])
```

## Note

The project is called `compress` because I think compression is central to intelligence.

Compression as the ability to discover structure, preserve what matters, discard what does not, and reuse abstractions across situations.

A system that truly understands should not need to see every possible version of a problem. It should be able to form compact internal models that generalize.

That is the bet this repo is trying to test.
