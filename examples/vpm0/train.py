"""Train and evaluate the VPM-0 trainable C0 prototype.

Usage:
    python examples/vpm0/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

from vpm.tasks import arithmetic_task
from vpm.training import (
    TrainingConfig,
    evaluate_saved_prototype,
    run_learned_task,
    train_c0_prototype,
)


def main() -> None:
    """Train, reload, evaluate, and run one verifier-gated learned proposal."""
    artifact = Path("artifacts/vpm_c0_prototype.npz")
    model, report = train_c0_prototype(
        TrainingConfig(limit=2, epochs=8, hidden_dim=8, device="cpu", artifact=artifact)
    )
    inference = run_learned_task(model, arithmetic_task("mul", 2, 1))
    payload = {
        "training": report.to_dict(),
        "reloaded_eval": evaluate_saved_prototype(artifact, limit=2, device="cpu").to_dict(),
        "inference": inference.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
