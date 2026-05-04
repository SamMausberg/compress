"""Train and evaluate the VPM-0 trainable C0 prototype.

Usage:
    python examples/vpm0/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

from vpm.tasks import typed_hidden_task
from vpm.training import (
    TrainingConfig,
    evaluate_saved_c1_prototype,
    evaluate_saved_prototype,
    run_learned_task,
    train_c0_prototype,
    train_c1_prototype,
)


def main() -> None:
    """Train C0/C1 prototypes and show verifier-gated learned outcomes."""
    c0_artifact = Path("artifacts/vpm_c0_prototype.npz")
    c1_artifact = Path("artifacts/vpm_c1_prototype.npz")
    c0_model, c0_report = train_c0_prototype(
        TrainingConfig(limit=2, epochs=12, hidden_dim=12, device="cpu", artifact=c0_artifact)
    )
    _, c1_report = train_c1_prototype(
        TrainingConfig(limit=2, epochs=12, hidden_dim=12, device="cpu", artifact=c1_artifact)
    )
    hidden = typed_hidden_task("t1", "t2", "t1t2")
    inference = run_learned_task(c0_model, hidden)
    authority_refusal = run_learned_task(c0_model, hidden, labels=("capability",))
    risk_refusal = run_learned_task(c0_model, hidden, risk={"privacy": 0.1})
    payload = {
        "c0_training": c0_report.to_dict(),
        "c0_reloaded_eval": evaluate_saved_prototype(c0_artifact, limit=2, device="cpu").to_dict(),
        "c1_training": c1_report.to_dict(),
        "c1_reloaded_eval": evaluate_saved_c1_prototype(
            c1_artifact, limit=2, device="cpu"
        ).to_dict(),
        "inference": inference.to_dict(),
        "authority_refusal": authority_refusal.to_dict(),
        "risk_refusal": risk_refusal.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
