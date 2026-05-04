"""Run the VPM-0 MVP vertical slice from source checkout.

Usage:
    python examples/vpm0/run.py
"""

from __future__ import annotations

import json

from vpm.evaluation import evaluate_c0
from vpm.infer import run_c0_add
from vpm.training import allocate_budget


def main() -> None:
    """Execute one task, then the bundled C0 curriculum."""
    result = run_c0_add(2, 3)
    metrics = evaluate_c0()
    budget = allocate_budget(metrics)
    payload = {
        "single": result.to_dict(),
        "metrics": metrics.to_dict(),
        "budget": budget.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
