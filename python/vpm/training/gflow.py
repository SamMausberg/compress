"""GFlowNet reward and trajectory-balance scalar helpers."""

from __future__ import annotations

from math import exp, log


def mechanism_reward(
    energy: float,
    *,
    certificate: float,
    certificate_threshold: float,
    support_loss: float,
    support_max: float,
    frontier_delta: float,
    risk_ok: bool,
    epsilon: float = 1.0e-6,
) -> float:
    """Eq. 166 reward with hard certificate/support/frontier/risk gates."""
    gates_pass = (
        certificate >= certificate_threshold
        and support_loss <= support_max
        and frontier_delta > 0.0
        and risk_ok
    )
    return epsilon + (exp(-energy) if gates_pass else 0.0)


def trajectory_balance_loss(
    log_z: float,
    forward_log_probs: tuple[float, ...],
    backward_log_probs: tuple[float, ...],
    reward: float,
) -> float:
    """Eq. 167 trajectory-balance loss."""
    residual = log_z + sum(forward_log_probs) - log(max(reward, 1.0e-12)) - sum(backward_log_probs)
    return residual * residual


__all__ = ["mechanism_reward", "trajectory_balance_loss"]
