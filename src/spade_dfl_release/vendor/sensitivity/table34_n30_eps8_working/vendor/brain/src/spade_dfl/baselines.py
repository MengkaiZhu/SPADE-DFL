from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BaselineResult:
    theta: np.ndarray
    theta_history: np.ndarray
    losses: np.ndarray
    query_count: int


def train_exact_gradient(
    theta0: np.ndarray,
    gradient: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    evaluate: Callable[[np.ndarray], float],
    n_rounds: int,
    eta0: float,
    eta_exponent: float,
    rng: np.random.Generator,
) -> BaselineResult:
    theta = np.asarray(theta0, dtype=np.float64).copy()
    theta_history = [theta.copy()]
    losses = [float(evaluate(theta))]
    for k in range(n_rounds):
        eta = eta0 * (k + 1.0) ** (-eta_exponent)
        direction = np.asarray(gradient(theta, rng), dtype=np.float64)
        theta -= eta * direction
        if not np.all(np.isfinite(theta)):
            raise FloatingPointError(f"non-finite exact-gradient state at iteration {k}")
        losses.append(float(evaluate(theta)))
        theta_history.append(theta.copy())
    return BaselineResult(
        theta=theta,
        theta_history=np.vstack(theta_history),
        losses=np.asarray(losses),
        query_count=0,
    )
