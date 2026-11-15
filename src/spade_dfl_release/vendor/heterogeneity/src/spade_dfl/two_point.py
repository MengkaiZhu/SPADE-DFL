from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from .zo import normalized_rademacher


BatchU = Union[float, np.ndarray, Callable[[], float]]


@dataclass(frozen=True)
class TwoPointQueryResult:
    vector: np.ndarray
    direction: np.ndarray
    value_plus: float
    value_minus: float
    cz: float
    query_count: int = 2


def symmetric_two_point_query(
    oracle: Callable[[np.ndarray], float],
    theta: np.ndarray,
    mu: float,
    rng: np.random.Generator,
) -> TwoPointQueryResult:
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    theta = np.asarray(theta, dtype=np.float64)
    direction, cz = normalized_rademacher(rng, theta.size)
    value_plus = float(oracle(theta + mu * direction))
    value_minus = float(oracle(theta - mu * direction))
    if not np.isfinite(value_plus) or not np.isfinite(value_minus):
        raise FloatingPointError("black-box query returned a non-finite value")
    vector = direction * (value_plus - value_minus) / (2.0 * mu * cz)
    return TwoPointQueryResult(
        vector=vector,
        direction=direction,
        value_plus=value_plus,
        value_minus=value_minus,
        cz=cz,
    )


def symmetric_two_point_component_batch(
    objective,
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: float,
    rng: np.random.Generator,
    u: BatchU,
    noise_std: float,
) -> tuple[np.ndarray, int]:
    """Evaluate independent symmetric component queries in two objective calls."""
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    if noise_std < 0.0:
        raise ValueError("noise_std must be nonnegative")
    theta = np.asarray(theta, dtype=np.float64)
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y))
    if theta.ndim != 1:
        raise ValueError("theta must be a one-dimensional parameter vector")
    count = x.shape[0]
    dimension = theta.size
    if y.shape != (count,):
        raise ValueError("labels must have one entry per sample")

    if callable(u):
        u_values = np.empty(count, dtype=np.float64)
    else:
        u_values = np.broadcast_to(
            np.asarray(u, dtype=np.float64), (count,)
        ).copy()
    directions = np.empty((count, dimension), dtype=np.float64)
    plus_noise = np.zeros(count, dtype=np.float64)
    minus_noise = np.zeros(count, dtype=np.float64)
    for sample in range(count):
        if callable(u):
            u_values[sample] = float(u())
        directions[sample], _ = normalized_rademacher(rng, dimension)
        if noise_std > 0.0:
            plus_noise[sample] = float(rng.normal(0.0, noise_std))
            minus_noise[sample] = float(rng.normal(0.0, noise_std))

    points_plus = theta[None, :] + mu * directions
    points_minus = theta[None, :] - mu * directions
    values_plus = objective.component_values(
        points_plus, x, y, u=u_values
    ) + plus_noise
    values_minus = objective.component_values(
        points_minus, x, y, u=u_values
    ) + minus_noise
    if not np.all(np.isfinite(values_plus)) or not np.all(
        np.isfinite(values_minus)
    ):
        raise FloatingPointError("black-box batch query returned a non-finite value")
    cz = 1.0 / float(dimension)
    estimates = directions * (
        (values_plus - values_minus) / (2.0 * mu * cz)
    )[:, None]
    return estimates, 2 * count
