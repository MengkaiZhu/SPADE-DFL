from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .objectives import Objective


@dataclass(frozen=True)
class QueryResult:
    vector: np.ndarray
    unscaled_vector: np.ndarray
    direction: np.ndarray
    value: float
    cz: float
    query_count: int = 1


def normalized_rademacher(
    rng: np.random.Generator, dimension: int
) -> tuple[np.ndarray, float]:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    signs = rng.choice(np.array([-1.0, 1.0]), size=dimension)
    direction = signs / np.sqrt(float(dimension))
    return direction, 1.0 / float(dimension)


def spade_one_point_query(
    oracle: Callable[[np.ndarray], float],
    theta: np.ndarray,
    mu: float,
    rng: np.random.Generator,
) -> QueryResult:
    if mu <= 0:
        raise ValueError("mu must be positive")
    theta = np.asarray(theta, dtype=np.float64)
    direction, cz = normalized_rademacher(rng, theta.size)
    value = float(oracle(theta + mu * direction))
    if not np.isfinite(value):
        raise FloatingPointError("black-box query returned a non-finite value")
    unscaled = direction * value
    return QueryResult(
        vector=unscaled / cz,
        unscaled_vector=unscaled,
        direction=direction,
        value=value,
        cz=cz,
    )


def spade_one_point_batch_queries(
    objective: Objective,
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: float,
    directions: np.ndarray,
    multiplicative_noise: np.ndarray,
    additive_noise: np.ndarray,
) -> np.ndarray:
    """Vectorized SPADE-DFL Eq. (10) for independent sample queries."""
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    theta = np.asarray(theta, dtype=np.float64)
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    directions = np.atleast_2d(np.asarray(directions, dtype=np.float64))
    multiplicative_noise = np.atleast_1d(
        np.asarray(multiplicative_noise, dtype=np.float64)
    )
    additive_noise = np.atleast_1d(np.asarray(additive_noise, dtype=np.float64))
    count, dimension = x.shape
    if theta.shape != (dimension,):
        raise ValueError("theta dimension does not match sample features")
    if (
        y.shape != (count,)
        or directions.shape != (count, dimension)
        or multiplicative_noise.shape != (count,)
        or additive_noise.shape != (count,)
    ):
        raise ValueError("batch query inputs must share the same sample count")
    points = theta[None, :] + mu * directions
    values = objective.component_values(
        points, x, y, u=multiplicative_noise
    ) + additive_noise
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("black-box batch query returned a non-finite value")
    cz = 1.0 / float(dimension)
    return directions * values[:, None] / cz
