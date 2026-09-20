from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .zo import normalized_rademacher


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
