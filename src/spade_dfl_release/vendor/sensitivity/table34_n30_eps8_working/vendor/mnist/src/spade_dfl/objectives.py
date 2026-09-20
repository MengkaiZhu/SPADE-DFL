from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Union

import numpy as np


ArrayLikeU = Union[float, np.ndarray]


class Objective(Protocol):
    regularization: float

    def value(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        u: ArrayLikeU = 1.0,
    ) -> float: ...

    def gradient(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        u: ArrayLikeU = 1.0,
    ) -> np.ndarray: ...

    def component_gradients(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        u: ArrayLikeU = 1.0,
    ) -> np.ndarray: ...

    def component_values(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        u: ArrayLikeU = 1.0,
    ) -> np.ndarray: ...

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray: ...

    def accuracy(
        self, theta: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> float: ...


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


@dataclass(frozen=True)
class PaperSigmoidObjective:
    """Literal implementation of Reference [20], equation (21)."""

    regularization: float = 0.1

    def value(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        u: ArrayLikeU = 1.0,
    ) -> float:
        theta = np.asarray(theta, dtype=np.float64)
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        u_values = np.asarray(u, dtype=np.float64)
        margins = u_values * y * (x @ theta)
        return float(np.mean(stable_sigmoid(margins)) + self.regularization * (theta @ theta))

    def gradient(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        u: ArrayLikeU = 1.0,
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        u_values = np.asarray(u, dtype=np.float64)
        margins = u_values * y * (x @ theta)
        probabilities = stable_sigmoid(margins)
        weights = probabilities * (1.0 - probabilities) * u_values * y
        return np.mean(weights[:, None] * x, axis=0) + 2.0 * self.regularization * theta

    def component_gradients(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        u: ArrayLikeU = 1.0,
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        u_values = np.asarray(u, dtype=np.float64)
        margins = u_values * y * (x @ theta)
        probabilities = stable_sigmoid(margins)
        weights = probabilities * (1.0 - probabilities) * u_values * y
        return weights[:, None] * x + 2.0 * self.regularization * theta[None, :]

    def component_values(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        u: ArrayLikeU = 1.0,
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if theta.ndim == 1:
            theta = np.repeat(theta[None, :], x.shape[0], axis=0)
        if theta.shape != x.shape or y.shape != (x.shape[0],):
            raise ValueError("component inputs must have matching sample rows")
        u_values = np.broadcast_to(np.asarray(u, dtype=np.float64), y.shape)
        margins = u_values * y * np.einsum("ij,ij->i", x, theta)
        penalties = self.regularization * np.einsum("ij,ij->i", theta, theta)
        return stable_sigmoid(margins) + penalties

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        # Equation (21) is minimized, so the lower-loss label is opposite sign(x @ theta).
        score = np.asarray(x, dtype=np.float64) @ np.asarray(theta, dtype=np.float64)
        return np.where(score > 0.0, -1.0, 1.0)

    def accuracy(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(theta, x) == np.asarray(y)))


@dataclass(frozen=True)
class LTADMMNonconvexLogisticObjective:
    """Reference [13], Section IV, smooth nonconvex logistic objective."""

    regularization: float = 0.01

    def value(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        u: ArrayLikeU = 1.0,
    ) -> float:
        del u  # Reference [13] defines a deterministic black-box objective.
        theta = np.asarray(theta, dtype=np.float64)
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        margins = y * (x @ theta)
        logistic = np.mean(np.logaddexp(0.0, -margins))
        penalty = self.regularization * np.sum(theta**2 / (1.0 + theta**2))
        return float(logistic + penalty)

    def gradient(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        u: ArrayLikeU = 1.0,
    ) -> np.ndarray:
        del u
        theta = np.asarray(theta, dtype=np.float64)
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        margins = y * (x @ theta)
        weights = -y * stable_sigmoid(-margins)
        regularizer = (
            2.0 * self.regularization * theta / np.square(1.0 + theta**2)
        )
        return np.mean(weights[:, None] * x, axis=0) + regularizer

    def component_gradients(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        u: ArrayLikeU = 1.0,
    ) -> np.ndarray:
        del u
        theta = np.asarray(theta, dtype=np.float64)
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        margins = y * (x @ theta)
        weights = -y * stable_sigmoid(-margins)
        regularizer = (
            2.0 * self.regularization * theta / np.square(1.0 + theta**2)
        )
        return weights[:, None] * x + regularizer[None, :]

    def component_values(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        u: ArrayLikeU = 1.0,
    ) -> np.ndarray:
        del u
        theta = np.asarray(theta, dtype=np.float64)
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if theta.ndim == 1:
            theta = np.repeat(theta[None, :], x.shape[0], axis=0)
        if theta.shape != x.shape or y.shape != (x.shape[0],):
            raise ValueError("component inputs must have matching sample rows")
        margins = y * np.einsum("ij,ij->i", x, theta)
        penalties = self.regularization * np.sum(
            theta**2 / (1.0 + theta**2), axis=1
        )
        return np.logaddexp(0.0, -margins) + penalties

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        score = np.asarray(x, dtype=np.float64) @ np.asarray(theta, dtype=np.float64)
        return np.where(score >= 0.0, 1.0, -1.0)

    def accuracy(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(theta, x) == np.asarray(y)))


def build_objective(name: str, regularization: float) -> Objective:
    if name == "paper_sigmoid":
        return PaperSigmoidObjective(regularization)
    if name == "lt_admm_nonconvex_logistic":
        return LTADMMNonconvexLogisticObjective(regularization)
    raise ValueError(f"unsupported objective_model {name!r}")


def central_difference(
    function: Callable[[np.ndarray], float],
    theta: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    result = np.empty_like(theta)
    for index in range(theta.size):
        offset = np.zeros_like(theta)
        offset[index] = epsilon
        result[index] = (function(theta + offset) - function(theta - offset)) / (2.0 * epsilon)
    return result
