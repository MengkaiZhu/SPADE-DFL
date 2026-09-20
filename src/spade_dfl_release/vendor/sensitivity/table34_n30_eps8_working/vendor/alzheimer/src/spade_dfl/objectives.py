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


@dataclass(frozen=True)
class MulticlassSoftmaxObjective:
    """Linear multiclass cross-entropy objective for black-box EMNIST tests."""

    regularization: float
    feature_dimension: int
    n_classes: int

    def __post_init__(self) -> None:
        if self.regularization < 0:
            raise ValueError("regularization cannot be negative")
        if self.feature_dimension <= 0:
            raise ValueError("feature_dimension must be positive")
        if self.n_classes <= 1:
            raise ValueError("n_classes must be greater than one")

    @property
    def dimension(self) -> int:
        return self.feature_dimension * self.n_classes

    @property
    def one_point_baseline(self) -> float:
        # Subtracting this known constant defines an optimization-equivalent
        # shifted loss and needs no second black-box query.
        return float(np.log(self.n_classes))

    def _shared_weights(self, theta: np.ndarray) -> np.ndarray:
        values = np.asarray(theta, dtype=np.float64)
        if values.shape != (self.dimension,):
            raise ValueError(
                f"parameter dimension must be {self.dimension}, got {values.shape}"
            )
        return values.reshape(self.feature_dimension, self.n_classes)

    @staticmethod
    def _logsumexp(logits: np.ndarray) -> np.ndarray:
        maximum = np.max(logits, axis=1, keepdims=True)
        return maximum[:, 0] + np.log(
            np.exp(logits - maximum).sum(axis=1)
        )

    def _validate_samples(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        features = np.atleast_2d(np.asarray(x, dtype=np.float64))
        labels = np.atleast_1d(np.asarray(y, dtype=np.int64))
        if features.shape[1] != self.feature_dimension:
            raise ValueError(
                f"sample feature dimension must be {self.feature_dimension}"
            )
        if labels.shape != (features.shape[0],):
            raise ValueError("labels must have one entry per sample")
        if np.any(labels < 0) or np.any(labels >= self.n_classes):
            raise ValueError(f"labels must lie in 0..{self.n_classes - 1}")
        return features, labels

    def probabilities(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        features = np.atleast_2d(np.asarray(x, dtype=np.float64))
        weights = self._shared_weights(theta)
        if features.shape[1] != self.feature_dimension:
            raise ValueError(
                f"sample feature dimension must be {self.feature_dimension}"
            )
        logits = features @ weights
        logits -= np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(logits)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def value(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        u: ArrayLikeU = 1.0,
    ) -> float:
        return float(np.mean(self.component_values(theta, x, y, u)))

    def gradient(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        u: ArrayLikeU = 1.0,
    ) -> np.ndarray:
        weights = self._shared_weights(theta)
        features, labels = self._validate_samples(x, y)
        u_values = np.broadcast_to(
            np.asarray(u, dtype=np.float64), labels.shape
        )
        logits = u_values[:, None] * (features @ weights)
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        probabilities[np.arange(len(labels)), labels] -= 1.0
        gradient = features.T @ (probabilities * u_values[:, None])
        gradient /= len(labels)
        gradient += 2.0 * self.regularization * weights
        return gradient.reshape(self.dimension)

    def component_gradients(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        u: ArrayLikeU = 1.0,
    ) -> np.ndarray:
        theta_values = np.asarray(theta, dtype=np.float64)
        weights = self._shared_weights(theta_values)
        features, labels = self._validate_samples(x, y)
        u_values = np.broadcast_to(
            np.asarray(u, dtype=np.float64), labels.shape
        )
        logits = u_values[:, None] * (features @ weights)
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        probabilities[np.arange(len(labels)), labels] -= 1.0
        probabilities *= u_values[:, None]
        gradients = np.einsum("nf,nc->nfc", features, probabilities)
        gradients += 2.0 * self.regularization * weights[None, :, :]
        return gradients.reshape(len(labels), self.dimension)

    def component_values(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        u: ArrayLikeU = 1.0,
    ) -> np.ndarray:
        features, labels = self._validate_samples(x, y)
        theta_values = np.asarray(theta, dtype=np.float64)
        if theta_values.ndim == 1:
            weights = self._shared_weights(theta_values)
        else:
            if theta_values.shape != (len(features), self.dimension):
                raise ValueError(
                    "component parameter dimension or sample count does not match"
                )
            weights = theta_values.reshape(
                len(features), self.feature_dimension, self.n_classes
            )
        u_values = np.broadcast_to(
            np.asarray(u, dtype=np.float64), labels.shape
        )
        if theta_values.ndim == 1:
            logits = u_values[:, None] * (features @ weights)
            penalties = self.regularization * np.sum(theta_values**2)
        else:
            logits = u_values[:, None] * np.einsum("nf,nfc->nc", features, weights)
            penalties = self.regularization * np.einsum(
                "nd,nd->n", theta_values, theta_values
            )
        losses = self._logsumexp(logits) - logits[np.arange(len(labels)), labels]
        return losses + penalties

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        features = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return np.argmax(features @ self._shared_weights(theta), axis=1)

    def accuracy(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        return float(
            np.mean(self.predict(theta, x) == np.asarray(y, dtype=np.int64))
        )


def build_objective(
    name: str,
    regularization: float,
    *,
    feature_dimension: int | None = None,
    n_classes: int | None = None,
) -> Objective:
    if name == "paper_sigmoid":
        return PaperSigmoidObjective(regularization)
    if name == "lt_admm_nonconvex_logistic":
        return LTADMMNonconvexLogisticObjective(regularization)
    if name == "multiclass_softmax":
        if feature_dimension is None or n_classes is None:
            raise ValueError(
                "multiclass_softmax requires feature_dimension and n_classes"
            )
        return MulticlassSoftmaxObjective(
            regularization, feature_dimension, n_classes
        )
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
