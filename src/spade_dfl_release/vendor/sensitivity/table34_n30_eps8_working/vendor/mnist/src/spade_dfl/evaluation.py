from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .data import DatasetBundle
from .metrics import consensus_error, penalty_energy
from .objectives import Objective


COMMON_HISTORY_FIELDS = (
    "loss",
    "accuracy",
    "train_accuracy",
    "generalization_error",
    "gradient_norm",
    "stationarity",
    "consensus_error",
    "stationarity_consensus",
    "penalty_energy",
)


def new_history(*extra_fields: str) -> Dict[str, List[float]]:
    fields = (*COMMON_HISTORY_FIELDS, *extra_fields)
    if len(set(fields)) != len(fields):
        raise ValueError("history fields must be unique")
    return {name: [] for name in fields}


def record_metrics(
    history: Dict[str, List[float]],
    states: np.ndarray,
    objective: Objective,
    dataset: DatasetBundle,
    penalties: Optional[np.ndarray] = None,
) -> None:
    values = np.asarray(states, dtype=np.float64)
    average = values.mean(axis=0)
    train_accuracy = objective.accuracy(average, dataset.train_x, dataset.train_y)
    test_accuracy = objective.accuracy(average, dataset.test_x, dataset.test_y)
    gradient = objective.gradient(average, dataset.train_x, dataset.train_y, u=1.0)
    gradient_norm = float(np.linalg.norm(gradient))
    consensus = consensus_error(values)
    history["loss"].append(
        objective.value(average, dataset.train_x, dataset.train_y, u=1.0)
    )
    history["accuracy"].append(test_accuracy)
    history["train_accuracy"].append(train_accuracy)
    history["generalization_error"].append(train_accuracy - test_accuracy)
    history["gradient_norm"].append(gradient_norm)
    history["stationarity"].append(gradient_norm**2)
    history["consensus_error"].append(consensus)
    history["stationarity_consensus"].append(gradient_norm**2 + consensus)
    history["penalty_energy"].append(
        penalty_energy(penalties) if penalties is not None else 0.0
    )
