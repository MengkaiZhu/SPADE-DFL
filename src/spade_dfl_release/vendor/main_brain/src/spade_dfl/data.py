from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAState:
    mean: np.ndarray
    components: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class DatasetBundle:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    pca: PCAState
    source_path: str
    sha256: str
    dataset_name: str
    original_train_count: int
    original_test_count: int
    filtered_train_count: int
    filtered_test_count: int
    original_feature_dimension: int
    pca_dimension: int
    task_type: str
    n_classes: int
    class_names: tuple[str, ...]
    model_dimension: int

