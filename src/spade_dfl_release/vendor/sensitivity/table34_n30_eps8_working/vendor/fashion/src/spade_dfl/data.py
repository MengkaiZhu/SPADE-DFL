from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlopen

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
    dataset_name: str = "mnist"
    negative_class: int = 6
    positive_class: int = 7
    negative_class_name: str = "digit 6"
    positive_class_name: str = "digit 7"
    original_train_count: int = 0
    original_test_count: int = 0
    filtered_train_count: int = 0
    filtered_test_count: int = 0
    original_feature_dimension: int = 0
    pca_dimension: int = 0


def filter_binary_classes(
    x: np.ndarray,
    y: np.ndarray,
    negative_class: int,
    positive_class: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if negative_class == positive_class:
        raise ValueError("negative_class and positive_class must be different")
    mask = (y == negative_class) | (y == positive_class)
    selected_y = np.where(y[mask] == positive_class, 1.0, -1.0)
    return x[mask], selected_y.astype(np.float64)


def filter_binary_digits(
    x: np.ndarray,
    y: np.ndarray,
    negative_digit: int = 6,
    positive_digit: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    return filter_binary_classes(x, y, negative_digit, positive_digit)


def fit_pca_train_only(train_x: np.ndarray, dimension: int) -> PCAState:
    x = np.asarray(train_x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("PCA input must be a two-dimensional matrix")
    if dimension <= 0 or dimension > min(x.shape):
        raise ValueError("dimension must be between 1 and min(n_samples, n_features)")
    mean = x.mean(axis=0)
    centered = x - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:dimension].copy()
    scores = centered @ components.T
    scale = scores.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return PCAState(mean=mean, components=components, scale=scale)


def apply_pca(x: np.ndarray, pca: PCAState) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    return ((values - pca.mean) @ pca.components.T) / pca.scale


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
    temporary.replace(path)


def load_binary_image_dataset(
    path: Path,
    dimension: int = 10,
    negative_class: int = 6,
    positive_class: int = 7,
    download_url: Optional[str] = None,
    dataset_name: str = "mnist",
    negative_class_name: str = "digit 6",
    positive_class_name: str = "digit 7",
) -> DatasetBundle:
    path = Path(path)
    if not path.exists():
        if not download_url:
            raise FileNotFoundError(f"MNIST file not found: {path}")
        download_file(download_url, path)
    with np.load(path) as dataset:
        original_train_count = int(dataset["x_train"].shape[0])
        original_test_count = int(dataset["x_test"].shape[0])
        original_feature_dimension = int(
            np.prod(dataset["x_train"].shape[1:], dtype=np.int64)
        )
        train_x, train_y = filter_binary_classes(
            dataset["x_train"], dataset["y_train"], negative_class, positive_class
        )
        test_x, test_y = filter_binary_classes(
            dataset["x_test"], dataset["y_test"], negative_class, positive_class
        )
    train_flat = train_x.reshape(train_x.shape[0], -1).astype(np.float64) / 255.0
    test_flat = test_x.reshape(test_x.shape[0], -1).astype(np.float64) / 255.0
    pca = fit_pca_train_only(train_flat, dimension)
    return DatasetBundle(
        train_x=apply_pca(train_flat, pca),
        train_y=train_y,
        test_x=apply_pca(test_flat, pca),
        test_y=test_y,
        pca=pca,
        source_path=str(path.resolve()),
        sha256=_sha256(path),
        dataset_name=dataset_name,
        negative_class=negative_class,
        positive_class=positive_class,
        negative_class_name=negative_class_name,
        positive_class_name=positive_class_name,
        original_train_count=original_train_count,
        original_test_count=original_test_count,
        filtered_train_count=int(train_y.shape[0]),
        filtered_test_count=int(test_y.shape[0]),
        original_feature_dimension=original_feature_dimension,
        pca_dimension=int(dimension),
    )


def load_mnist_binary(
    path: Path,
    dimension: int = 10,
    download_url: Optional[str] = None,
) -> DatasetBundle:
    return load_binary_image_dataset(
        path,
        dimension=dimension,
        negative_class=6,
        positive_class=7,
        download_url=download_url,
        dataset_name="mnist",
        negative_class_name="digit 6",
        positive_class_name="digit 7",
    )
