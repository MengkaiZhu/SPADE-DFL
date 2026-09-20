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


def filter_binary_digits(
    x: np.ndarray,
    y: np.ndarray,
    negative_digit: int = 6,
    positive_digit: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (y == negative_digit) | (y == positive_digit)
    selected_y = np.where(y[mask] == positive_digit, 1.0, -1.0)
    return x[mask], selected_y.astype(np.float64)


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


def load_mnist_binary(
    path: Path,
    dimension: int = 10,
    download_url: Optional[str] = None,
) -> DatasetBundle:
    path = Path(path)
    if not path.exists():
        if not download_url:
            raise FileNotFoundError(f"MNIST file not found: {path}")
        download_file(download_url, path)
    with np.load(path) as dataset:
        train_x, train_y = filter_binary_digits(dataset["x_train"], dataset["y_train"])
        test_x, test_y = filter_binary_digits(dataset["x_test"], dataset["y_test"])
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
    )

