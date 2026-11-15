from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Optional, Sequence, Tuple
from urllib.request import urlopen

import numpy as np


RANDOMIZED_PCA_SOLVER = "halko_randomized_svd"
RANDOMIZED_PCA_SEED = 20260720
RANDOMIZED_PCA_OVERSAMPLES = 10
RANDOMIZED_PCA_POWER_ITERATIONS = 2


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
    task_type: str = "binary"
    n_classes: int = 2
    class_names: tuple = ()
    model_dimension: int = 0


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


def _fit_scaled_image_pca(
    flat_images: np.ndarray,
    dimension: int,
    *,
    chunk_size: int = 2048,
) -> PCAState:
    """Fit train-only PCA without materializing the full float64 image matrix."""
    values = np.asarray(flat_images)
    if values.ndim != 2:
        raise ValueError("PCA input must be a two-dimensional matrix")
    if dimension <= 0 or dimension > min(values.shape):
        raise ValueError("dimension must be between 1 and min(n_samples, n_features)")
    mean = values.mean(axis=0, dtype=np.float64) / 255.0
    covariance = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    for start in range(0, len(values), chunk_size):
        block = values[start : start + chunk_size].astype(np.float64) / 255.0
        block -= mean
        covariance += block.T @ block
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:dimension]
    components = eigenvectors[:, order].T.copy()
    score_squares = np.zeros(dimension, dtype=np.float64)
    for start in range(0, len(values), chunk_size):
        block = values[start : start + chunk_size].astype(np.float64) / 255.0
        scores = (block - mean) @ components.T
        score_squares += np.square(scores).sum(axis=0)
    scale = np.sqrt(score_squares / len(values))
    scale = np.where(scale > 1e-12, scale, 1.0)
    return PCAState(mean=mean, components=components, scale=scale)


def fit_randomized_scaled_image_pca(
    flat_images: np.ndarray,
    dimension: int,
    random_seed: int = RANDOMIZED_PCA_SEED,
    oversamples: int = RANDOMIZED_PCA_OVERSAMPLES,
    power_iterations: int = RANDOMIZED_PCA_POWER_ITERATIONS,
    chunk_size: int = 2048,
) -> PCAState:
    """Fit deterministic train-only image PCA with a randomized range finder.

    Image values are scaled by ``1/255``.  Only small pixel blocks are cast to
    float64, avoiding a full float64 copy of a large image matrix.
    """
    values = np.asarray(flat_images)
    if values.ndim != 2:
        raise ValueError("PCA input must be a two-dimensional matrix")
    n_samples, n_features = values.shape
    if dimension <= 0 or dimension > min(values.shape):
        raise ValueError("dimension must be between 1 and min(n_samples, n_features)")
    if oversamples < 0:
        raise ValueError("oversamples cannot be negative")
    if power_iterations < 0:
        raise ValueError("power_iterations cannot be negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    mean_sum = np.zeros(n_features, dtype=np.float64)
    for start in range(0, n_samples, chunk_size):
        block = values[start : start + chunk_size].astype(np.float64) / 255.0
        mean_sum += block.sum(axis=0)
    mean = mean_sum / n_samples

    range_dimension = min(dimension + oversamples, n_samples, n_features)
    generator = np.random.default_rng(random_seed)
    test_matrix = generator.standard_normal((n_features, range_dimension))
    sample_range = np.empty((n_samples, range_dimension), dtype=np.float64)
    for start in range(0, n_samples, chunk_size):
        block = values[start : start + chunk_size].astype(np.float64) / 255.0
        block -= mean
        sample_range[start : start + len(block)] = block @ test_matrix

    for _ in range(power_iterations):
        sample_basis, _ = np.linalg.qr(sample_range, mode="reduced")
        feature_range = np.zeros(
            (n_features, sample_basis.shape[1]), dtype=np.float64
        )
        for start in range(0, n_samples, chunk_size):
            block = values[start : start + chunk_size].astype(np.float64) / 255.0
            block -= mean
            feature_range += block.T @ sample_basis[start : start + len(block)]
        feature_basis, _ = np.linalg.qr(feature_range, mode="reduced")
        sample_range = np.empty(
            (n_samples, feature_basis.shape[1]), dtype=np.float64
        )
        for start in range(0, n_samples, chunk_size):
            block = values[start : start + chunk_size].astype(np.float64) / 255.0
            block -= mean
            sample_range[start : start + len(block)] = block @ feature_basis

    sample_basis, _ = np.linalg.qr(sample_range, mode="reduced")
    projected = np.zeros((sample_basis.shape[1], n_features), dtype=np.float64)
    for start in range(0, n_samples, chunk_size):
        block = values[start : start + chunk_size].astype(np.float64) / 255.0
        block -= mean
        projected += sample_basis[start : start + len(block)].T @ block
    _, _, right_vectors = np.linalg.svd(projected, full_matrices=False)
    components = right_vectors[:dimension].copy()

    score_squares = np.zeros(dimension, dtype=np.float64)
    for start in range(0, n_samples, chunk_size):
        block = values[start : start + chunk_size].astype(np.float64) / 255.0
        scores = (block - mean) @ components.T
        score_squares += np.square(scores).sum(axis=0)
    scale = np.sqrt(score_squares / n_samples)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return PCAState(mean=mean, components=components, scale=scale)


def _cache_scalar(cache: np.lib.npyio.NpzFile, name: str):
    value = np.asarray(cache[name])
    if value.ndim != 0:
        raise ValueError(f"PCA cache identity field {name!r} must be scalar")
    return value.item()


def _load_pca_cache(
    path: Path,
    *,
    dataset_sha256: str,
    feature_dimension: int,
    original_feature_dimension: int,
    solver_name: str,
    random_seed: int,
    oversamples: int,
    power_iterations: int,
) -> PCAState:
    required = {
        "dataset_sha256",
        "feature_dimension",
        "solver_name",
        "random_seed",
        "oversamples",
        "power_iterations",
        "mean_shape",
        "components_shape",
        "scale_shape",
        "mean",
        "components",
        "scale",
    }
    try:
        with np.load(path, allow_pickle=False) as cache:
            missing = required.difference(cache.files)
            if missing:
                raise ValueError(
                    "PCA cache identity is missing fields: " + ", ".join(sorted(missing))
                )
            expected_scalars = {
                "dataset_sha256": dataset_sha256,
                "feature_dimension": feature_dimension,
                "solver_name": solver_name,
                "random_seed": random_seed,
                "oversamples": oversamples,
                "power_iterations": power_iterations,
            }
            mismatched = [
                name
                for name, expected in expected_scalars.items()
                if _cache_scalar(cache, name) != expected
            ]
            expected_shapes = {
                "mean_shape": (original_feature_dimension,),
                "components_shape": (
                    feature_dimension,
                    original_feature_dimension,
                ),
                "scale_shape": (feature_dimension,),
            }
            arrays = {
                "mean": np.asarray(cache["mean"], dtype=np.float64),
                "components": np.asarray(cache["components"], dtype=np.float64),
                "scale": np.asarray(cache["scale"], dtype=np.float64),
            }
            for shape_name, expected_shape in expected_shapes.items():
                declared_shape = tuple(
                    int(item) for item in np.asarray(cache[shape_name]).tolist()
                )
                array_name = shape_name.removesuffix("_shape")
                if declared_shape != expected_shape or arrays[array_name].shape != expected_shape:
                    mismatched.append(shape_name)
            if mismatched:
                raise ValueError(
                    "PCA cache identity mismatch: " + ", ".join(sorted(set(mismatched)))
                )
            if not all(np.all(np.isfinite(array)) for array in arrays.values()):
                raise ValueError("PCA cache identity mismatch: non-finite PCA arrays")
            if np.any(arrays["scale"] <= 0.0):
                raise ValueError("PCA cache identity mismatch: non-positive scale")
            return PCAState(**arrays)
    except (OSError, EOFError) as error:
        raise ValueError(f"PCA cache identity could not be read: {path}") from error


def _save_pca_cache(
    path: Path,
    pca: PCAState,
    *,
    dataset_sha256: str,
    feature_dimension: int,
    solver_name: str,
    random_seed: int,
    oversamples: int,
    power_iterations: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            dataset_sha256=np.asarray(dataset_sha256),
            feature_dimension=np.asarray(feature_dimension, dtype=np.int64),
            solver_name=np.asarray(solver_name),
            random_seed=np.asarray(random_seed, dtype=np.int64),
            oversamples=np.asarray(oversamples, dtype=np.int64),
            power_iterations=np.asarray(power_iterations, dtype=np.int64),
            mean_shape=np.asarray(pca.mean.shape, dtype=np.int64),
            components_shape=np.asarray(pca.components.shape, dtype=np.int64),
            scale_shape=np.asarray(pca.scale.shape, dtype=np.int64),
            mean=pca.mean,
            components=pca.components,
            scale=pca.scale,
        )
    temporary.replace(path)


def _apply_scaled_image_pca(
    flat_images: np.ndarray,
    pca: PCAState,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    values = np.asarray(flat_images)
    transformed = np.empty((len(values), len(pca.scale)), dtype=np.float64)
    for start in range(0, len(values), chunk_size):
        block = values[start : start + chunk_size].astype(np.float64) / 255.0
        transformed[start : start + len(block)] = (
            (block - pca.mean) @ pca.components.T
        ) / pca.scale
    return transformed


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
        task_type="binary",
        n_classes=2,
        class_names=(negative_class_name, positive_class_name),
        model_dimension=int(dimension),
    )


def load_multiclass_image_dataset(
    path: Path,
    feature_dimension: int,
    n_classes: int,
    dataset_name: str,
    download_url: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
    preprocessing_cache: Optional[Path] = None,
) -> DatasetBundle:
    path = Path(path)
    if not path.exists():
        if not download_url:
            raise FileNotFoundError(f"multiclass image dataset not found: {path}")
        download_file(download_url, path)
    dataset_sha256 = _sha256(path)
    with np.load(path) as dataset:
        train_images = np.asarray(dataset["x_train"])
        test_images = np.asarray(dataset["x_test"])
        train_y = np.asarray(dataset["y_train"], dtype=np.int64)
        test_y = np.asarray(dataset["y_test"], dtype=np.int64)
    if train_y.ndim != 1 or test_y.ndim != 1:
        raise ValueError("multiclass labels must be one-dimensional")
    expected_labels = np.arange(n_classes, dtype=np.int64)
    if not np.array_equal(np.unique(train_y), expected_labels):
        raise ValueError(f"training labels must cover integers 0..{n_classes - 1}")
    if np.any(test_y < 0) or np.any(test_y >= n_classes):
        raise ValueError(f"test labels must lie in 0..{n_classes - 1}")
    if len(train_images) != len(train_y) or len(test_images) != len(test_y):
        raise ValueError("image and label counts do not match")
    original_feature_dimension = int(
        np.prod(train_images.shape[1:], dtype=np.int64)
    )
    train_flat = train_images.reshape(len(train_images), -1)
    test_flat = test_images.reshape(len(test_images), -1)
    if test_flat.shape[1] != original_feature_dimension:
        raise ValueError("training and test images must have the same feature shape")
    resolved_class_names = (
        tuple(f"class {index}" for index in range(n_classes))
        if class_names is None
        else tuple(str(name) for name in class_names)
    )
    if len(resolved_class_names) != n_classes or any(
        not name.strip() for name in resolved_class_names
    ):
        raise ValueError("class_names must contain one nonempty name per class")
    cache_path = Path(preprocessing_cache) if preprocessing_cache is not None else None
    if cache_path is not None and cache_path.exists():
        pca = _load_pca_cache(
            cache_path,
            dataset_sha256=dataset_sha256,
            feature_dimension=feature_dimension,
            original_feature_dimension=original_feature_dimension,
            solver_name=RANDOMIZED_PCA_SOLVER,
            random_seed=RANDOMIZED_PCA_SEED,
            oversamples=RANDOMIZED_PCA_OVERSAMPLES,
            power_iterations=RANDOMIZED_PCA_POWER_ITERATIONS,
        )
    else:
        pca = fit_randomized_scaled_image_pca(
            train_flat,
            feature_dimension,
            random_seed=RANDOMIZED_PCA_SEED,
            oversamples=RANDOMIZED_PCA_OVERSAMPLES,
            power_iterations=RANDOMIZED_PCA_POWER_ITERATIONS,
        )
        if cache_path is not None:
            _save_pca_cache(
                cache_path,
                pca,
                dataset_sha256=dataset_sha256,
                feature_dimension=feature_dimension,
                solver_name=RANDOMIZED_PCA_SOLVER,
                random_seed=RANDOMIZED_PCA_SEED,
                oversamples=RANDOMIZED_PCA_OVERSAMPLES,
                power_iterations=RANDOMIZED_PCA_POWER_ITERATIONS,
            )
    return DatasetBundle(
        train_x=_apply_scaled_image_pca(train_flat, pca),
        train_y=train_y,
        test_x=_apply_scaled_image_pca(test_flat, pca),
        test_y=test_y,
        pca=pca,
        source_path=str(path.resolve()),
        sha256=dataset_sha256,
        dataset_name=dataset_name,
        original_train_count=int(len(train_y)),
        original_test_count=int(len(test_y)),
        filtered_train_count=int(len(train_y)),
        filtered_test_count=int(len(test_y)),
        original_feature_dimension=original_feature_dimension,
        pca_dimension=int(feature_dimension),
        task_type="multiclass",
        n_classes=int(n_classes),
        class_names=resolved_class_names,
        model_dimension=int(feature_dimension * n_classes),
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
