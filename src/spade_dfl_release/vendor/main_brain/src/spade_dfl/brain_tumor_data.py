from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from .data import DatasetBundle, PCAState


BRAIN_TUMOR_CLASS_NAMES = (
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
)
PCA_SEED = 20260724
PCA_OVERSAMPLES = 8
PCA_POWER_ITERATIONS = 2


@dataclass(frozen=True)
class TrainingHoldoutBundle:
    bundle: DatasetBundle
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    fraction: float
    seed: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resize_with_square_padding(image: Image.Image, size: int = 28) -> np.ndarray:
    if size <= 0:
        raise ValueError("image size must be positive")
    grayscale = image.convert("L")
    width, height = grayscale.size
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = grayscale.resize((resized_width, resized_height), resampling)
    canvas = Image.new("L", (size, size), color=0)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas.paste(resized, (left, top))
    return np.asarray(canvas, dtype=np.float64) / 255.0


def _fit_train_only_pca(
    train_images: np.ndarray,
    dimension: int,
) -> PCAState:
    values = np.asarray(train_images, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("PCA input must be two-dimensional")
    if dimension <= 0 or dimension > min(values.shape):
        raise ValueError("invalid PCA feature dimension")
    mean = values.mean(axis=0)
    centered = values - mean
    range_dimension = min(
        dimension + PCA_OVERSAMPLES,
        values.shape[0],
        values.shape[1],
    )
    generator = np.random.default_rng(PCA_SEED)
    omega = generator.standard_normal((values.shape[1], range_dimension))
    sample_range = centered @ omega
    for _ in range(PCA_POWER_ITERATIONS):
        sample_range = centered @ (centered.T @ sample_range)
    basis, _ = np.linalg.qr(sample_range, mode="reduced")
    projected = basis.T @ centered
    _, _, right_vectors = np.linalg.svd(projected, full_matrices=False)
    components = right_vectors[:dimension].copy()
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    scores = centered @ components.T
    scale = scores.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return PCAState(mean=mean, components=components, scale=scale)


def _apply_pca(images: np.ndarray, pca: PCAState) -> np.ndarray:
    return (np.asarray(images, dtype=np.float64) - pca.mean) @ pca.components.T / pca.scale


def _dataset_names(archive: ZipFile, split: str, class_name: str) -> list[str]:
    prefix = f"{split}/{class_name}/"
    supported = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted(
        name
        for name in archive.namelist()
        if name.startswith(prefix) and name.lower().endswith(supported)
    )


def _read_split(
    archive: ZipFile,
    split: str,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for label, class_name in enumerate(BRAIN_TUMOR_CLASS_NAMES):
        names = _dataset_names(archive, split, class_name)
        if not names:
            raise ValueError(f"no images found for {split}/{class_name}")
        for name in names:
            with Image.open(BytesIO(archive.read(name))) as image:
                images.append(resize_with_square_padding(image, image_size).reshape(-1))
            labels.append(label)
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def _load_cache(
    cache_path: Path,
    dataset_sha256: str,
    feature_dimension: int,
    image_size: int,
) -> DatasetBundle | None:
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache["dataset_sha256"].item()) != dataset_sha256:
                return None
            if int(cache["feature_dimension"].item()) != feature_dimension:
                return None
            if int(cache["image_size"].item()) != image_size:
                return None
            pca = PCAState(
                mean=cache["pca_mean"].copy(),
                components=cache["pca_components"].copy(),
                scale=cache["pca_scale"].copy(),
            )
            source_path = str(cache["source_path"].item())
            train_x = cache["train_x"].copy()
            train_y = cache["train_y"].copy()
            test_x = cache["test_x"].copy()
            test_y = cache["test_y"].copy()
    except (OSError, ValueError, KeyError):
        return None
    return _bundle(
        train_x,
        train_y,
        test_x,
        test_y,
        pca,
        source_path,
        dataset_sha256,
        image_size,
        feature_dimension,
    )


def _bundle(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    pca: PCAState,
    source_path: str,
    dataset_sha256: str,
    image_size: int,
    feature_dimension: int,
) -> DatasetBundle:
    return DatasetBundle(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        pca=pca,
        source_path=source_path,
        sha256=dataset_sha256,
        dataset_name="brain_tumor_mri",
        original_train_count=int(len(train_y)),
        original_test_count=int(len(test_y)),
        filtered_train_count=int(len(train_y)),
        filtered_test_count=int(len(test_y)),
        original_feature_dimension=int(image_size * image_size),
        pca_dimension=int(feature_dimension),
        task_type="multiclass",
        n_classes=4,
        class_names=BRAIN_TUMOR_CLASS_NAMES,
        model_dimension=int(feature_dimension * 4),
    )


def load_brain_tumor_zip(
    path: Path,
    feature_dimension: int = 10,
    image_size: int = 28,
    cache_path: Path | None = None,
) -> DatasetBundle:
    dataset_path = Path(path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Brain Tumor MRI ZIP not found: {dataset_path}")
    dataset_sha256 = _sha256(dataset_path)
    if cache_path is not None:
        cached = _load_cache(
            Path(cache_path),
            dataset_sha256,
            feature_dimension,
            image_size,
        )
        if cached is not None:
            return cached
    with ZipFile(dataset_path) as archive:
        train_images, train_y = _read_split(archive, "Training", image_size)
        test_images, test_y = _read_split(archive, "Testing", image_size)
    pca = _fit_train_only_pca(train_images, feature_dimension)
    train_x = _apply_pca(train_images, pca)
    test_x = _apply_pca(test_images, pca)
    result = _bundle(
        train_x,
        train_y,
        test_x,
        test_y,
        pca,
        str(dataset_path),
        dataset_sha256,
        image_size,
        feature_dimension,
    )
    if cache_path is not None:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            dataset_sha256=np.asarray(dataset_sha256),
            feature_dimension=np.asarray(feature_dimension),
            image_size=np.asarray(image_size),
            source_path=np.asarray(str(dataset_path)),
            pca_mean=pca.mean,
            pca_components=pca.components,
            pca_scale=pca.scale,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
        )
    return result


def load_brain_tumor_training_holdout_zip(
    path: Path,
    *,
    feature_dimension: int = 10,
    image_size: int = 28,
    fraction: float = 0.1,
    seed: int = 20260724,
) -> TrainingHoldoutBundle:
    """Create a tuning split before PCA so validation images cannot fit PCA."""
    if not 0.0 < fraction < 0.5:
        raise ValueError("holdout fraction must lie in (0, 0.5)")
    dataset_path = Path(path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Brain Tumor MRI ZIP not found: {dataset_path}")
    dataset_sha256 = _sha256(dataset_path)
    with ZipFile(dataset_path) as archive:
        train_images, train_y = _read_split(archive, "Training", image_size)
    generator = np.random.default_rng(seed)
    inner_train: list[int] = []
    validation: list[int] = []
    for label in np.unique(train_y):
        indices = generator.permutation(np.flatnonzero(train_y == label))
        count = max(1, int(round(indices.size * fraction)))
        validation.extend(int(index) for index in indices[:count])
        inner_train.extend(int(index) for index in indices[count:])
    train_indices = tuple(sorted(inner_train))
    validation_indices = tuple(sorted(validation))
    pca = _fit_train_only_pca(
        train_images[np.asarray(train_indices, dtype=np.int64)],
        feature_dimension,
    )
    train_x = _apply_pca(
        train_images[np.asarray(train_indices, dtype=np.int64)],
        pca,
    )
    validation_x = _apply_pca(
        train_images[np.asarray(validation_indices, dtype=np.int64)],
        pca,
    )
    bundle = _bundle(
        train_x,
        train_y[np.asarray(train_indices, dtype=np.int64)],
        validation_x,
        train_y[np.asarray(validation_indices, dtype=np.int64)],
        pca,
        f"{dataset_path}::inner_training_pca_holdout",
        dataset_sha256,
        image_size,
        feature_dimension,
    )
    return TrainingHoldoutBundle(
        bundle=bundle,
        train_indices=train_indices,
        validation_indices=validation_indices,
        fraction=float(fraction),
        seed=int(seed),
    )
