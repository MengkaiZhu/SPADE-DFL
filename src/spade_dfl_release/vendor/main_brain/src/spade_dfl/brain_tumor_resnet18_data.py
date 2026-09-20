from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile

import numpy as np
from PIL import Image

from .brain_tumor_data import (
    BRAIN_TUMOR_CLASS_NAMES,
    PCA_OVERSAMPLES,
    PCA_POWER_ITERATIONS,
    PCA_SEED,
)
from .data import DatasetBundle, PCAState


SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class ImageRecord:
    archive_name: str
    class_name: str
    label: int


@dataclass(frozen=True)
class TrainingHoldout:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    fraction: float
    seed: int


@dataclass(frozen=True)
class FeatureMatrix:
    features: np.ndarray
    labels: np.ndarray
    records: tuple[ImageRecord, ...]
    dataset_sha256: str
    sample_order_sha256: str
    extractor_name: str
    source_path: str
    split: str


@dataclass(frozen=True)
class FeatureHoldoutBundle:
    bundle: DatasetBundle
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    fraction: float
    seed: int


class ResNet18V1Extractor:
    name = "torchvision_resnet18_imagenet1k_v1_avgpool_512d"

    def __init__(self, *, device: str = "cpu", batch_size: int = 32):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        try:
            import torch
            import torchvision
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as error:
            raise RuntimeError(
                "ResNet18 extraction requires the locked torch and torchvision dependencies"
            ) from error
        self._torch = torch
        self._torchvision = torchvision
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.weights = ResNet18_Weights.IMAGENET1K_V1
        self.weight_name = self.weights.name
        self.transform = self.weights.transforms()
        self.model = resnet18(weights=self.weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.to(self.device)

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, 512), dtype=np.float32)
        tensor = self._torch.stack([self.transform(image) for image in images])
        tensor = tensor.to(self.device)
        with self._torch.inference_mode():
            features = self.model(tensor)
        return features.detach().cpu().numpy()

    @property
    def weight_cache_path(self) -> Path:
        filename = Path(urlparse(self.weights.url).path).name
        return Path(self._torch.hub.get_dir()) / "checkpoints" / filename

    @property
    def weight_sha256(self) -> str:
        if not self.weight_cache_path.exists():
            raise FileNotFoundError(
                f"official ResNet18 weight cache not found: {self.weight_cache_path}"
            )
        return _sha256(self.weight_cache_path)

    def audit_metadata(self) -> dict[str, object]:
        return {
            "weight_name": self.weight_name,
            "weight_url": self.weights.url,
            "weight_cache_path": str(self.weight_cache_path),
            "weight_sha256": self.weight_sha256,
            "transform_repr": repr(self.transform),
            "torch_version": self._torch.__version__,
            "torchvision_version": self._torchvision.__version__,
            "device": self.device,
            "dtype": "float32",
            "batch_size": self.batch_size,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_split_records(path: Path, split: str) -> tuple[ImageRecord, ...]:
    if split not in {"Training", "Testing"}:
        raise ValueError("split must be exactly Training or Testing")
    dataset_path = Path(path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Brain Tumor MRI ZIP not found: {dataset_path}")
    records: list[ImageRecord] = []
    with ZipFile(dataset_path) as archive:
        archive_names = archive.namelist()
        for label, class_name in enumerate(BRAIN_TUMOR_CLASS_NAMES):
            prefix = f"{split}/{class_name}/"
            names = sorted(
                name
                for name in archive_names
                if name.startswith(prefix)
                and name.lower().endswith(SUPPORTED_IMAGE_SUFFIXES)
            )
            if not names:
                raise ValueError(f"no images found for {split}/{class_name}")
            records.extend(
                ImageRecord(
                    archive_name=name,
                    class_name=class_name,
                    label=label,
                )
                for name in names
            )
    return tuple(records)


def ordered_training_records(path: Path) -> tuple[ImageRecord, ...]:
    return ordered_split_records(path, "Training")


def build_training_holdout(
    labels: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> TrainingHoldout:
    if not 0.0 < fraction < 0.5:
        raise ValueError("holdout fraction must lie in (0, 0.5)")
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    required_labels = np.arange(len(BRAIN_TUMOR_CLASS_NAMES), dtype=np.int64)
    if not np.array_equal(np.unique(values), required_labels):
        raise ValueError("labels must contain exactly the four required classes")
    generator = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    for label in np.unique(values):
        indices = generator.permutation(np.flatnonzero(values == label))
        validation_count = max(1, int(round(indices.size * fraction)))
        if validation_count >= indices.size:
            raise ValueError("each class must retain at least one training sample")
        validation.extend(int(index) for index in indices[:validation_count])
        train.extend(int(index) for index in indices[validation_count:])
    return TrainingHoldout(
        train_indices=tuple(sorted(train)),
        validation_indices=tuple(sorted(validation)),
        fraction=float(fraction),
        seed=int(seed),
    )


def fit_training_pca(features: np.ndarray, dimension: int) -> PCAState:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("PCA input must be two-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("PCA input must contain only finite values")
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


def apply_training_pca(features: np.ndarray, pca: PCAState) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("PCA input must be two-dimensional")
    if values.shape[1] != pca.mean.shape[0]:
        raise ValueError("PCA input feature dimension does not match fitted state")
    if not np.isfinite(values).all():
        raise ValueError("PCA input must contain only finite values")
    return (values - pca.mean) @ pca.components.T / pca.scale


def extract_split_features(
    path: Path,
    split: str,
    extractor: object,
) -> FeatureMatrix:
    dataset_path = Path(path).resolve()
    records = ordered_split_records(dataset_path, split)
    batch_size = int(getattr(extractor, "batch_size", 32))
    if batch_size <= 0:
        raise ValueError("extractor batch_size must be positive")
    batches: list[np.ndarray] = []
    with ZipFile(dataset_path) as archive:
        for start in range(0, len(records), batch_size):
            images: list[Image.Image] = []
            for record in records[start : start + batch_size]:
                with Image.open(BytesIO(archive.read(record.archive_name))) as image:
                    images.append(image.convert("L").convert("RGB"))
            output = np.asarray(extractor(images), dtype=np.float64)
            if output.shape != (len(images), 512):
                raise ValueError(
                    "feature extractor must return one 512-dimensional row per image"
                )
            if not np.isfinite(output).all():
                raise ValueError("feature extractor output must contain only finite values")
            batches.append(output)
    features = np.concatenate(batches, axis=0)
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    order_payload = "\n".join(record.archive_name for record in records).encode("utf-8")
    return FeatureMatrix(
        features=features,
        labels=labels,
        records=records,
        dataset_sha256=_sha256(dataset_path),
        sample_order_sha256=hashlib.sha256(order_payload).hexdigest(),
        extractor_name=str(getattr(extractor, "name", type(extractor).__name__)),
        source_path=str(dataset_path),
        split=split,
    )


def extract_training_features(path: Path, extractor: object) -> FeatureMatrix:
    return extract_split_features(path, "Training", extractor)


def build_holdout_bundle(
    matrix: FeatureMatrix,
    *,
    fraction: float,
    seed: int,
    dimension: int = 10,
) -> FeatureHoldoutBundle:
    if matrix.split != "Training":
        raise ValueError("Training holdout requires a Training feature matrix")
    holdout = build_training_holdout(matrix.labels, fraction=fraction, seed=seed)
    train_indices = np.asarray(holdout.train_indices, dtype=np.int64)
    validation_indices = np.asarray(holdout.validation_indices, dtype=np.int64)
    pca = fit_training_pca(matrix.features[train_indices], dimension=dimension)
    train_x = apply_training_pca(matrix.features[train_indices], pca)
    validation_x = apply_training_pca(matrix.features[validation_indices], pca)
    bundle = DatasetBundle(
        train_x=train_x,
        train_y=matrix.labels[train_indices].copy(),
        test_x=validation_x,
        test_y=matrix.labels[validation_indices].copy(),
        pca=pca,
        source_path=f"{matrix.source_path}::resnet18_v1_training_holdout",
        sha256=matrix.dataset_sha256,
        dataset_name="brain_tumor_mri",
        original_train_count=int(matrix.labels.size),
        original_test_count=0,
        filtered_train_count=int(train_indices.size),
        filtered_test_count=int(validation_indices.size),
        original_feature_dimension=int(matrix.features.shape[1]),
        pca_dimension=int(dimension),
        task_type="multiclass",
        n_classes=len(BRAIN_TUMOR_CLASS_NAMES),
        class_names=BRAIN_TUMOR_CLASS_NAMES,
        model_dimension=int(dimension * len(BRAIN_TUMOR_CLASS_NAMES)),
    )
    return FeatureHoldoutBundle(
        bundle=bundle,
        train_indices=holdout.train_indices,
        validation_indices=holdout.validation_indices,
        fraction=holdout.fraction,
        seed=holdout.seed,
    )


def build_full_dataset_bundle(
    training: FeatureMatrix,
    testing: FeatureMatrix,
    *,
    dimension: int = 10,
) -> DatasetBundle:
    if training.split != "Training" or testing.split != "Testing":
        raise ValueError("full dataset requires separate Training and Testing matrices")
    if training.dataset_sha256 != testing.dataset_sha256:
        raise ValueError("Training and Testing dataset SHA-256 values do not match")
    if training.extractor_name != testing.extractor_name:
        raise ValueError("Training and Testing feature extractors do not match")
    if training.features.shape[1] != testing.features.shape[1]:
        raise ValueError("Training and Testing feature dimensions do not match")
    pca = fit_training_pca(training.features, dimension=dimension)
    train_x = apply_training_pca(training.features, pca)
    test_x = apply_training_pca(testing.features, pca)
    return DatasetBundle(
        train_x=train_x,
        train_y=training.labels.copy(),
        test_x=test_x,
        test_y=testing.labels.copy(),
        pca=pca,
        source_path=f"{training.source_path}::resnet18_v1_training_fit_testing_eval",
        sha256=training.dataset_sha256,
        dataset_name="brain_tumor_mri",
        original_train_count=int(training.labels.size),
        original_test_count=int(testing.labels.size),
        filtered_train_count=int(training.labels.size),
        filtered_test_count=int(testing.labels.size),
        original_feature_dimension=int(training.features.shape[1]),
        pca_dimension=int(dimension),
        task_type="multiclass",
        n_classes=len(BRAIN_TUMOR_CLASS_NAMES),
        class_names=BRAIN_TUMOR_CLASS_NAMES,
        model_dimension=int(dimension * len(BRAIN_TUMOR_CLASS_NAMES)),
    )


def write_feature_cache(
    path: Path,
    matrix: FeatureMatrix,
    metadata: dict[str, object],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=matrix.features,
        labels=matrix.labels,
        archive_names=np.asarray(
            [record.archive_name for record in matrix.records],
            dtype=np.str_,
        ),
        class_names=np.asarray(
            [record.class_name for record in matrix.records],
            dtype=np.str_,
        ),
        dataset_sha256=np.asarray(matrix.dataset_sha256),
        sample_order_sha256=np.asarray(matrix.sample_order_sha256),
        extractor_name=np.asarray(matrix.extractor_name),
        source_path=np.asarray(matrix.source_path),
        split=np.asarray(matrix.split),
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True, ensure_ascii=False)
        ),
    )


def load_feature_cache(path: Path) -> tuple[FeatureMatrix, dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as cache:
        features = np.asarray(cache["features"], dtype=np.float64)
        labels = np.asarray(cache["labels"], dtype=np.int64)
        archive_names = tuple(str(value) for value in cache["archive_names"])
        class_names = tuple(str(value) for value in cache["class_names"])
        dataset_sha256 = str(cache["dataset_sha256"].item())
        sample_order_sha256 = str(cache["sample_order_sha256"].item())
        extractor_name = str(cache["extractor_name"].item())
        source_path = str(cache["source_path"].item())
        metadata = json.loads(str(cache["metadata_json"].item()))
        split = (
            str(cache["split"].item())
            if "split" in cache.files
            else str(metadata.get("split", ""))
        )
    if features.ndim != 2 or features.shape[1] != 512:
        raise ValueError("feature cache must contain a two-dimensional 512d matrix")
    if labels.shape != (features.shape[0],):
        raise ValueError("feature cache labels do not match feature rows")
    if len(archive_names) != features.shape[0] or len(class_names) != features.shape[0]:
        raise ValueError("feature cache records do not match feature rows")
    if not np.isfinite(features).all():
        raise ValueError("feature cache must contain only finite values")
    records = tuple(
        ImageRecord(name, class_name, int(label))
        for name, class_name, label in zip(archive_names, class_names, labels)
    )
    order_payload = "\n".join(archive_names).encode("utf-8")
    if hashlib.sha256(order_payload).hexdigest() != sample_order_sha256:
        raise ValueError("feature cache sample order hash does not match records")
    return (
        FeatureMatrix(
            features=features,
            labels=labels,
            records=records,
            dataset_sha256=dataset_sha256,
            sample_order_sha256=sample_order_sha256,
            extractor_name=extractor_name,
            source_path=source_path,
            split=split,
        ),
        metadata,
    )
