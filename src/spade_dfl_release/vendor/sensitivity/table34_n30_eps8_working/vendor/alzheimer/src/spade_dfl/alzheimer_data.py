"""Strict, leak-free preprocessing for the Alzheimer disease CSV dataset."""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .data import DatasetBundle, PCAState

REQUIRED_COLUMNS = (
    "PatientID", "Age", "Gender", "Ethnicity", "EducationLevel", "BMI",
    "Smoking", "AlcoholConsumption", "PhysicalActivity", "DietQuality",
    "SleepQuality", "FamilyHistoryAlzheimers", "CardiovascularDisease",
    "Diabetes", "Depression", "HeadInjury", "Hypertension", "SystolicBP",
    "DiastolicBP", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL",
    "CholesterolTriglycerides", "MMSE", "FunctionalAssessment",
    "MemoryComplaints", "BehavioralProblems", "ADL", "Confusion",
    "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
    "Forgetfulness", "Diagnosis", "DoctorInCharge",
)
CONTINUOUS_COLUMNS = (
    "Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality",
    "SleepQuality", "SystolicBP", "DiastolicBP", "CholesterolTotal",
    "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", "MMSE",
    "FunctionalAssessment", "ADL",
)
BINARY_COLUMNS = (
    "Gender", "Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease",
    "Diabetes", "Depression", "HeadInjury", "Hypertension", "MemoryComplaints",
    "BehavioralProblems", "Confusion", "Disorientation", "PersonalityChanges",
    "DifficultyCompletingTasks", "Forgetfulness",
)
CATEGORICAL_COLUMNS = ("Ethnicity", "EducationLevel")
CATEGORY_LEVELS = ("0", "1", "2", "3")
PREPROCESSING_NAME = "standardize_plus_one_hot_no_pca"


@dataclass(frozen=True)
class AlzheimerPreprocessorState:
    continuous_mean: np.ndarray
    continuous_scale: np.ndarray
    continuous_feature_names: tuple[str, ...]
    binary_feature_names: tuple[str, ...]
    category_levels: tuple[str, ...]
    encoded_feature_names: tuple[str, ...]


@dataclass(frozen=True)
class AlzheimerSplit:
    bundle: DatasetBundle
    preprocessor: AlzheimerPreprocessorState
    model_x: np.ndarray
    model_y: np.ndarray
    validation_x: Optional[np.ndarray]  # noqa: UP045  (Python 3.9 support)
    validation_y: Optional[np.ndarray]  # noqa: UP045  (Python 3.9 support)
    test_x: np.ndarray
    test_y: np.ndarray
    raw_model_continuous: np.ndarray
    model_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    metadata: dict[str, str]


@dataclass(frozen=True)
class InnerValidationSplit:
    """Inner train/validation data with no outer-test fields or arrays."""

    bundle: DatasetBundle
    preprocessor: AlzheimerPreprocessorState
    model_x: np.ndarray
    model_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    raw_model_continuous: np.ndarray
    model_indices: np.ndarray
    validation_indices: np.ndarray
    metadata: dict[str, str]


def _zip_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_csv_member(raw: bytes, member_name: str) -> list[dict[str, str]]:
    try:
        reader = csv.DictReader(io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", newline=""))
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError(
                f"CSV header in {member_name!r} must contain exactly the required 35 columns in order"
            )
        rows = list(reader)
    except UnicodeDecodeError as error:
        raise ValueError(f"CSV member {member_name!r} is not UTF-8") from error
    if not rows:
        raise ValueError("Alzheimer CSV must contain at least one data row")
    for row_number, row in enumerate(rows, start=2):
        if set(row) != set(REQUIRED_COLUMNS) or any(row[column] is None for column in REQUIRED_COLUMNS):
            raise ValueError(f"row {row_number} does not match the required CSV schema")
        blank = [column for column in REQUIRED_COLUMNS if not row[column].strip()]
        if blank:
            raise ValueError(f"row {row_number} has blank value(s): {', '.join(blank)}")
    return rows


def read_alzheimer_zip(path: Path) -> tuple[list[dict[str, str]], str]:
    """Read the sole schema-conforming CSV from ``path`` and hash ZIP bytes."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Alzheimer ZIP file not found: {path}")
    digest = _zip_sha256(path)
    try:
        with zipfile.ZipFile(path) as archive:
            candidates: list[tuple[str, list[dict[str, str]]]] = []
            header_errors: list[ValueError] = []
            for info in archive.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".csv"):
                    continue
                try:
                    candidates.append((info.filename, _parse_csv_member(archive.read(info), info.filename)))
                except ValueError as error:
                    header_errors.append(error)
            if len(candidates) == 1:
                rows = candidates[0][1]
            elif len(candidates) > 1:
                raise ValueError("ZIP contains more than one CSV with the required header")
            elif header_errors:
                raise header_errors[0]
            else:
                raise ValueError("ZIP does not contain a CSV file")
    except zipfile.BadZipFile as error:
        raise ValueError(f"Alzheimer dataset is not a valid ZIP file: {path}") from error
    _validate_rows(rows)
    return rows, digest


def _validate_rows(rows: list[dict[str, str]]) -> None:
    doctors = {row["DoctorInCharge"] for row in rows}
    if len(doctors) != 1:
        raise ValueError("DoctorInCharge must be constant across all rows")
    for row_number, row in enumerate(rows, start=2):
        for column in CONTINUOUS_COLUMNS:
            try:
                value = float(row[column])
            except ValueError as error:
                raise ValueError(f"row {row_number} {column} must be numeric") from error
            if not np.isfinite(value):
                raise ValueError(f"row {row_number} {column} must be finite")
        for column in (*BINARY_COLUMNS, "Diagnosis"):
            if row[column] not in {"0", "1"}:
                raise ValueError(f"row {row_number} {column} must be binary 0 or 1")
        for column in CATEGORICAL_COLUMNS:
            if row[column] not in CATEGORY_LEVELS:
                raise ValueError(
                    f"row {row_number} {column} must be one of {', '.join(CATEGORY_LEVELS)}"
                )


def _stratified_held_out_indices(
    labels: np.ndarray, indices: np.ndarray, split_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    generator = np.random.default_rng(split_seed)
    train_parts = []
    held_out_parts = []
    for label in (-1.0, 1.0):
        class_indices = indices[labels[indices] == label]
        if not len(class_indices):
            raise ValueError("each split source must contain both Diagnosis classes")
        shuffled = generator.permutation(class_indices)
        held_out_count = int(np.floor(len(shuffled) * 0.2 + 0.5))
        if held_out_count == 0 or held_out_count == len(shuffled):
            raise ValueError("each Diagnosis class needs enough rows for an 80/20 split")
        held_out_parts.append(shuffled[:held_out_count])
        train_parts.append(shuffled[held_out_count:])
    train = np.concatenate(train_parts)
    held_out = np.concatenate(held_out_parts)
    return generator.permutation(train), generator.permutation(held_out)


def _rows_to_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    continuous = np.asarray(
        [[float(row[column]) for column in CONTINUOUS_COLUMNS] for row in rows],
        dtype=np.float64,
    )
    binary = np.asarray(
        [[float(row[column]) for column in BINARY_COLUMNS] for row in rows],
        dtype=np.float64,
    )
    categories = np.asarray(
        [[row[column] for column in CATEGORICAL_COLUMNS] for row in rows], dtype="U1"
    )
    labels = np.asarray(
        [1.0 if row["Diagnosis"] == "1" else -1.0 for row in rows], dtype=np.float64
    )
    return continuous, binary, categories, labels


def _fit_preprocessor(continuous: np.ndarray) -> AlzheimerPreprocessorState:
    mean = continuous.mean(axis=0)
    scale = continuous.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    encoded_names = (
        *CONTINUOUS_COLUMNS,
        *BINARY_COLUMNS,
        *(f"Ethnicity_{level}" for level in CATEGORY_LEVELS),
        *(f"EducationLevel_{level}" for level in CATEGORY_LEVELS),
        "intercept",
    )
    return AlzheimerPreprocessorState(
        continuous_mean=mean,
        continuous_scale=scale,
        continuous_feature_names=CONTINUOUS_COLUMNS,
        binary_feature_names=BINARY_COLUMNS,
        category_levels=CATEGORY_LEVELS,
        encoded_feature_names=encoded_names,
    )


def _transform(
    continuous: np.ndarray,
    binary: np.ndarray,
    categories: np.ndarray,
    state: AlzheimerPreprocessorState,
) -> np.ndarray:
    unknown = [
        f"{CATEGORICAL_COLUMNS[column]}={value}"
        for column in range(categories.shape[1])
        for value in np.unique(categories[:, column])
        if value not in state.category_levels
    ]
    if unknown:
        raise ValueError("unknown categorical level(s): " + ", ".join(unknown))
    one_hot = np.concatenate(
        [
            (categories[:, column : column + 1] == np.asarray(state.category_levels)).astype(np.float64)
            for column in range(len(CATEGORICAL_COLUMNS))
        ],
        axis=1,
    )
    transformed = np.concatenate(
        [
            (continuous - state.continuous_mean) / state.continuous_scale,
            binary,
            one_hot,
            np.ones((len(continuous), 1), dtype=np.float64),
        ],
        axis=1,
    )
    if transformed.shape[1] != 39:
        raise RuntimeError("Alzheimer feature encoding must have exactly 39 columns")
    return transformed


def _identity_pca() -> PCAState:
    return PCAState(
        mean=np.zeros(39, dtype=np.float64),
        components=np.eye(39, dtype=np.float64),
        scale=np.ones(39, dtype=np.float64),
    )


def _make_split(path: Path, split_seed: int, with_validation: bool) -> AlzheimerSplit:
    rows, dataset_sha256 = read_alzheimer_zip(path)
    continuous, binary, categories, labels = _rows_to_arrays(rows)
    all_indices = np.arange(len(rows), dtype=np.int64)
    outer_model, test_indices = _stratified_held_out_indices(labels, all_indices, split_seed)
    validation_indices = np.empty(0, dtype=np.int64)
    model_indices = outer_model
    if with_validation:
        model_indices, validation_indices = _stratified_held_out_indices(
            labels, outer_model, split_seed
        )
    state = _fit_preprocessor(continuous[model_indices])
    model_x = _transform(continuous[model_indices], binary[model_indices], categories[model_indices], state)
    test_x = _transform(continuous[test_indices], binary[test_indices], categories[test_indices], state)
    validation_x = (
        _transform(continuous[validation_indices], binary[validation_indices], categories[validation_indices], state)
        if with_validation
        else None
    )
    validation_y = labels[validation_indices] if with_validation else None
    bundle_test_x = validation_x if with_validation else test_x
    bundle_test_y = validation_y if with_validation else labels[test_indices]
    bundle = DatasetBundle(
        train_x=model_x,
        train_y=labels[model_indices],
        test_x=bundle_test_x,
        test_y=bundle_test_y,
        pca=_identity_pca(),
        source_path=str(Path(path).resolve()),
        sha256=dataset_sha256,
        dataset_name="alzheimer",
        negative_class=0,
        positive_class=1,
        negative_class_name="Diagnosis 0",
        positive_class_name="Diagnosis 1",
        original_train_count=len(rows),
        original_test_count=len(test_indices),
        filtered_train_count=len(model_indices),
        filtered_test_count=len(bundle_test_y),
        original_feature_dimension=39,
        pca_dimension=39,
        task_type="binary",
        n_classes=2,
        class_names=("Diagnosis 0", "Diagnosis 1"),
        model_dimension=39,
    )
    return AlzheimerSplit(
        bundle=bundle,
        preprocessor=state,
        model_x=model_x,
        model_y=labels[model_indices],
        validation_x=validation_x,
        validation_y=validation_y,
        test_x=test_x,
        test_y=labels[test_indices],
        raw_model_continuous=continuous[model_indices],
        model_indices=model_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        metadata={"preprocessing": PREPROCESSING_NAME},
    )


def build_final_split(path: Path, split_seed: int = 1) -> AlzheimerSplit:
    """Build an outer 80/20 train/test split, fitting on the outer model subset."""
    return _make_split(path, split_seed, with_validation=False)


def build_stratified_alzheimer_fold(
    path: Path,
    *,
    n_folds: int,
    fold_index: int,
    split_seed: int = 1,
) -> AlzheimerSplit:
    """Build one leak-free stratified fold over the complete Alzheimer cohort."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    if fold_index < 0 or fold_index >= n_folds:
        raise ValueError("fold_index must lie in [0, n_folds)")
    rows, dataset_sha256 = read_alzheimer_zip(path)
    continuous, binary, categories, labels = _rows_to_arrays(rows)
    generator = np.random.default_rng(split_seed)
    class_folds: dict[float, list[np.ndarray]] = {}
    for label in (-1.0, 1.0):
        indices = np.flatnonzero(labels == label).astype(np.int64)
        if len(indices) < n_folds:
            raise ValueError("each Diagnosis class must cover every fold")
        class_folds[label] = [
            np.sort(part.astype(np.int64))
            for part in np.array_split(generator.permutation(indices), n_folds)
        ]
    test_indices = np.sort(
        np.concatenate(
            [class_folds[label][fold_index] for label in (-1.0, 1.0)]
        )
    )
    all_indices = np.arange(len(rows), dtype=np.int64)
    model_indices = np.setdiff1d(
        all_indices, test_indices, assume_unique=True
    )
    state = _fit_preprocessor(continuous[model_indices])
    model_x = _transform(
        continuous[model_indices],
        binary[model_indices],
        categories[model_indices],
        state,
    )
    test_x = _transform(
        continuous[test_indices],
        binary[test_indices],
        categories[test_indices],
        state,
    )
    bundle = DatasetBundle(
        train_x=model_x,
        train_y=labels[model_indices],
        test_x=test_x,
        test_y=labels[test_indices],
        pca=_identity_pca(),
        source_path=str(Path(path).resolve()),
        sha256=dataset_sha256,
        dataset_name="alzheimer",
        negative_class=0,
        positive_class=1,
        negative_class_name="Diagnosis 0",
        positive_class_name="Diagnosis 1",
        original_train_count=len(rows),
        original_test_count=len(test_indices),
        filtered_train_count=len(model_indices),
        filtered_test_count=len(test_indices),
        original_feature_dimension=39,
        pca_dimension=39,
        task_type="binary",
        n_classes=2,
        class_names=("Diagnosis 0", "Diagnosis 1"),
        model_dimension=39,
    )
    return AlzheimerSplit(
        bundle=bundle,
        preprocessor=state,
        model_x=model_x,
        model_y=labels[model_indices],
        validation_x=None,
        validation_y=None,
        test_x=test_x,
        test_y=labels[test_indices],
        raw_model_continuous=continuous[model_indices],
        model_indices=model_indices,
        validation_indices=np.empty(0, dtype=np.int64),
        test_indices=test_indices,
        metadata={
            "preprocessing": PREPROCESSING_NAME,
            "split_scope": "stratified_cross_validation",
            "n_folds": str(n_folds),
            "fold_index": str(fold_index),
            "split_seed": str(split_seed),
        },
    )


def build_validation_split(path: Path, split_seed: int = 1) -> AlzheimerSplit:
    """Build outer 80/20 plus inner train/validation splits without leakage."""
    return _make_split(path, split_seed, with_validation=True)


def build_inner_validation_split(
    path: Path, split_seed: int = 1
) -> InnerValidationSplit:
    """Build only the deterministic 64/16 inner split, never outer-test arrays.

    The first deterministic stratified split supplies the original protocol's
    80% model population; its held-out indices are deliberately discarded
    before preprocessing. Only the model population is then split and
    transformed into the 64% inner-train and 16% inner-validation arrays.
    """
    rows, dataset_sha256 = read_alzheimer_zip(path)
    continuous, binary, categories, labels = _rows_to_arrays(rows)
    all_indices = np.arange(len(rows), dtype=np.int64)
    outer_model_indices, _discarded_outer_indices = _stratified_held_out_indices(
        labels, all_indices, split_seed
    )
    model_indices, validation_indices = _stratified_held_out_indices(
        labels, outer_model_indices, split_seed
    )
    state = _fit_preprocessor(continuous[model_indices])
    model_x = _transform(
        continuous[model_indices], binary[model_indices], categories[model_indices], state
    )
    validation_x = _transform(
        continuous[validation_indices],
        binary[validation_indices],
        categories[validation_indices],
        state,
    )
    validation_y = labels[validation_indices]
    bundle = DatasetBundle(
        train_x=model_x,
        train_y=labels[model_indices],
        test_x=validation_x,
        test_y=validation_y,
        pca=_identity_pca(),
        source_path="inner_validation_only",
        sha256=dataset_sha256,
        dataset_name="alzheimer",
        negative_class=0,
        positive_class=1,
        negative_class_name="Diagnosis 0",
        positive_class_name="Diagnosis 1",
        original_train_count=len(outer_model_indices),
        original_test_count=0,
        filtered_train_count=len(model_indices),
        filtered_test_count=len(validation_y),
        original_feature_dimension=39,
        pca_dimension=39,
        task_type="binary",
        n_classes=2,
        class_names=("Diagnosis 0", "Diagnosis 1"),
        model_dimension=39,
    )
    return InnerValidationSplit(
        bundle=bundle,
        preprocessor=state,
        model_x=model_x,
        model_y=labels[model_indices],
        validation_x=validation_x,
        validation_y=validation_y,
        raw_model_continuous=continuous[model_indices],
        model_indices=model_indices,
        validation_indices=validation_indices,
        metadata={
            "preprocessing": PREPROCESSING_NAME,
            "split_scope": "inner_validation_only",
        },
    )


def save_alzheimer_preprocessor(state: AlzheimerPreprocessorState, path: Path) -> Path:
    """Save the tabular preprocessing state separately from DatasetBundle's PCA shim."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            continuous_mean=state.continuous_mean,
            continuous_scale=state.continuous_scale,
            continuous_feature_names=np.asarray(state.continuous_feature_names),
            binary_feature_names=np.asarray(state.binary_feature_names),
            category_levels=np.asarray(state.category_levels),
            encoded_feature_names=np.asarray(state.encoded_feature_names),
            preprocessing=np.asarray(PREPROCESSING_NAME),
        )
    temporary.replace(path)
    return path
