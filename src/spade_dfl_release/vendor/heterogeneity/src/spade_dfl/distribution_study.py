"""Read-only analysis of saved SPADE-DFL distribution-study artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


_HISTORY_POINTS = 51
_AGGREGATE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class DistributionStudy:
    """Raw, artifact-derived distribution-study measurements.

    The 51 points are round 0 through round 50.  ``None`` in
    ``rounds_to_threshold`` is a right-censored (not reached) observation.
    """

    partitions: tuple[str, ...]
    seeds: tuple[int, ...]
    accuracy_mean: Mapping[str, np.ndarray]
    accuracy_std: Mapping[str, np.ndarray]
    accuracy_by_seed: Mapping[str, np.ndarray]
    label_proportions: Mapping[str, np.ndarray]
    label_entropy: Mapping[str, np.ndarray]
    rounds_to_threshold: Mapping[str, tuple[int | None, ...]]
    threshold: float


def binary_label_entropy(proportions: np.ndarray) -> np.ndarray:
    """Return base-2 binary entropy for observed positive-label fractions."""

    values = np.asarray(proportions, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("label proportions must be finite values in [0, 1]")
    entropy = np.zeros_like(values, dtype=np.float64)
    interior = (values > 0.0) & (values < 1.0)
    p = values[interior]
    entropy[interior] = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return entropy


def first_reach_round(accuracy: np.ndarray, threshold: float) -> int | None:
    """Return the first saved round at or above ``threshold``; otherwise ``None``."""

    curve = np.asarray(accuracy, dtype=np.float64)
    if curve.ndim != 1 or curve.size == 0 or not np.all(np.isfinite(curve)):
        raise ValueError("accuracy must be a nonempty finite one-dimensional array")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    reached = np.flatnonzero(curve >= threshold)
    return int(reached[0]) if reached.size else None


def load_distribution_study(
    result_root: Path,
    partitions: Sequence[str],
    threshold: float = 0.95,
    *,
    method: str = "spade_dfl",
    n_clients: int = 31,
) -> DistributionStudy:
    """Load a 50-round SPADE-only study from its already saved artifacts.

    This function never reruns a partition or training run.  It validates the
    aggregate NPZ against the raw per-seed CSV curves before returning data.
    """

    root = Path(result_root).resolve()
    if not root.is_dir():
        raise ValueError(f"result root is not a directory: {root}")
    requested = tuple(partitions)
    if not requested or any(not isinstance(item, str) or not item for item in requested):
        raise ValueError("partitions must be a nonempty sequence of names")
    if len(set(requested)) != len(requested):
        raise ValueError("partitions must not contain duplicates")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not isinstance(method, str) or not method:
        raise ValueError("method must be a nonempty string")
    if not isinstance(n_clients, int) or isinstance(n_clients, bool) or n_clients <= 0:
        raise ValueError("n_clients must be a positive integer")

    manifest = _read_json(root / "suite_manifest.json", "suite manifest")
    seeds = _manifest_seeds(manifest)
    run_records = manifest.get("runs")
    if not isinstance(run_records, list):
        raise ValueError("suite manifest runs must be a list")

    mean_by_partition: dict[str, np.ndarray] = {}
    std_by_partition: dict[str, np.ndarray] = {}
    curves_by_partition: dict[str, np.ndarray] = {}
    proportions_by_partition: dict[str, np.ndarray] = {}
    entropy_by_partition: dict[str, np.ndarray] = {}
    rounds_by_partition: dict[str, tuple[int | None, ...]] = {}

    for partition in requested:
        selected = [
            record
            for record in run_records
            if isinstance(record, dict) and record.get("partition") == partition
        ]
        for record in selected:
            _validate_run_seed(record.get("seed"), partition)
        unexpected = [record for record in selected if int(record["seed"]) not in seeds]
        if unexpected:
            raise ValueError(f"inconsistent seeds for partition {partition}")

        curves: list[np.ndarray] = []
        client_proportions: list[np.ndarray] = []
        for seed in seeds:
            matching = [record for record in selected if int(record["seed"]) == seed]
            if not matching:
                raise ValueError(f"missing run identity for {partition}/seed_{seed}")
            if len(matching) > 1:
                raise ValueError(f"duplicate run identity for {partition}/seed_{seed}")
            record = matching[0]
            if record.get("method") != method:
                raise ValueError(
                    f"run {partition}/seed_{seed} must use method {method}, "
                    f"not {record.get('method')!r}"
                )
            run_dir = record.get("run_dir")
            if not isinstance(run_dir, str) or not run_dir:
                raise ValueError(f"missing run_dir for {partition}/seed_{seed}")
            path = _resolve_run_dir(root, run_dir, partition, seed)
            curves.append(_read_accuracy_history(path / "history.csv"))
            client_proportions.append(
                _read_label_proportions(path / "environment.json", n_clients=n_clients)
            )

        raw = np.vstack(curves)
        aggregate_path = root / "aggregates" / partition / f"{method}.npz"
        aggregate_mean, aggregate_std = _read_accuracy_aggregate(aggregate_path)
        expected_mean = raw.mean(axis=0)
        expected_std = raw.std(axis=0, ddof=0)
        if not np.allclose(aggregate_mean, expected_mean, rtol=0.0, atol=_AGGREGATE_TOLERANCE):
            raise ValueError(f"aggregate accuracy_mean disagrees with raw histories for {partition}")
        if not np.allclose(aggregate_std, expected_std, rtol=0.0, atol=_AGGREGATE_TOLERANCE):
            raise ValueError(f"aggregate accuracy_std disagrees with raw histories for {partition}")

        values = np.concatenate(client_proportions)
        mean_by_partition[partition] = _freeze(aggregate_mean)
        std_by_partition[partition] = _freeze(aggregate_std)
        curves_by_partition[partition] = _freeze(raw)
        proportions_by_partition[partition] = _freeze(values)
        entropy_by_partition[partition] = _freeze(binary_label_entropy(values))
        rounds_by_partition[partition] = tuple(
            first_reach_round(curve, threshold) for curve in raw
        )

    return DistributionStudy(
        partitions=requested,
        seeds=seeds,
        accuracy_mean=MappingProxyType(mean_by_partition),
        accuracy_std=MappingProxyType(std_by_partition),
        accuracy_by_seed=MappingProxyType(curves_by_partition),
        label_proportions=MappingProxyType(proportions_by_partition),
        label_entropy=MappingProxyType(entropy_by_partition),
        rounds_to_threshold=MappingProxyType(rounds_by_partition),
        threshold=float(threshold),
    )


def _manifest_seeds(manifest: Mapping[str, object]) -> tuple[int, ...]:
    config = manifest.get("suite_config")
    if not isinstance(config, dict) or not isinstance(config.get("seeds"), list):
        raise ValueError("suite manifest must contain suite_config.seeds")
    raw_seeds = tuple(config["seeds"])
    if not raw_seeds or any(not _is_integer_seed(seed) for seed in raw_seeds):
        raise ValueError("manifest seeds must be nonempty unique integers")
    seeds = tuple(int(seed) for seed in raw_seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("manifest seeds must be unique integers")
    return seeds


def _is_integer_seed(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _validate_run_seed(value: object, partition: str) -> None:
    if not _is_integer_seed(value):
        raise ValueError(
            f"run seed for partition {partition} must be a non-boolean built-in or NumPy integer"
        )


def _resolve_run_dir(root: Path, run_dir: str, partition: str, seed: int) -> Path:
    """Resolve a manifest location into the current result root only."""

    declared = Path(run_dir)
    suffix = _runs_suffix(declared)
    if not declared.is_absolute():
        declared_candidate = (root / declared).resolve()
        if not _is_within(declared_candidate, root):
            raise ValueError(
                f"run_dir for {partition}/seed_{seed} resolves outside result root: {run_dir}"
            )
    runs_root = (root / "runs").resolve()
    candidate = (root / suffix).resolve()
    if not _is_within(candidate, runs_root):
        raise ValueError(
            f"run_dir for {partition}/seed_{seed} resolves outside result root/runs: {run_dir}"
        )
    if not candidate.is_dir():
        raise ValueError(f"missing run directory for {partition}/seed_{seed}: {candidate}")
    return candidate


def _runs_suffix(path: Path) -> Path:
    for index, part in enumerate(path.parts):
        if part == "runs":
            return Path(*path.parts[index:])
    raise ValueError(f"run_dir must contain a runs anchor: {path}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_accuracy_history(path: Path) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"missing history.csv: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "accuracy" not in rows[0]:
        raise ValueError(f"history.csv lacks an accuracy column: {path}")
    try:
        accuracy = np.asarray([float(row["accuracy"]) for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"history.csv has malformed accuracy values: {path}") from exc
    _validate_accuracy(accuracy, f"history.csv {path}")
    return accuracy


def _read_accuracy_aggregate(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"missing aggregate NPZ: {path}")
    try:
        with np.load(path) as saved:
            mean = np.asarray(saved["accuracy_mean"], dtype=np.float64)
            std = np.asarray(saved["accuracy_std"], dtype=np.float64)
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"malformed aggregate NPZ: {path}") from exc
    _validate_accuracy(mean, f"aggregate accuracy_mean {path}")
    _validate_accuracy(std, f"aggregate accuracy_std {path}")
    if np.any(std < 0.0):
        raise ValueError(f"aggregate accuracy_std must be nonnegative: {path}")
    return mean, std


def _validate_accuracy(values: np.ndarray, name: str) -> None:
    if values.ndim != 1 or values.size != _HISTORY_POINTS:
        raise ValueError(f"{name} must contain exactly 51 accuracy points for 50 rounds")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite accuracy in {name}")


def _read_label_proportions(path: Path, *, n_clients: int) -> np.ndarray:
    environment = _read_json(path, "environment.json")
    dataset = environment.get("dataset")
    if not isinstance(dataset, dict) or "label_histograms" not in dataset:
        raise ValueError(f"environment.json lacks dataset.label_histograms: {path}")
    try:
        histograms = np.asarray(dataset["label_histograms"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed label_histograms: {path}") from exc
    expected_shape = (n_clients, 2)
    if histograms.shape != expected_shape:
        raise ValueError(
            f"label_histograms must have shape ({n_clients}, 2): {path}"
        )
    if not np.all(np.isfinite(histograms)) or np.any(histograms < 0.0):
        raise ValueError(f"label_histograms must be finite nonnegative counts: {path}")
    totals = histograms.sum(axis=1)
    if np.any(totals <= 0.0):
        raise ValueError(f"label_histograms rows must have positive sums: {path}")
    return histograms[:, 1] / totals


def _read_json(path: Path, description: str) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing {description}: {path}")
    try:
        contents = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed {description}: {path}") from exc
    if not isinstance(contents, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return contents


def _freeze(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result
