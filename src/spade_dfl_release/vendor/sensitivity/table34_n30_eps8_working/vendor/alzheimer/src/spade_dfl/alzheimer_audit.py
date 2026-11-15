"""Read-only integrity checks and evidence notes for Alzheimer suite results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .alzheimer_contract import ALZHEIMER_METHOD_IDS, validate_alzheimer_method_contract
from .alzheimer_frozen_config import load_frozen_alzheimer_suite_config
from .alzheimer_suite import (
    ALZHEIMER_PARTITIONS,
    DISPLAY_LABELS,
    _apply_v2_final_method_parameters,
    _partition_parameters,
)
from .alzheimer_v2_frozen import load_v2_suite_config
from .artifacts import load_run
from .methods import method_config

_SEEDS = (0, 1, 2)
_ROUNDS = 50
_POINTS = _ROUNDS + 1
_TOLERANCE = 1e-12
_DEPRECATED_LABEL_PARTS = ("private", "non-private", "nonprivate")
_V2_PROTOCOL_VERSION = "alzheimer_v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("JSON object required")
    return loaded


def _append_error(errors: list[str], message: str) -> None:
    if message not in errors:
        errors.append(message)


def _is_deprecated_label(value: object) -> bool:
    rendered = str(value).lower()
    return any(part in rendered for part in _DEPRECATED_LABEL_PARTS)


def _finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _array_error(actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.shape != expected.shape:
        return float("inf")
    if not np.all(np.isfinite(actual)):
        return float("inf")
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.0


def _json_safe(value: object) -> object:
    """Replace diagnostic infinities with JSON null without masking the error."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _audit_suite_manifest(root: Path, errors: list[str]) -> dict[str, Any] | None:
    path = root / "suite_manifest.json"
    try:
        manifest = _read_json(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        _append_error(errors, f"suite_manifest invalid: {error}")
        return None
    expected = {
        "status": "completed",
        "methods": list(ALZHEIMER_METHOD_IDS),
        "partitions": list(ALZHEIMER_PARTITIONS),
        "seeds": list(_SEEDS),
        "rounds": _ROUNDS,
        "protocol_role": "scientific_local",
        "evaluation_scope": "outer_final_test_frozen",
        "eligible_for_final_plots": True,
    }
    is_v2 = manifest.get("protocol_version") == _V2_PROTOCOL_VERSION
    if is_v2:
        expected["protocol_version"] = _V2_PROTOCOL_VERSION
    for key, value in expected.items():
        if manifest.get(key) != value:
            _append_error(errors, f"suite_manifest {key} does not meet local contract")
    try:
        validate_alzheimer_method_contract(manifest.get("methods", ()))
    except (TypeError, ValueError) as error:
        _append_error(errors, f"suite_manifest method contract invalid: {error}")
    if _is_deprecated_label(manifest.get("methods", ())):
        _append_error(errors, "suite_manifest contains deprecated SPADE label")
    config_path = Path(str(manifest.get("config_path", "")))
    if not config_path.is_file():
        _append_error(errors, "frozen config referenced by suite_manifest is missing")
    else:
        if manifest.get("config_sha256") != _sha256(config_path):
            _append_error(errors, "suite_manifest frozen config SHA-256 mismatch")
        try:
            frozen = (
                load_v2_suite_config(config_path)
                if is_v2
                else load_frozen_alzheimer_suite_config(config_path)
            )
            if (
                tuple(frozen.methods),
                tuple(frozen.partitions),
                tuple(frozen.experiment.seeds),
                frozen.experiment.n_rounds,
            ) != (ALZHEIMER_METHOD_IDS, ALZHEIMER_PARTITIONS, _SEEDS, _ROUNDS):
                _append_error(errors, "frozen config does not meet local protocol")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            _append_error(errors, f"frozen config cannot be validated: {error}")
    return manifest


def _audit_raw_directory_set(root: Path, errors: list[str]) -> None:
    raw_root = root / "raw"
    expected = {
        Path(partition) / method / f"seed_{seed}"
        for partition in ALZHEIMER_PARTITIONS
        for method in ALZHEIMER_METHOD_IDS
        for seed in _SEEDS
    }
    actual: set[Path] = set()
    if not raw_root.is_dir():
        _append_error(errors, "raw directory is missing")
        return
    for partition_dir in raw_root.iterdir():
        if not partition_dir.is_dir():
            _append_error(errors, "raw directory contains a non-directory entry")
            continue
        for method_dir in partition_dir.iterdir():
            if not method_dir.is_dir():
                _append_error(errors, "raw method directory contains a non-directory entry")
                continue
            for seed_dir in method_dir.iterdir():
                if seed_dir.is_dir():
                    actual.add(seed_dir.relative_to(raw_root))
                else:
                    _append_error(errors, "raw seed directory contains a non-directory entry")
    if actual != expected:
        _append_error(errors, "raw directory set is not the exact canonical 84-run set")
    for path in raw_root.rglob("*"):
        if _is_deprecated_label(path.name):
            _append_error(errors, f"deprecated SPADE label in raw path {path}")
    for summary_path in raw_root.rglob("summary.json"):
        try:
            summary = _read_json(summary_path)
            metadata = summary.get("metadata", {})
            labels = (
                summary.get("method"),
                summary.get("method_id"),
                metadata.get("stage") if isinstance(metadata, Mapping) else None,
            )
            if any(_is_deprecated_label(label) for label in labels if label is not None):
                _append_error(errors, f"deprecated SPADE label in raw metadata {summary_path}")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            _append_error(errors, f"raw metadata invalid {summary_path}: {error}")


def _expected_v2_raw_config(
    suite_path: Path, *, method: str, partition: str
) -> dict[str, object]:
    """Rebuild the exact method/partition config admitted by the v2 lock."""
    suite = load_v2_suite_config(suite_path)
    mode, alpha = _partition_parameters(partition)
    base = replace(suite.experiment, seeds=tuple(int(seed) for seed in suite.seeds))
    configured = method_config(base, method, base.seeds, mode, alpha)
    return _apply_v2_final_method_parameters(
        configured, suite=suite, method=method
    ).to_dict()


def _audit_v2_raw_config(
    *,
    config_path: Path,
    raw_manifest: Mapping[str, object],
    suite_path: Path,
    method: str,
    partition: str,
    errors: list[str],
) -> None:
    """Fail closed on a raw config hash or protocol-contract mismatch."""
    actual_sha = _sha256(config_path)
    if raw_manifest.get("config_sha256") != actual_sha:
        _append_error(errors, f"v2 raw config SHA-256 mismatch: {config_path}")
    expected = _expected_v2_raw_config(
        suite_path, method=method, partition=partition
    )
    actual = _read_json(config_path)
    for field in sorted((set(actual) | set(expected)) - {"data_path", "output_dir"}):
        if actual.get(field) != expected.get(field):
            _append_error(
                errors,
                f"v2 raw config differs from frozen method config: {config_path} field={field}",
            )


def _raw_expected_aggregates(
    root: Path,
    errors: list[str],
    dataset_sha256: str | None,
    *,
    is_v2: bool,
    v2_suite_path: Path | None,
) -> tuple[dict[tuple[str, str], dict[str, np.ndarray]], int, int]:
    aggregates: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    run_count = 0
    query_count = 0
    _audit_raw_directory_set(root, errors)
    for partition in ALZHEIMER_PARTITIONS:
        for method in ALZHEIMER_METHOD_IDS:
            histories: list[Mapping[str, np.ndarray]] = []
            for seed in _SEEDS:
                run_dir = root / "raw" / partition / method / f"seed_{seed}"
                try:
                    result = load_run(
                        run_dir,
                        expected_seed=seed,
                        expected_stage=method,
                        expected_history_length=_POINTS,
                        expected_dataset_sha256=dataset_sha256,
                        expected_partition=partition,
                    )
                    config = _read_json(run_dir / "config.json")
                    summary = _read_json(run_dir / "summary.json")
                    if is_v2:
                        raw_manifest = _read_json(run_dir / "raw_manifest.json")
                        if raw_manifest.get("protocol_version") != _V2_PROTOCOL_VERSION:
                            _append_error(errors, f"v2 raw manifest version mismatch: {run_dir}")
                        if v2_suite_path is None:
                            _append_error(errors, f"v2 suite path is missing for raw config: {run_dir}")
                        else:
                            _audit_v2_raw_config(
                                config_path=run_dir / "config.json",
                                raw_manifest=raw_manifest,
                                suite_path=v2_suite_path,
                                method=method,
                                partition=partition,
                                errors=errors,
                            )
                    if config.get("stage") != method:
                        _append_error(errors, f"raw config stage mismatch: {run_dir}")
                    if _is_deprecated_label(config.get("stage")) or _is_deprecated_label(
                        result.metadata.get("stage")
                    ):
                        _append_error(errors, f"deprecated SPADE label in {run_dir}")
                    if not isinstance(summary.get("query_count"), int) or summary[
                        "query_count"
                    ] < 0:
                        _append_error(errors, f"invalid query_count in {run_dir}")
                    else:
                        query_count += int(summary["query_count"])
                    histories.append(result.history)
                    run_count += 1
                except (
                    FileNotFoundError,
                    KeyError,
                    OSError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as error:
                    _append_error(errors, f"invalid raw run {run_dir}: {error}")
            if len(histories) != len(_SEEDS):
                continue
            names = tuple(histories[0])
            if any(tuple(history) != names for history in histories[1:]):
                _append_error(errors, f"inconsistent raw history columns for {partition}/{method}")
                continue
            expected: dict[str, np.ndarray] = {
                "round": np.arange(_POINTS, dtype=np.int64)
            }
            for name in names:
                values = np.vstack([np.asarray(history[name], dtype=np.float64) for history in histories])
                if values.shape != (len(_SEEDS), _POINTS) or not np.all(np.isfinite(values)):
                    _append_error(errors, f"nonfinite or wrong-length raw history for {partition}/{method}")
                    continue
                expected[f"{name}_mean"] = values.mean(axis=0)
                expected[f"{name}_std"] = values.std(axis=0, ddof=0)
            aggregates[(partition, method)] = expected
    if run_count != len(ALZHEIMER_PARTITIONS) * len(ALZHEIMER_METHOD_IDS) * len(_SEEDS):
        _append_error(errors, f"expected 84 raw runs, found {run_count}")
    return aggregates, run_count, query_count


def _audit_aggregate_and_figures(
    root: Path,
    suite_manifest: Mapping[str, object] | None,
    expected: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    errors: list[str],
    *,
    is_v2: bool,
) -> tuple[list[dict[str, object]], float]:
    rows: list[dict[str, object]] = []
    max_error = 0.0
    aggregate_path = root / "aggregates" / "aggregate_manifest.json"
    try:
        aggregate_manifest = _read_json(aggregate_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        _append_error(errors, f"aggregate manifest invalid: {error}")
        aggregate_manifest = {}
    suite_sha = _sha256(root / "suite_manifest.json") if suite_manifest else ""
    if aggregate_manifest.get("source_suite_manifest_sha256") != suite_sha:
        _append_error(errors, "aggregate manifest suite SHA-256 chain mismatch")
    if (
        aggregate_manifest.get("status") != "completed"
        or aggregate_manifest.get("protocol_role") != "scientific_local"
        or aggregate_manifest.get("eligible_for_final_plots") is not True
    ):
        _append_error(errors, "aggregate manifest protocol provenance mismatch")
    if is_v2 and aggregate_manifest.get("protocol_version") != _V2_PROTOCOL_VERSION:
        _append_error(errors, "aggregate manifest v2 version mismatch")
    if tuple(aggregate_manifest.get("partitions", {})) != ALZHEIMER_PARTITIONS:
        _append_error(errors, "aggregate manifest partitions are incomplete or unordered")
    for partition in ALZHEIMER_PARTITIONS:
        record = aggregate_manifest.get("partitions", {}).get(partition, {})
        if record.get("methods") != list(ALZHEIMER_METHOD_IDS) or record.get("seeds") != list(_SEEDS):
            _append_error(errors, f"aggregate manifest contract mismatch for {partition}")
        if set(record.get("artifacts", {})) != set(ALZHEIMER_METHOD_IDS):
            _append_error(errors, f"aggregate manifest artifacts incomplete for {partition}")
        for method in ALZHEIMER_METHOD_IDS:
            artifact = root / "aggregates" / partition / f"{method}.npz"
            expected_arrays = expected.get((partition, method))
            artifact_error = float("inf")
            try:
                with np.load(artifact, allow_pickle=False) as loaded:
                    actual = {name: np.asarray(loaded[name]) for name in loaded.files}
                if expected_arrays is None:
                    raise ValueError("raw aggregate could not be reconstructed")
                for name, wanted in expected_arrays.items():
                    if name not in actual:
                        artifact_error = float("inf")
                        break
                    artifact_error = max(artifact_error if math.isfinite(artifact_error) else 0.0, _array_error(actual[name], wanted))
                if set(actual) != set(expected_arrays):
                    artifact_error = float("inf")
            except (OSError, ValueError, KeyError) as error:
                _append_error(errors, f"aggregate artifact invalid {artifact}: {error}")
            manifest_artifact = record.get("artifacts", {}).get(method, {})
            if (
                manifest_artifact.get("path") != f"aggregates/{partition}/{method}.npz"
                or not artifact.is_file()
                or manifest_artifact.get("sha256") != _sha256(artifact)
            ):
                _append_error(errors, f"aggregate SHA-256 chain mismatch for {partition}/{method}")
            if artifact_error > _TOLERANCE:
                _append_error(errors, f"raw-to-aggregate mismatch for {partition}/{method}")
            max_error = max(max_error, artifact_error)
            rows.append(
                {
                    "partition": partition,
                    "method_id": method,
                    "artifact": artifact.relative_to(root).as_posix(),
                    "max_abs_error": artifact_error,
                    "accepted": artifact_error <= _TOLERANCE,
                }
            )
        source_error = _audit_figure_source(
            root, partition, expected, aggregate_path, suite_sha, errors, is_v2=is_v2
        )
        max_error = max(max_error, source_error)
    return rows, max_error


def _audit_figure_source(
    root: Path,
    partition: str,
    expected: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    aggregate_manifest_path: Path,
    suite_sha: str,
    errors: list[str],
    *,
    is_v2: bool,
) -> float:
    figure_dir = root / "figures" / partition
    source_path = figure_dir / "accuracy.source_data.csv"
    max_error = 0.0
    try:
        with source_path.open(newline="", encoding="utf-8-sig") as handle:
            source_rows = list(csv.DictReader(handle))
        if len(source_rows) != len(ALZHEIMER_METHOD_IDS) * _POINTS:
            raise ValueError("wrong source-data row count")
        for index, method in enumerate(ALZHEIMER_METHOD_IDS):
            wanted = expected[(partition, method)]
            for point in range(_POINTS):
                row = source_rows[index * _POINTS + point]
                if (
                    row.get("method_id") != method
                    or row.get("method") != DISPLAY_LABELS[method]
                    or int(row["round"]) != point
                ):
                    raise ValueError("source-data method or round contract mismatch")
                max_error = max(
                    max_error,
                    abs(float(row["mean"]) - float(wanted["accuracy_mean"][point])),
                    abs(float(row["std"]) - float(wanted["accuracy_std"][point])),
                )
    except (KeyError, OSError, TypeError, ValueError) as error:
        max_error = float("inf")
        _append_error(errors, f"figure source data invalid for {partition}: {error}")
    if max_error > _TOLERANCE:
        _append_error(errors, f"raw-to-figure-source mismatch for {partition}")
    try:
        figure_manifest = _read_json(figure_dir / "figure_manifest.json")
        if (
            figure_manifest.get("source_suite_manifest_sha256") != suite_sha
            or figure_manifest.get("source_aggregate_manifest_sha256")
            != _sha256(aggregate_manifest_path)
            or figure_manifest.get("partition") != partition
            or figure_manifest.get("protocol_role") != "scientific_local"
            or figure_manifest.get("eligible_for_final_plots") is not True
        ):
            _append_error(errors, f"figure SHA-256 chain mismatch for {partition}")
        if is_v2 and figure_manifest.get("protocol_version") != _V2_PROTOCOL_VERSION:
            _append_error(errors, f"figure manifest v2 version mismatch for {partition}")
        expected_artifacts = {
            "accuracy.png",
            "accuracy.pdf",
            "accuracy.svg",
            "accuracy.source_data.csv",
        }
        if set(figure_manifest.get("artifacts", {})) != expected_artifacts:
            _append_error(errors, f"figure manifest artifacts incomplete for {partition}")
        for name, digest in figure_manifest.get("artifacts", {}).items():
            path = figure_dir / name
            if not path.is_file() or _sha256(path) != digest:
                _append_error(errors, f"figure artifact SHA-256 mismatch for {partition}/{name}")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        _append_error(errors, f"figure manifest invalid for {partition}: {error}")
    return max_error


def _audit_final_metrics(
    root: Path,
    expected: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    errors: list[str],
    *,
    is_v2: bool,
) -> tuple[list[dict[str, object]], float]:
    rankings: list[dict[str, object]] = []
    max_error = 0.0
    try:
        with (root / "final_metrics.csv").open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(ALZHEIMER_PARTITIONS) * len(ALZHEIMER_METHOD_IDS):
            raise ValueError("wrong final-metric row count")
        position = 0
        for partition in ALZHEIMER_PARTITIONS:
            values: list[tuple[str, float, float]] = []
            for method in ALZHEIMER_METHOD_IDS:
                row = rows[position]
                position += 1
                wanted = expected[(partition, method)]
                if (
                    row.get("partition") != partition
                    or row.get("method_id") != method
                    or row.get("method") != DISPLAY_LABELS[method]
                    or int(row["seeds"]) != len(_SEEDS)
                    or int(row["rounds"]) != _ROUNDS
                ):
                    raise ValueError("final_metrics contract mismatch")
                mean = float(row["final_accuracy_mean"])
                std = float(row["final_accuracy_std"])
                if not _finite_number(mean) or not _finite_number(std):
                    raise ValueError("final_metrics contains nonfinite value")
                max_error = max(
                    max_error,
                    abs(mean - float(wanted["accuracy_mean"][-1])),
                    abs(std - float(wanted["accuracy_std"][-1])),
                )
                values.append((method, mean, std))
            ordered = sorted(values, key=lambda item: (-item[1], ALZHEIMER_METHOD_IDS.index(item[0])))
            value_by_method = {method: (mean, std) for method, mean, std in values}
            spade_mean, spade_std = value_by_method["spade_dfl"]
            ranking: dict[str, object] = {
                "partition": partition,
                "spade_rank": next(
                    index for index, (method, _, _) in enumerate(ordered, start=1) if method == "spade_dfl"
                ),
                "spade_final_accuracy": spade_mean,
                "best_method_id": ordered[0][0],
                "best_final_accuracy": ordered[0][1],
                "gap_to_best": ordered[0][1] - spade_mean,
            }
            if is_v2:
                stationarity = expected[(partition, "spade_dfl")]
                stationarity_start = float(stationarity["stationarity_consensus_mean"][0])
                stationarity_final = float(stationarity["stationarity_consensus_mean"][-1])
                ranking.update(
                    {
                        "spade_final_accuracy_std": spade_std,
                        "best_final_accuracy_std": ordered[0][2],
                        "stationarity_consensus_start": stationarity_start,
                        "stationarity_consensus_final": stationarity_final,
                        "stationarity_consensus_final_std": float(
                            stationarity["stationarity_consensus_std"][-1]
                        ),
                        "stationarity_consensus_change": stationarity_final - stationarity_start,
                        "stationarity_consensus_increased": stationarity_final > stationarity_start,
                    }
                )
            rankings.append(ranking)
    except (KeyError, OSError, TypeError, ValueError) as error:
        max_error = float("inf")
        _append_error(errors, f"final_metrics invalid: {error}")
    if max_error > _TOLERANCE:
        _append_error(errors, "final_metrics does not equal raw aggregate")
    return rankings, max_error


def _parameter_rows(
    root: Path,
    suite_manifest: Mapping[str, object] | None,
    errors: list[str],
    *,
    is_v2: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if suite_manifest is None:
        return rows
    config_path = Path(str(suite_manifest.get("config_path", "")))
    try:
        config = _read_json(config_path)
        provenance = config["selection_provenance"]
        config_dir = config_path.parent
        if is_v2:
            lock = config_dir / str(provenance["selection_lock"])
            lock_data = _read_json(lock)
            selected = (lock.parent / str(lock_data["selected_parameters_path"])).resolve()
            manifest = (lock.parent / str(lock_data["validation_manifest_path"])).resolve()
            if provenance.get("selection_lock_sha256") != _sha256(lock):
                _append_error(errors, "v2 selection lock SHA-256 provenance mismatch")
            manifest_passed = (
                manifest.is_file()
                and lock_data.get("validation_manifest_sha256") == _sha256(manifest)
                and provenance.get("validation_manifest_sha256") == _sha256(manifest)
            )
            rows.append({"scope": "selection", "parameter": "validation_manifest_sha256", "source_type": "Frozen", "source": manifest.as_posix(), "value": provenance.get("validation_manifest_sha256", ""), "accepted": manifest_passed})
            if not manifest_passed:
                _append_error(errors, "v2 validation manifest SHA-256 provenance mismatch")
        else:
            selected = config_dir.parent.parent / "results" / "parameter_validation" / str(
                provenance["selected_parameters"]
            )
            lock = config_dir / "alzheimer_reviewed_selection_lock.json"
        passed = selected.is_file() and provenance.get("selected_parameters_sha256") == _sha256(selected)
        rows.append({"scope": "selection", "parameter": "selected_parameters_sha256", "source_type": "Frozen", "source": selected.as_posix(), "value": provenance.get("selected_parameters_sha256", ""), "accepted": passed})
        if not passed:
            _append_error(errors, "selected parameter SHA-256 provenance mismatch")
        lock_key = "selection_lock_sha256" if is_v2 else "reviewed_selection_lock_sha256"
        lock_passed = lock.is_file() and provenance.get(lock_key) == _sha256(lock)
        rows.append({"scope": "selection", "parameter": lock_key, "source_type": "Frozen", "source": lock.as_posix(), "value": provenance.get(lock_key, ""), "accepted": lock_passed})
        if not lock_passed:
            _append_error(errors, f"{lock_key} provenance mismatch")
        for name, record in config.get("parameter_sources", {}).items():
            valid = isinstance(record, Mapping) and bool(record.get("source")) and bool(record.get("source_type"))
            rows.append({"scope": "experiment", "parameter": name, "source_type": record.get("source_type", "") if isinstance(record, Mapping) else "", "source": record.get("source", "") if isinstance(record, Mapping) else "", "value": record.get("value", "") if isinstance(record, Mapping) else "", "accepted": valid})
            if not valid:
                _append_error(errors, f"parameter source record invalid for {name}")
        for method, values in config.get("method_parameters", {}).items():
            for name, value in values.items():
                if name.endswith("_source"):
                    valid = (
                        isinstance(value, Mapping)
                        and bool(value.get("source"))
                        and bool(value.get("source_type"))
                    )
                    rows.append({"scope": method, "parameter": name.removesuffix("_source"), "source_type": value.get("source_type", "") if isinstance(value, Mapping) else "", "source": value.get("source", "") if isinstance(value, Mapping) else "", "value": value.get("value", "") if isinstance(value, Mapping) else "", "accepted": valid})
                    if not valid:
                        _append_error(
                            errors,
                            f"method parameter source record invalid for {method}.{name}",
                        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        _append_error(errors, f"frozen parameter provenance invalid: {error}")
    return rows


def audit_alzheimer_results(root: Path, output: Path | None = None) -> dict[str, object]:
    """Audit an immutable local suite; only derived audit files are written."""
    root = Path(root)
    output = Path(output) if output is not None else root / "audit"
    errors: list[str] = []
    suite_manifest = _audit_suite_manifest(root, errors)
    protocol_version = (
        str(suite_manifest.get("protocol_version"))
        if suite_manifest is not None and suite_manifest.get("protocol_version")
        else None
    )
    is_v2 = protocol_version == _V2_PROTOCOL_VERSION
    v2_suite_path = (
        Path(str(suite_manifest.get("config_path", ""))) if is_v2 and suite_manifest else None
    )
    dataset_sha256 = (
        str(suite_manifest["dataset_sha256"])
        if suite_manifest is not None and suite_manifest.get("dataset_sha256")
        else None
    )
    if dataset_sha256 is None:
        _append_error(errors, "suite_manifest dataset SHA-256 is missing")
    expected, run_count, query_count = _raw_expected_aggregates(
        root,
        errors,
        dataset_sha256,
        is_v2=is_v2,
        v2_suite_path=v2_suite_path,
    )
    curve_rows, aggregate_error = _audit_aggregate_and_figures(
        root, suite_manifest, expected, errors, is_v2=is_v2
    )
    rankings, final_error = _audit_final_metrics(root, expected, errors, is_v2=is_v2)
    parameter_rows = _parameter_rows(root, suite_manifest, errors, is_v2=is_v2)
    max_curve_error = max(aggregate_error, final_error)
    accepted = not errors
    stationarity_increased = any(
        bool(row.get("stationarity_consensus_increased", False)) for row in rankings
    )
    scientific_acceptance = (
        all(row["spade_rank"] == 1 for row in rankings) and not stationarity_increased
        if is_v2
        else False
    )
    scientific_reason = (
        "SPADE-DFL is not rank 1 in every frozen local partition; this is a valid scientific negative result."
        if not is_v2
        else (
            "SPADE-DFL is rank 1 in every frozen local partition and its "
            "stationarity-consensus did not increase."
            if scientific_acceptance
            else "SPADE-DFL is not rank 1 in every frozen v2 local partition "
            "and/or its stationarity-consensus increased; this is a scientific negative result."
        )
    )
    summary: dict[str, object] = {
        "accepted": accepted,
        "scientific_acceptance": scientific_acceptance,
        "scientific_acceptance_reason": scientific_reason,
        "run_count": run_count,
        "point_count": _POINTS,
        "methods": list(ALZHEIMER_METHOD_IDS),
        "partitions": list(ALZHEIMER_PARTITIONS),
        "seeds": list(_SEEDS),
        "query_count_total": query_count,
        "tolerance": _TOLERANCE,
        "max_curve_error": max_curve_error if math.isfinite(max_curve_error) else None,
        "spade_rankings": rankings,
        "errors": errors,
    }
    if is_v2:
        summary["protocol_version"] = _V2_PROTOCOL_VERSION
    _write_csv(
        output / "curve_integrity.csv",
        ("partition", "method_id", "artifact", "max_abs_error", "accepted"),
        curve_rows,
    )
    _write_csv(
        output / "parameter_integrity.csv",
        ("scope", "parameter", "source_type", "source", "value", "accepted"),
        parameter_rows,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit_summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary


def _note_markdown(
    config: Mapping[str, object], audit: Mapping[str, object]
) -> str:
    sources = config.get("parameter_sources", {})
    source_rows = []
    for name, record in sources.items() if isinstance(sources, Mapping) else ():
        if isinstance(record, Mapping):
            source_rows.append(
                f"| {name} | {record.get('value', '')} | {record.get('source_type', '')} | {record.get('source', '')} |"
            )
    methods = config.get("method_parameters", {})
    for method, values in methods.items() if isinstance(methods, Mapping) else ():
        if not isinstance(values, Mapping):
            continue
        for name, value in values.items():
            source = values.get(f"{name}_source")
            if name.endswith("_source") or not isinstance(source, Mapping):
                continue
            source_rows.append(
                f"| {method}.{name} | {value} | {source.get('source_type', '')} | {source.get('source', '')} |"
            )
    ranking_rows = []
    for row in audit.get("spade_rankings", []):
        if isinstance(row, Mapping):
            ranking_rows.append(
                "| {partition} | {rank}/7 | {spade:.6f} | {best} | {gap:.6f} |".format(
                    partition=row["partition"],
                    rank=row["spade_rank"],
                    spade=float(row["spade_final_accuracy"]),
                    best=row["best_method_id"],
                    gap=float(row["gap_to_best"]),
                )
            )
    return "\n".join(
        [
            "# Alzheimer SPADE-DFL 参数来源与证据说明",
            "",
            "## 先给初学者的结论",
            "",
            "本说明只转录冻结配置和训练期验证记录，不新增或猜测任何参数。当前本地实验的机械完整性（文件、均值和标准差是否能由原始运行重算）与科学结论要分开：即使完整性通过，当前结果也**不能**证明 SPADE-DFL 最好，也不能据此宣称服务器规模实验成功。",
            "",
            "- **冻结配置**：指在最终测试前已写死、不能再根据测试结果调节的参数文件。",
            "- **随机种子**：控制随机过程的整数；这里本地只用了 3 个种子（0、1、2）。",
            "- **验证集**：训练数据中留出的比较区域，只用于选参数，不能替代最终测试集。",
            "- **总体标准差**：3 个种子结果的离散程度，公式分母是 3（不是 2）。",
            "",
            "## 参数来源分类",
            "",
            "- **直接参考值（Reference）**：直接来自论文或实现说明的值。",
            "- **理论推导值（TheoryDerived）**：按图结构、维度或算法公式计算出的值，而不是看最终测试后选择的值。",
            "- **训练期验证值（Validation）**：仅在训练/验证划分上比较候选值后冻结的值。",
            "- **实验协议假设（Protocol assumption）**：客户端数、轮数、划分和随机种子等比较规则；它们定义实验，不是由结果反推。",
            "",
            "| 参数 | 冻结值 | 来源类别 | 冻结来源 |",
            "|---|---:|---|---|",
            *source_rows,
            "",
            "## 当前四分区负结果（最终轮准确率）",
            "",
            "排名 1/7 表示七种方法中最终准确率最高；gap 是该分区最佳方法减去 SPADE-DFL 的准确率。",
            "",
            "| 分区 | SPADE 排名 | SPADE 准确率 | 最佳方法 | gap |",
            "|---|---:|---:|---|---:|",
            *ranking_rows,
            "",
            "四个分区的冻结本地证据显示 SPADE-DFL 未在每个分区取得第 1 名。因此当前结果是科学负结果：它不支持“SPADE-DFL 最好”的表述，也不能作为 30 种子服务器成功的证据。30 个随机种子的服务器实验尚未执行，仍处于 pending 状态。",
            "",
            "## 必须披露的限制和可比性说明",
            "",
            "1. 本 Alzheimer 数据集是合成/教学用途数据，不应直接外推为临床有效性结论。",
            "2. 本地结果只有 3 个随机种子，统计证据有限；计划中的 30 个随机种子服务器结果尚未产生。",
            "3. 图中 DP-SGD 的横轴是对齐的外层优化步（为了与其他方法逐轮比较），不是网络通信事件。",
            "4. **函数查询预算**指算法向目标函数索取数值的次数。不同方法每轮需要的查询次数不同，因此当前曲线不是等查询预算比较；审计文件保留每个原始运行的查询计数。",
            "",
            "## 审计状态",
            "",
            f"机械完整性 accepted = `{audit.get('accepted')}`；科学接受 scientific_acceptance = `{audit.get('scientific_acceptance')}`。前者只判断工件是否可追溯、可重算，后者判断能否支持科学成功主张，两者不能混为一谈。",
            "",
        ]
    )


def _v2_note_markdown(config: Mapping[str, object], audit: Mapping[str, object]) -> str:
    """Render the v2 evidence note without changing the frozen v1 note."""
    methods = config.get("method_parameters", {})
    if not isinstance(methods, Mapping):
        raise TypeError("v2 method parameters must be an object")
    eta_rows = []
    for method in ALZHEIMER_METHOD_IDS:
        values = methods.get(method, {})
        if not isinstance(values, Mapping):
            raise TypeError(f"v2 method parameters missing for {method}")
        eta_rows.append(
            "| {method} | {eta} | {source_type} | {source} |".format(
                method=DISPLAY_LABELS[method],
                eta=values["eta0"],
                source_type=values["eta0_source"]["source_type"],
                source=values["eta0_source"]["source"],
            )
        )
    spade = methods.get("spade_dfl", {})
    if not isinstance(spade, Mapping):
        raise TypeError("v2 SPADE parameter record missing")
    rankings = []
    for row in audit.get("spade_rankings", []):
        if not isinstance(row, Mapping):
            continue
        rankings.append(
            "| {partition} | {rank}/7 | {accuracy:.6f} ± {std:.6f} | {best} | "
            "{best_accuracy:.6f} ± {best_std:.6f} | {gap:.6f} | "
            "{start:.6f} → {final:.6f} (Δ={change:+.6f}) |".format(
                partition=row["partition"],
                rank=row["spade_rank"],
                accuracy=float(row["spade_final_accuracy"]),
                std=float(row["spade_final_accuracy_std"]),
                best=row["best_method_id"],
                best_accuracy=float(row["best_final_accuracy"]),
                best_std=float(row["best_final_accuracy_std"]),
                gap=float(row["gap_to_best"]),
                start=float(row["stationarity_consensus_start"]),
                final=float(row["stationarity_consensus_final"]),
                change=float(row["stationarity_consensus_change"]),
            )
        )
    return "\n".join(
        [
            "# Alzheimer SPADE-DFL v2 参数来源与验证说明",
            "",
            "## 结论（先读这一段）",
            "",
            "历史 `alzheimer_v2_local_3seed_50round` 工件记录了两个独立状态：原始运行、聚合、图和指标的可重算性审计为 `accepted=true`，科学结果审计为 `scientific_acceptance=false`。因此，该 v2 本地变体是可追溯的科学负结果；它不是 30-seed 服务器变体，也不构成服务器变体成功的记录。",
            "",
            "## v2 目标函数与公平协议",
            "",
            "- 目标函数：`lt_admm_nonconvex_logistic`；正则化为 0.01，依据 Reference [13] 的非凸 logistic 目标函数。",
            "- 零噪声 oracle：乘性噪声 `sigma_u=0`、加性查询噪声 `query_noise_std=0`。这是论文有界零均值 oracle 噪声假设的零方差特例；SPADE-DFL 仍只能进行 one-point 损失查询，未获得解析梯度。",
            "- 初始化：所有方法、所有客户端采用 `shared_zero`，即同一个全零初始点，避免初始随机性造成的方法间不公平。",
            "- 公平本地训练：SPADE-DFL、LT-ADMM-VR、LT-ADMM 都使用 `tau=8`，该有限本地训练值来自 Reference [13]。",
            f"- SAGA batch：`batch_size={spade['batch_size']}`；来源为 Validation，是 8 和 16 两个最佳括号候选之间唯一允许的整数中点细化，并非论文直接给定值。",
            "",
            "## 训练/验证选参记录（未读取正式测试集）",
            "",
            "所有七种方法的 eta0 都在训练/验证划分上选择；下表完整列出 SPADE-DFL 和六个基线的冻结 eta0，而不是只报告有利的方法。",
            "",
            "| 方法 | eta0 | 来源类别 | 来源 |",
            "|---|---:|---|---|",
            *eta_rows,
            "",
            "## 正式 v2 四分区结果：排名、gap、标准差与 stationarity-consensus",
            "",
            "标准差为 3 个随机种子的总体标准差（ddof=0）。gap 是分区最佳方法的平均最终准确率减去 SPADE-DFL 的平均最终准确率。`stationarity-consensus` 从第 0 轮到第 50 轮均上升；这不是可忽略的成功信号，必须如实报告。",
            "",
            "| 分区 | SPADE 排名 | SPADE 最终准确率 ± std | 最佳方法 | 最佳准确率 ± std | gap | SPADE stationarity-consensus |",
            "|---|---:|---:|---|---:|---:|---:|",
            *rankings,
            "",
            "四个分区的实际 SPADE 排名依次为 3/7、3/7、3/7、2/7；因此并非每个分区第一。并且四个分区的 stationarity-consensus 上升。两项事实对应历史审计记录中的 `scientific_acceptance=false`；可重算性审计通过不改变这一科学负结果。",
            "",
            "## v1 负结果与迭代确认性限制",
            "",
            "历史 v1 `alzheimer_local_3seed_50round` 工件的审计状态为 `accepted=true`、`scientific_acceptance=false`；SPADE-DFL 在 dir_0.3、dir_0.6、iid、pathological 的排名分别为 3/7、4/7、4/7、7/7。历史 v2 `alzheimer_v2_local_3seed_50round` 变体改为 Reference-[13] 目标函数、零噪声 oracle 和 shared_zero 后执行正式测试；因为 v1 最终测试结果已被看到，v2 是迭代确认性（iterative-confirmatory）实验，而不是首次使用的纯净留出测试。",
            "",
            "## 已记录的解释边界",
            "",
            "1. 函数查询预算：不同方法每轮损失查询次数不同，因此当前横轴比较的是通信/外层轮次，不是等函数查询预算比较。",
            "2. 3 个随机种子：正式本地结果只有 3 个随机种子，表中的标准差只描述这 3 次运行的离散程度，不能当作统计显著性证明。",
            "3. 30 个随机种子：历史 v2 本地审计记录为 `scientific_acceptance=false`；`alzheimer_v2_server_30seed_50round` 服务器变体仍为 pending，未生成带成功状态的 30-seed 归档包。",
            "4. 数据集为合成/教学用途；结果不能外推为临床有效性结论。",
            "",
        ]
    )


def generate_alzheimer_parameter_note(
    *, project_root: Path, results_root: Path, output_dir: Path
) -> dict[str, Path]:
    """Write a Markdown and Chinese Word note from frozen metadata only."""
    project_root = Path(project_root)
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    audit_path = results_root / "audit" / "audit_summary.json"
    suite = _read_json(results_root / "suite_manifest.json")
    is_v2 = suite.get("protocol_version") == _V2_PROTOCOL_VERSION
    audit = (
        audit_alzheimer_results(results_root)
        if is_v2
        else (_read_json(audit_path) if audit_path.is_file() else audit_alzheimer_results(results_root))
    )
    config = _read_json(Path(str(suite["config_path"])))
    markdown = _v2_note_markdown(config, audit) if is_v2 else _note_markdown(config, audit)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = (
        "Alzheimer_SPADE_DFL_v2_参数来源与验证说明"
        if is_v2
        else "Alzheimer_SPADE_DFL_参数来源与验证说明"
    )
    markdown_path = output_dir / f"{name}.md"
    docx_path = output_dir / f"{name}.docx"
    markdown_path.write_text(markdown, encoding="utf-8")
    from docx import Document

    document = Document()
    table_lines: list[list[str]] = []

    def flush_table() -> None:
        if not table_lines:
            return
        table = document.add_table(rows=1, cols=len(table_lines[0]))
        table.style = "Table Grid"
        for cell, value in zip(table.rows[0].cells, table_lines[0]):
            cell.text = value
        for row_values in table_lines[1:]:
            row = table.add_row().cells
            for cell, value in zip(row, row_values):
                cell.text = value
        table_lines.clear()

    for line in markdown.splitlines():
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            table_lines.append(cells)
            continue
        flush_table()
        if line.startswith("# "):
            document.add_heading(line[2:], level=0)
        elif line.startswith("## "):
            document.add_heading(line[3:], level=1)
        elif line.startswith("- "):
            document.add_paragraph(line[2:], style="List Bullet")
        elif line:
            document.add_paragraph(line)
    flush_table()
    document.save(docx_path)
    return {"markdown": markdown_path, "docx": docx_path}
