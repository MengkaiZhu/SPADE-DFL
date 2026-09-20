from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from .artifacts import load_run, save_run
from .brain_tumor_contract import (
    brain_tumor_base_config,
    brain_tumor_method_config,
    partition_brain_tumor,
)
from .brain_tumor_resnet18_data import build_holdout_bundle, load_feature_cache
from .isolation import assert_write_path
from .methods import run_method


GATE2_HOLDOUT_SEEDS = (20260724, 20260725, 20260726)
GATE2_PANEL_ORDER = ("dir_0.3", "dir_0.6", "iid", "pathological")
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
_NUMERIC_FIELDS = (
    "final_validation_accuracy",
    "initial_loss",
    "final_loss",
    "initial_stationarity_consensus",
    "final_stationarity_consensus",
)


def derive_partition_seed(holdout_seed: int, panel: str) -> int:
    if holdout_seed not in GATE2_HOLDOUT_SEEDS:
        raise ValueError(f"unknown Gate 2 holdout seed: {holdout_seed}")
    try:
        panel_index = GATE2_PANEL_ORDER.index(panel)
    except ValueError as error:
        raise ValueError(f"unknown Gate 2 panel: {panel}") from error
    algorithm_seed = holdout_seed - GATE2_HOLDOUT_SEEDS[0]
    return int(
        np.random.SeedSequence(
            [algorithm_seed, panel_index, holdout_seed]
        ).generate_state(1)[0]
    )


def evaluate_gate2(
    rows: Sequence[Mapping[str, object]],
    old_panel_accuracy: Mapping[str, float],
) -> dict[str, object]:
    failures: list[str] = []
    expected = {
        (panel, seed)
        for panel in GATE2_PANEL_ORDER
        for seed in GATE2_HOLDOUT_SEEDS
    }
    keys = [
        (str(row.get("panel")), int(row.get("holdout_seed", -1)))
        for row in rows
    ]
    counts = Counter(keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    missing = sorted(expected - set(keys))
    unexpected = sorted(set(keys) - expected)
    if duplicates:
        failures.append(f"duplicate panel/holdout rows: {duplicates}")
    if missing:
        failures.append(f"missing panel/holdout rows: {missing}")
    if unexpected:
        failures.append(f"unexpected panel/holdout rows: {unexpected}")

    for index, row in enumerate(rows):
        label = f"row {index} ({row.get('panel')}, {row.get('holdout_seed')})"
        numeric_values: list[float] = []
        for field in _NUMERIC_FIELDS:
            try:
                numeric_values.append(float(row[field]))
            except (KeyError, TypeError, ValueError):
                numeric_values.append(math.nan)
        if not bool(row.get("finite")) or not all(
            math.isfinite(value) for value in numeric_values
        ):
            failures.append(f"{label} contains non-finite or invalid values")
        if bool(row.get("official_test_used")):
            failures.append(f"{label} used official Testing data")
        if numeric_values[2] >= numeric_values[1]:
            failures.append(f"{label} final loss did not strictly decrease")
        if numeric_values[4] >= numeric_values[3]:
            failures.append(
                f"{label} final stationarity-consensus did not strictly decrease"
            )
        if str(row.get("error_type", "")) or str(row.get("error", "")):
            failures.append(f"{label} recorded an execution error")

    panel_reports: dict[str, dict[str, object]] = {}
    for panel in GATE2_PANEL_ORDER:
        if panel not in old_panel_accuracy:
            failures.append(f"old accuracy is missing for panel {panel}")
            old_mean = math.nan
        else:
            old_mean = float(old_panel_accuracy[panel])
        panel_rows = [row for row in rows if row.get("panel") == panel]
        accuracies = np.asarray(
            [float(row.get("final_validation_accuracy", math.nan)) for row in panel_rows],
            dtype=np.float64,
        )
        new_mean = (
            float(accuracies.mean())
            if accuracies.size and np.isfinite(accuracies).all()
            else math.nan
        )
        strict_improvement = bool(
            math.isfinite(old_mean)
            and math.isfinite(new_mean)
            and new_mean > old_mean
        )
        if not strict_improvement:
            failures.append(
                f"panel {panel} does not strictly improve: old={old_mean}, new={new_mean}"
            )
        panel_reports[panel] = {
            "old_mean_accuracy": old_mean,
            "new_mean_accuracy": new_mean,
            "accuracy_delta": new_mean - old_mean,
            "new_seed_accuracies": accuracies.tolist(),
            "strict_accuracy_improvement": strict_improvement,
            "run_count": len(panel_rows),
        }

    return {
        "passed": not failures,
        "run_count": len(rows),
        "expected_run_count": len(expected),
        "holdout_seeds": list(GATE2_HOLDOUT_SEEDS),
        "panel_order": list(GATE2_PANEL_ORDER),
        "panels": panel_reports,
        "failure_reasons": failures,
        "official_testing_used": any(
            bool(row.get("official_test_used")) for row in rows
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_state_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def gate2_protocol_sha256(config_path: Path, frozen_path: Path) -> str:
    paths = sorted((EXPERIMENT_ROOT / "src" / "spade_dfl").glob("*.py"))
    paths.extend(
        [
            EXPERIMENT_ROOT / "scripts" / "run_resnet18_gate2.py",
            Path(config_path).resolve(),
            Path(frozen_path).resolve(),
            EXPERIMENT_ROOT / "requirements.txt",
            EXPERIMENT_ROOT / "pyproject.toml",
        ]
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Gate 2 protocol input is missing: "
            + ", ".join(str(path) for path in missing)
        )
    digest = hashlib.sha256()
    for path in sorted(paths):
        try:
            label = path.relative_to(EXPERIMENT_ROOT).as_posix()
        except ValueError:
            label = str(path)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_protocol(
    config_path: Path,
    frozen_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    protocol = json.loads(Path(config_path).read_text(encoding="utf-8"))
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    if tuple(protocol.get("holdout_seeds", ())) != GATE2_HOLDOUT_SEEDS:
        raise ValueError("Gate 2 holdout seeds do not match the frozen protocol")
    if tuple(protocol.get("partition_seed_panel_order", ())) != GATE2_PANEL_ORDER:
        raise ValueError("Gate 2 panel order does not match the frozen protocol")
    if bool(protocol.get("official_testing_allowed", True)):
        raise ValueError("Gate 2 must exclude official Testing")
    if frozen.get("selection_status") != "selected":
        raise ValueError("the frozen SPADE parameter selection was not accepted")
    if bool(frozen.get("official_test_used_for_selection", True)):
        raise ValueError("the frozen SPADE selection used official Testing")
    expected_parameters = dict(protocol.get("selected_parameters", {}))
    actual_parameters = {
        key: frozen.get(key)
        for key in ("eta0", "rho", "batch_size", "n_rounds")
    }
    if expected_parameters != actual_parameters:
        raise ValueError("Gate 2 parameters do not match selected_parameters_robust.json")
    old_panel_accuracy = frozen.get("per_panel_accuracy")
    if not isinstance(old_panel_accuracy, dict) or set(old_panel_accuracy) != set(
        GATE2_PANEL_ORDER
    ):
        raise ValueError("frozen old-representation panel accuracies are incomplete")
    return protocol, frozen


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _result_is_finite(result: object) -> bool:
    return bool(
        np.isfinite(result.states).all()
        and all(np.isfinite(values).all() for values in result.history.values())
    )


def run_gate2(
    feature_cache: Path,
    config_path: Path,
    frozen_path: Path,
    output_root: Path,
    *,
    resume: bool = True,
) -> dict[str, object]:
    protocol, frozen = _load_protocol(config_path, frozen_path)
    matrix, feature_metadata = load_feature_cache(feature_cache)
    if matrix.dataset_sha256 != protocol["dataset_sha256"]:
        raise ValueError("feature cache dataset SHA-256 does not match Gate 2 protocol")
    if bool(feature_metadata.get("official_testing_read", True)):
        raise ValueError("feature cache metadata indicates official Testing was read")
    output = assert_write_path(Path(output_root), EXPERIMENT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    feature_cache_hash = _sha256(Path(feature_cache))
    protocol_hash = gate2_protocol_sha256(config_path, frozen_path)
    parameters = dict(protocol["selected_parameters"])
    rows: list[dict[str, object]] = []
    protocol_started = time.perf_counter()
    for holdout_seed in GATE2_HOLDOUT_SEEDS:
        holdout = build_holdout_bundle(
            matrix,
            fraction=float(protocol["holdout_fraction"]),
            seed=holdout_seed,
            dimension=int(protocol["feature_dimension"]),
        )
        dataset = holdout.bundle
        algorithm_seed = holdout_seed - GATE2_HOLDOUT_SEEDS[0]
        pca_sha256 = _array_state_sha256(
            dataset.pca.mean,
            dataset.pca.components,
            dataset.pca.scale,
        )
        for panel in GATE2_PANEL_ORDER:
            partition_seed = derive_partition_seed(holdout_seed, panel)
            partition = partition_brain_tumor(
                dataset.train_y,
                int(protocol["n_clients"]),
                panel,
                seed=partition_seed,
            )
            base = brain_tumor_base_config(
                n_rounds=int(parameters["n_rounds"]),
                seeds=(algorithm_seed,),
                n_clients=int(protocol["n_clients"]),
                eta0=float(parameters["eta0"]),
                rho=float(parameters["rho"]),
                batch_size=int(parameters["batch_size"]),
            )
            config = brain_tumor_method_config(
                base,
                "spade_dfl",
                (algorithm_seed,),
                partition.mode,
                partition.alpha,
            )
            config = replace(
                config,
                data_path=str(Path(feature_cache).resolve()),
                output_dir=str(output),
            )
            run_parent = output / "raw" / panel / f"holdout_{holdout_seed}"
            run_dir = run_parent / config.stage / f"seed_{algorithm_seed}"
            dataset_metadata = {
                "name": dataset.dataset_name,
                "sha256": dataset.sha256,
                "source_path": dataset.source_path,
                "feature_cache_sha256": feature_cache_hash,
                "sample_order_sha256": matrix.sample_order_sha256,
                "pca_sha256": pca_sha256,
                "train_count": int(dataset.train_x.shape[0]),
                "validation_count": int(dataset.test_x.shape[0]),
                "holdout_seed": holdout_seed,
                "partition": panel,
                "partition_seed": partition_seed,
                "label_histograms": partition.label_histograms.tolist(),
                "official_test_role": "excluded_from_gate2",
                "source_tree_sha256": protocol_hash,
            }
            try:
                if resume and (run_dir / "COMPLETED").is_file():
                    result = load_run(
                        run_dir,
                        expected_config=config,
                        expected_dataset_sha256=dataset.sha256,
                        expected_partition=panel,
                        expected_label_histograms=partition.label_histograms,
                        expected_seed=algorithm_seed,
                        expected_stage=config.stage,
                        expected_history_length=int(parameters["n_rounds"]) + 1,
                        expected_state_shape=(int(protocol["n_clients"]), 40),
                        expected_source_sha256=protocol_hash,
                    )
                    resumed = True
                else:
                    result = run_method(
                        "spade_dfl",
                        config,
                        dataset,
                        partition,
                        algorithm_seed,
                    )
                    save_run(result, config, run_parent, dataset_metadata)
                    resumed = False
                row = {
                    "candidate_id": frozen["candidate_id"],
                    "holdout_seed": holdout_seed,
                    "algorithm_seed": algorithm_seed,
                    "partition_seed": partition_seed,
                    "panel": panel,
                    "initial_accuracy": float(result.history["accuracy"][0]),
                    "final_validation_accuracy": float(result.history["accuracy"][-1]),
                    "initial_loss": float(result.history["loss"][0]),
                    "final_loss": float(result.history["loss"][-1]),
                    "initial_stationarity_consensus": float(
                        result.history["stationarity_consensus"][0]
                    ),
                    "final_stationarity_consensus": float(
                        result.history["stationarity_consensus"][-1]
                    ),
                    "query_count": int(result.query_count),
                    "wall_time_seconds": float(result.wall_time),
                    "finite": _result_is_finite(result),
                    "error_type": "",
                    "error": "",
                    "official_test_used": False,
                    "resumed": resumed,
                    "feature_cache_sha256": feature_cache_hash,
                    "pca_sha256": pca_sha256,
                    "protocol_sha256": protocol_hash,
                }
            except (FloatingPointError, OverflowError, ValueError) as error:
                row = {
                    "candidate_id": frozen["candidate_id"],
                    "holdout_seed": holdout_seed,
                    "algorithm_seed": algorithm_seed,
                    "partition_seed": partition_seed,
                    "panel": panel,
                    "initial_accuracy": math.nan,
                    "final_validation_accuracy": math.nan,
                    "initial_loss": math.nan,
                    "final_loss": math.nan,
                    "initial_stationarity_consensus": math.nan,
                    "final_stationarity_consensus": math.nan,
                    "query_count": 0,
                    "wall_time_seconds": 0.0,
                    "finite": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "official_test_used": False,
                    "resumed": False,
                    "feature_cache_sha256": feature_cache_hash,
                    "pca_sha256": pca_sha256,
                    "protocol_sha256": protocol_hash,
                }
            rows.append(row)
            _write_rows(output / "raw_runs.csv", rows)
            print(
                f"Gate2 {len(rows)}/12 holdout={holdout_seed} panel={panel} "
                f"accuracy={row['final_validation_accuracy']}",
                flush=True,
            )
    report = evaluate_gate2(rows, frozen["per_panel_accuracy"])
    report.update(
        {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_hash,
            "feature_cache_sha256": feature_cache_hash,
            "feature_cache_metadata": feature_metadata,
            "selected_parameters": parameters,
            "old_accuracy_source": str(Path(frozen_path).resolve()),
            "raw_runs_sha256": _sha256(output / "raw_runs.csv"),
            "total_query_count": int(sum(int(row["query_count"]) for row in rows)),
            "total_recorded_run_wall_time_seconds": float(
                sum(float(row["wall_time_seconds"]) for row in rows)
            ),
            "protocol_wall_time_seconds": time.perf_counter() - protocol_started,
        }
    )
    (output / "gate2_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
