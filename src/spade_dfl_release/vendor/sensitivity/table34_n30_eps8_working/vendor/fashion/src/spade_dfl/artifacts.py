from __future__ import annotations

import csv
import json
import platform
from pathlib import Path
import shutil
import sys
from typing import Dict, Iterable, Mapping
import uuid

import numpy as np

from .config import ExperimentConfig
from .data import PCAState
from .trainer import RunResult


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def save_run(
    result: RunResult,
    config: ExperimentConfig,
    output_root: Path,
    dataset_metadata: Mapping[str, object],
) -> Path:
    seed = result.metadata.get("seed", "unknown")
    stage = str(result.metadata.get("stage", config.stage))
    run_dir = Path(output_root) / stage / f"seed_{seed}"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = run_dir.parent / f".{run_dir.name}.part-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        (staging / "config.json").write_text(
            json.dumps(
                config.to_dict(), ensure_ascii=False, indent=2, default=_json_default
            ),
            encoding="utf-8",
        )
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "git": "unavailable",
            "dataset": dict(dataset_metadata),
        }
        (staging / "environment.json").write_text(
            json.dumps(
                environment, ensure_ascii=False, indent=2, default=_json_default
            ),
            encoding="utf-8",
        )
        names = list(result.history)
        n_rows = max(len(result.history[name]) for name in names)
        with (staging / "history.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["iteration", *names])
            for index in range(n_rows):
                values = [
                    float(result.history[name][index])
                    if index < len(result.history[name])
                    else ""
                    for name in names
                ]
                writer.writerow([index, *values])
        np.savez_compressed(staging / "states.npz", states=result.states)
        summary = {
            "query_count": result.query_count,
            "gradient_evaluations": int(
                result.metadata.get("gradient_evaluations", 0)
            ),
            "communication_rounds": int(
                result.metadata.get("communication_rounds", 0)
            ),
            "transmitted_bytes": result.transmitted_bytes,
            "wall_time": result.wall_time,
            "epsilon": result.metadata.get("epsilon"),
            "delta": result.metadata.get("delta", 0.0),
            "privacy_scope": result.metadata.get("privacy_scope", "none"),
            "privacy_valid": bool(result.metadata.get("privacy_valid", True)),
            "initial_loss": float(result.history["loss"][0]),
            "final_loss": float(result.history["loss"][-1]),
            "final_accuracy": float(result.history["accuracy"][-1]),
            "metadata": result.metadata,
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").unlink(missing_ok=True)
        for name in (
            "config.json",
            "environment.json",
            "history.csv",
            "states.npz",
            "summary.json",
        ):
            (staging / name).replace(run_dir / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return run_dir


def load_run(
    run_dir: Path,
    *,
    expected_config: ExperimentConfig | None = None,
    expected_dataset_sha256: str | None = None,
    expected_partition: str | None = None,
    expected_label_histograms: object | None = None,
    expected_seed: int | None = None,
    expected_stage: str | None = None,
    expected_history_length: int | None = None,
    expected_state_shape: tuple[int, ...] | None = None,
) -> RunResult:
    """Load a completed run artifact for deterministic suite resumption."""
    run_dir = Path(run_dir)
    required = (
        run_dir / "config.json",
        run_dir / "environment.json",
        run_dir / "history.csv",
        run_dir / "states.npz",
        run_dir / "summary.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("incomplete run artifact: " + ", ".join(missing))
    stored_config = json.loads(required[0].read_text(encoding="utf-8"))
    if expected_config is not None and stored_config != expected_config.to_dict():
        raise ValueError("stored run configuration does not match expected configuration")
    environment = json.loads(required[1].read_text(encoding="utf-8"))
    dataset_metadata = environment.get("dataset")
    if not isinstance(dataset_metadata, Mapping):
        raise ValueError("stored run environment has no dataset metadata")
    if (
        expected_dataset_sha256 is not None
        and dataset_metadata.get("sha256") != expected_dataset_sha256
    ):
        raise ValueError("stored run dataset SHA-256 does not match expected dataset")
    if (
        expected_partition is not None
        and dataset_metadata.get("partition") != expected_partition
    ):
        raise ValueError("stored run partition does not match expected partition")
    if expected_label_histograms is not None and dataset_metadata.get(
        "label_histograms"
    ) != np.asarray(expected_label_histograms).tolist():
        raise ValueError(
            "stored run partition label histograms do not match expected partition"
        )
    with required[2].open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "iteration" not in rows[0]:
        raise ValueError("run history is empty or malformed")
    if expected_history_length is None and expected_config is not None:
        expected_history_length = expected_config.n_rounds + 1
    if expected_history_length is not None and len(rows) != expected_history_length:
        raise ValueError(
            "stored run history length does not match expected n_rounds + 1"
        )
    try:
        iterations = np.asarray([float(row["iteration"]) for row in rows])
    except (TypeError, ValueError) as exc:
        raise ValueError("run history iteration column is malformed") from exc
    if not np.array_equal(iterations, np.arange(len(rows), dtype=np.float64)):
        raise ValueError("run history iteration column is malformed")
    history: Dict[str, np.ndarray] = {}
    for name in rows[0]:
        if name == "iteration":
            continue
        raw_values = [row[name] for row in rows]
        while raw_values and raw_values[-1] == "":
            raw_values.pop()
        try:
            values = np.asarray(
                [float(value) if value != "" else np.nan for value in raw_values],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("run history must contain finite numeric values") from exc
        if not values.size or not np.all(np.isfinite(values)):
            raise ValueError("run history must contain finite numeric values")
        history[name] = values
    summary = json.loads(required[4].read_text(encoding="utf-8"))
    metadata = summary.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("stored run summary has no execution metadata")
    if expected_seed is None and run_dir.name.startswith("seed_"):
        try:
            expected_seed = int(run_dir.name.removeprefix("seed_"))
        except ValueError:
            pass
    if expected_stage is None and expected_config is not None:
        expected_stage = expected_config.stage
    if expected_seed is not None and metadata.get("seed") != expected_seed:
        raise ValueError("stored run seed does not match expected seed")
    if expected_stage is not None and metadata.get("stage") != expected_stage:
        raise ValueError("stored run stage does not match expected stage")
    with np.load(required[3], allow_pickle=False) as archive:
        if "states" not in archive:
            raise ValueError("stored run state archive has no states array")
        states = np.asarray(archive["states"], dtype=np.float64)
    if expected_state_shape is None and expected_config is not None:
        expected_state_shape = (expected_config.n_clients, expected_config.dimension)
    if expected_state_shape is not None and states.shape != expected_state_shape:
        raise ValueError(
            f"stored run state array shape {states.shape} does not match "
            f"expected {expected_state_shape}"
        )
    if not np.all(np.isfinite(states)):
        raise ValueError("stored run state array must contain finite numeric values")
    return RunResult(
        states=states,
        history=history,
        query_count=int(summary["query_count"]),
        transmitted_bytes=int(summary["transmitted_bytes"]),
        wall_time=float(summary["wall_time"]),
        metadata=dict(metadata),
    )


def aggregate_results(results: Iterable[RunResult]) -> Dict[str, np.ndarray]:
    items = list(results)
    if not items:
        raise ValueError("at least one result is required")
    aggregate: Dict[str, np.ndarray] = {}
    for name in items[0].history:
        stacked = np.vstack([item.history[name] for item in items])
        aggregate[f"{name}_mean"] = stacked.mean(axis=0)
        aggregate[f"{name}_std"] = stacked.std(axis=0)
    return aggregate


def save_aggregate(aggregate: Mapping[str, np.ndarray], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **aggregate)
    return path


def save_preprocessing(pca: PCAState, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mean=np.asarray(pca.mean),
        components=np.asarray(pca.components),
        scale=np.asarray(pca.scale),
    )
    return path
