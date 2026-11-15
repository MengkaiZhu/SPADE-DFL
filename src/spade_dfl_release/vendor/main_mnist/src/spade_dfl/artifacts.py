from __future__ import annotations

import csv
import json
import platform
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping

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
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git": "unavailable",
        "dataset": dict(dataset_metadata),
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    names = list(result.history)
    n_rows = max(len(result.history[name]) for name in names)
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
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
    np.savez_compressed(run_dir / "states.npz", states=result.states)
    summary = {
        "query_count": result.query_count,
        "gradient_evaluations": int(result.metadata.get("gradient_evaluations", 0)),
        "communication_rounds": int(result.metadata.get("communication_rounds", 0)),
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
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return run_dir


def load_run(
    run_dir: Path,
    *,
    expected_config: ExperimentConfig | None = None,
) -> RunResult:
    """Load a completed run artifact for deterministic suite resumption."""
    run_dir = Path(run_dir)
    required = (
        run_dir / "config.json",
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
    with required[1].open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "iteration" not in rows[0]:
        raise ValueError("run history is empty or malformed")
    history: Dict[str, np.ndarray] = {}
    for name in rows[0]:
        if name == "iteration":
            continue
        values = np.asarray(
            [float(row[name]) if row[name] != "" else np.nan for row in rows],
            dtype=np.float64,
        )
        while values.size and np.isnan(values[-1]):
            values = values[:-1]
        history[name] = values
    summary = json.loads(required[3].read_text(encoding="utf-8"))
    states = np.load(required[2])["states"]
    return RunResult(
        states=np.asarray(states, dtype=np.float64),
        history=history,
        query_count=int(summary["query_count"]),
        transmitted_bytes=int(summary["transmitted_bytes"]),
        wall_time=float(summary["wall_time"]),
        metadata=dict(summary["metadata"]),
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
