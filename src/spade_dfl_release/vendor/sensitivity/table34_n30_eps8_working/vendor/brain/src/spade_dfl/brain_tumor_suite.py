from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .artifacts import load_run, save_run
from .brain_tumor_baselines import apply_frozen_baseline_parameters
from .brain_tumor_contract import (
    brain_tumor_base_config,
    brain_tumor_method_config,
    partition_brain_tumor,
    validate_brain_tumor_method_set,
)
from .data import DatasetBundle
from .isolation import BRAIN_TUMOR_METHOD_IDS, assert_write_path
from .methods import run_method
from .trainer import RunResult


BRAIN_TUMOR_PANELS = ("dir_0.3", "dir_0.6", "iid", "pathological")
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]

# Internal method IDs retain the implementation-level two-point distinction.
# The visible names follow the comparison labels requested for the paper figure.
VISIBLE_METHOD_LABELS = {
    "zo2_lt_admm_vr": "LT-ADMM-VR",
    "zo2_lt_admm": "LT-ADMM",
    "zo2_dp_sgd": "DP-SGD",
    "zo2_dp_fedavg": "DP-FedAvg",
    "one_point_dsg": "1P-DSG",
    "one_point_dsgt": "1P-DSGT",
    "spade_dfl": "SPADE-DFL",
}

# Nature Communications-inspired, color-blind-readable blue/green/orange/red
# family. Only the presentation is borrowed; no ProxyFL data or parameters are.
FIGURE_STYLES = {
    "zo2_lt_admm_vr": ("#2166ac", "-", "o"),
    "zo2_lt_admm": ("#67a9cf", "--", "s"),
    "zo2_dp_sgd": ("#1b9e77", "-.", "^"),
    "zo2_dp_fedavg": ("#66c2a5", "--", "D"),
    "one_point_dsg": ("#e6ab02", "-", "P"),
    "one_point_dsgt": ("#f46d43", "-.", "X"),
    "spade_dfl": ("#d73027", "-", "v"),
}


def runtime_protocol_paths() -> tuple[Path, ...]:
    source_paths = tuple(
        path
        for path in (EXPERIMENT_ROOT / "src" / "spade_dfl").rglob("*.py")
        if "__pycache__" not in path.parts
    )
    protocol_paths = (
        EXPERIMENT_ROOT / "scripts" / "run_brain_tumor_suite.py",
        EXPERIMENT_ROOT / "scripts" / "run_resnet18_server_suite.py",
        EXPERIMENT_ROOT / "scripts" / "prepare_resnet18_features.py",
        EXPERIMENT_ROOT / "frozen" / "selected_parameters_robust.json",
        EXPERIMENT_ROOT / "frozen" / "baseline_selected_parameters.json",
        EXPERIMENT_ROOT / "configs" / "brain_tumor_50round_server.json",
        EXPERIMENT_ROOT / "configs" / "resnet18_smoke.json",
        EXPERIMENT_ROOT / "configs" / "resnet18_server.json",
        EXPERIMENT_ROOT / "requirements.txt",
        EXPERIMENT_ROOT / "requirements_resnet18_cpu.txt",
        EXPERIMENT_ROOT / "pyproject.toml",
        EXPERIMENT_ROOT / "run_server.sh",
        EXPERIMENT_ROOT / "run_server.ps1",
    )
    paths = tuple(sorted(source_paths + protocol_paths))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "runtime protocol input is missing: "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def runtime_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in runtime_protocol_paths():
        relative = path.relative_to(EXPERIMENT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def aggregate_accuracy(results: Iterable[RunResult]) -> dict[str, np.ndarray]:
    items = list(results)
    if not items:
        raise ValueError("at least one run is required")
    curves = [np.asarray(item.history["accuracy"], dtype=np.float64) for item in items]
    lengths = {curve.size for curve in curves}
    if len(lengths) != 1:
        raise ValueError("all accuracy curves must have the same length")
    stacked = np.vstack(curves)
    return {
        "round": np.arange(stacked.shape[1], dtype=np.float64),
        "accuracy_mean": stacked.mean(axis=0),
        "accuracy_std": stacked.std(axis=0, ddof=0),
    }


def validate_accuracy_aggregate(
    aggregate: Mapping[str, np.ndarray], *, n_rounds: int
) -> None:
    required = ("round", "accuracy_mean", "accuracy_std")
    if any(name not in aggregate for name in required):
        raise ValueError("accuracy aggregate is missing required arrays")
    arrays = [np.asarray(aggregate[name], dtype=np.float64) for name in required]
    if any(array.shape != (n_rounds + 1,) for array in arrays):
        raise ValueError("accuracy aggregate does not match the configured horizon")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("accuracy aggregate must contain only finite values")
    if np.any(arrays[1] < 0.0) or np.any(arrays[1] > 1.0):
        raise ValueError("mean accuracy must lie in [0, 1]")
    if np.any(arrays[2] < 0.0):
        raise ValueError("accuracy standard deviation cannot be negative")


def common_accuracy_ylim(
    all_aggregates: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    padding: float = 0.03,
    quantum: float = 0.05,
) -> tuple[float, float]:
    """Return compact common y-axis bounds covering every accuracy shadow."""
    if not np.isfinite(padding) or padding <= 0.0:
        raise ValueError("padding must be finite and positive")
    if not np.isfinite(quantum) or quantum <= 0.0:
        raise ValueError("quantum must be finite and positive")

    lower_min = np.inf
    upper_max = -np.inf
    found = False
    for methods in all_aggregates.values():
        for aggregate in methods.values():
            if "accuracy_mean" not in aggregate or "accuracy_std" not in aggregate:
                raise ValueError("accuracy aggregate is missing required arrays")
            mean = np.asarray(aggregate["accuracy_mean"], dtype=np.float64)
            std = np.asarray(aggregate["accuracy_std"], dtype=np.float64)
            if mean.size == 0 or mean.shape != std.shape:
                raise ValueError("accuracy aggregate has invalid limits")
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
                raise ValueError("accuracy aggregate must contain only finite values")
            if np.any(mean < 0.0) or np.any(mean > 1.0) or np.any(std < 0.0):
                raise ValueError("accuracy aggregate has invalid limits")
            lower_min = min(lower_min, float(np.min(mean - std)))
            upper_max = max(upper_max, float(np.max(mean + std)))
            found = True

    if not found:
        raise ValueError("at least one accuracy aggregate is required")
    scaled_lower = (lower_min - padding) / quantum
    scaled_upper = (upper_max + padding) / quantum
    lower = max(0.0, float(np.floor(scaled_lower + 1e-12) * quantum))
    upper = min(1.0, float(np.ceil(scaled_upper - 1e-12) * quantum))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("accuracy limits are invalid")
    return lower, upper


def _panel_partition_parameters(panel: str) -> tuple[str, float]:
    mapping = {
        "dir_0.3": ("dirichlet", 0.3),
        "dir_0.6": ("dirichlet", 0.6),
        "iid": ("iid", 0.3),
        "pathological": ("pathological_shards", 0.3),
    }
    try:
        return mapping[panel]
    except KeyError as error:
        raise ValueError(f"unknown Brain Tumor panel: {panel}") from error


def load_frozen_selection(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"eta0", "rho", "batch_size", "n_rounds", "selection_status"}
    if not required.issubset(payload):
        raise ValueError("frozen selection is incomplete")
    if payload["selection_status"] != "selected":
        raise ValueError("frozen selection was not accepted")
    if bool(payload.get("official_test_used_for_selection", True)):
        raise ValueError("official test split must not be used for parameter selection")
    return payload


def load_frozen_baselines(
    path: Path,
) -> dict[str, dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("selection_status") != "selected":
        raise ValueError("baseline parameter selection was not accepted")
    if bool(payload.get("official_test_used_for_selection", True)):
        raise ValueError("official test split must not select baseline parameters")
    selected = payload.get("selected")
    required = {
        "zo2_lt_admm_vr",
        "zo2_lt_admm",
        "zo2_dp_sgd",
        "zo2_dp_fedavg",
    }
    if not isinstance(selected, dict) or set(selected) != required:
        raise ValueError("frozen baseline selection is incomplete")
    return {str(key): dict(value) for key, value in selected.items()}


def _dataset_metadata(dataset: DatasetBundle) -> dict[str, object]:
    return {
        "name": dataset.dataset_name,
        "sha256": dataset.sha256,
        "source_path": dataset.source_path,
        "train_count": int(dataset.train_x.shape[0]),
        "test_count": int(dataset.test_x.shape[0]),
        "pca_dimension": int(dataset.pca_dimension),
        "class_names": list(dataset.class_names),
        "official_test_role": "evaluation_only",
        "source_tree_sha256": runtime_source_sha256(),
    }


def run_brain_tumor_suite(
    dataset: DatasetBundle,
    *,
    frozen_selection: Mapping[str, object],
    frozen_baselines: Mapping[str, Mapping[str, object]],
    output_root: Path,
    seeds: Sequence[int],
    n_rounds: int | None = None,
    panels: Sequence[str] = BRAIN_TUMOR_PANELS,
    methods: Sequence[str] = BRAIN_TUMOR_METHOD_IDS,
    resume: bool = True,
    write_figures: bool = True,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    validate_brain_tumor_method_set(methods)
    if tuple(panels) != BRAIN_TUMOR_PANELS:
        raise ValueError("the complete suite requires the exact four-panel contract")
    if not seeds:
        raise ValueError("at least one seed is required")
    output_root = assert_write_path(Path(output_root), EXPERIMENT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    rounds = int(n_rounds or frozen_selection["n_rounds"])
    base = brain_tumor_base_config(
        n_rounds=rounds,
        seeds=tuple(int(seed) for seed in seeds),
        eta0=float(frozen_selection["eta0"]),
        rho=float(frozen_selection["rho"]),
        batch_size=int(frozen_selection["batch_size"]),
    )
    base = replace(
        base,
        data_path=dataset.source_path,
        output_dir=str(output_root),
    )
    all_aggregates: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    final_rows: list[dict[str, object]] = []
    metadata = _dataset_metadata(dataset)
    for panel in panels:
        mode, alpha = _panel_partition_parameters(panel)
        method_aggregates: dict[str, dict[str, np.ndarray]] = {}
        for method in methods:
            config = brain_tumor_method_config(
                base,
                method,
                tuple(int(seed) for seed in seeds),
                mode,
                alpha,
            )
            config = apply_frozen_baseline_parameters(
                config,
                method,
                frozen_baselines,
            )
            results: list[RunResult] = []
            for seed in seeds:
                run_dir = (
                    output_root
                    / "raw"
                    / panel
                    / config.stage
                    / f"seed_{int(seed)}"
                )
                if resume and (run_dir / "COMPLETED").is_file():
                    result = load_run(
                        run_dir,
                        expected_config=config,
                        expected_dataset_sha256=dataset.sha256,
                        expected_seed=int(seed),
                        expected_stage=config.stage,
                        expected_history_length=rounds + 1,
                        expected_state_shape=(config.n_clients, config.dimension),
                        expected_source_sha256=metadata["source_tree_sha256"],
                    )
                else:
                    partition = partition_brain_tumor(
                        dataset.train_y,
                        base.n_clients,
                        panel,
                        seed=int(seed),
                    )
                    result = run_method(
                        method,
                        config,
                        dataset,
                        partition,
                        int(seed),
                    )
                    save_run(
                        result,
                        config,
                        output_root / "raw" / panel,
                        metadata,
                    )
                if result.history["accuracy"].shape != (rounds + 1,):
                    raise ValueError(
                        f"{panel}/{method}/seed {seed} has an invalid accuracy horizon"
                    )
                if not np.all(np.isfinite(result.history["accuracy"])):
                    raise FloatingPointError(
                        f"{panel}/{method}/seed {seed} produced non-finite accuracy"
                    )
                results.append(result)
            aggregate = aggregate_accuracy(results)
            validate_accuracy_aggregate(aggregate, n_rounds=rounds)
            method_aggregates[method] = aggregate
            final_rows.append(
                {
                    "panel": panel,
                    "method_id": method,
                    "visible_method": VISIBLE_METHOD_LABELS[method],
                    "seeds": len(seeds),
                    "rounds": rounds,
                    "final_accuracy_mean": float(aggregate["accuracy_mean"][-1]),
                    "final_accuracy_std": float(aggregate["accuracy_std"][-1]),
                    "gradient_oracle": (
                        "one_point"
                        if method in {"one_point_dsg", "one_point_dsgt", "spade_dfl"}
                        else "symmetric_two_point"
                    ),
                }
            )
            aggregate_path = output_root / "aggregates" / panel / f"{method}.npz"
            aggregate_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(aggregate_path, **aggregate)
        all_aggregates[panel] = method_aggregates
    _write_final_summary(output_root / "final_accuracy.csv", final_rows)
    if write_figures:
        write_accuracy_figures(all_aggregates, output_root / "figures", n_rounds=rounds)
    return all_aggregates


def _write_final_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_accuracy_figures(
    aggregates: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    output_dir: Path,
    *,
    n_rounds: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, PercentFormatter

    output_dir.mkdir(parents=True, exist_ok=True)
    titles = {
        "dir_0.3": r"Dirichlet non-IID ($\alpha=0.3$)",
        "dir_0.6": r"Dirichlet non-IID ($\alpha=0.6$)",
        "iid": "IID",
        "pathological": "Pathological non-IID (two shards/client)",
    }
    marker_every = max(1, n_rounds // 10)
    y_limits = common_accuracy_ylim(aggregates)
    for panel in BRAIN_TUMOR_PANELS:
        figure, axis = plt.subplots(figsize=(8.2, 5.1), constrained_layout=True)
        for method in BRAIN_TUMOR_METHOD_IDS:
            aggregate = aggregates[panel][method]
            validate_accuracy_aggregate(aggregate, n_rounds=n_rounds)
            rounds = np.asarray(aggregate["round"])
            mean = np.asarray(aggregate["accuracy_mean"])
            std = np.asarray(aggregate["accuracy_std"])
            color, linestyle, marker = FIGURE_STYLES[method]
            axis.plot(
                rounds,
                mean,
                label=VISIBLE_METHOD_LABELS[method],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=marker_every,
                linewidth=2.0 if method != "spade_dfl" else 2.7,
                markersize=5.0,
                zorder=4 if method == "spade_dfl" else 3,
            )
            axis.fill_between(
                rounds,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.0, 1.0),
                color=color,
                alpha=0.14,
                linewidth=0,
                zorder=1,
            )
        axis.set_xlim(0, n_rounds)
        axis.set_ylim(*y_limits)
        axis.set_xlabel("Communication rounds")
        axis.set_ylabel("Test accuracy")
        axis.yaxis.set_major_locator(MultipleLocator(0.05))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.set_title(titles[panel])
        axis.grid(True, color="#b0b0b0", alpha=0.45, linewidth=0.8)
        axis.text(
            0.012,
            0.015,
            "DP-SGD: aligned outer optimization step (no network communication)",
            transform=axis.transAxes,
            fontsize=7.2,
            color="#555555",
        )
        axis.legend(
            loc="lower right",
            frameon=True,
            framealpha=0.94,
            edgecolor="#cccccc",
            fontsize=8.5,
            ncol=2,
        )
        for suffix in ("png", "pdf", "svg"):
            figure.savefig(
                output_dir / f"{panel}_accuracy.{suffix}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(figure)
