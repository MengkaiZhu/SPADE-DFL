from __future__ import annotations

import csv
import hashlib
import json
import warnings
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .alzheimer_contract import (
    ALZHEIMER_METHOD_IDS,
    validate_alzheimer_method_contract,
)
from .alzheimer_data import (
    build_final_split,
    build_validation_split,
    save_alzheimer_preprocessor,
)
from .alzheimer_frozen_config import (
    FrozenAlzheimerSuiteConfig,
    apply_frozen_method_parameters,
    load_frozen_alzheimer_suite_config,
)
from .alzheimer_v2_frozen import (
    FrozenAlzheimerSuiteConfig as V2FrozenAlzheimerSuiteConfig,
)
from .alzheimer_v2_frozen import (
    apply_v2_method_parameters,
    load_v2_suite_config,
)
from .artifacts import load_run, save_run
from .config import ExperimentConfig
from .methods import method_config, run_method
from .partitioning import PartitionResult, partition_labels
from .trainer import RunResult

ALZHEIMER_PARTITIONS = ("dir_0.3", "dir_0.6", "iid", "pathological")
ALZHEIMER_PARTITION_SPECS = {
    "dir_0.3": ("dirichlet", 0.3),
    "dir_0.6": ("dirichlet", 0.6),
    "iid": ("iid", 0.3),
    "pathological": ("pathological", 0.3),
}

DISPLAY_LABELS = {
    "zo2_lt_admm_vr": "LT-ADMM-VR",
    "zo2_lt_admm": "LT-ADMM",
    "zo2_dp_sgd": "DP-SGD",
    "zo2_dp_fedavg": "DP-FedAvg",
    "one_point_dsg": "1P-DSG",
    "one_point_dsgt": "1P-DSGT",
    "spade_dfl": "SPADE-DFL",
}

# Fixed, color-blind-readable presentation palette. It controls presentation
# only; no algorithm parameter is derived from the visual reference.
FIGURE_STYLES = {
    "zo2_lt_admm_vr": ("#2166ac", "-", "o"),
    "zo2_lt_admm": ("#67a9cf", "--", "s"),
    "zo2_dp_sgd": ("#1b9e77", "-.", "^"),
    "zo2_dp_fedavg": ("#66c2a5", "--", "D"),
    "one_point_dsg": ("#e6ab02", "-", "P"),
    "one_point_dsgt": ("#f46d43", "-.", "X"),
    "spade_dfl": ("#d73027", "-", "v"),
}
FIGURE_LINEWIDTHS = {
    method: 3.0 if method == "spade_dfl" else 1.8
    for method in ALZHEIMER_METHOD_IDS
}

_SUITE_CONFIG_FIELDS = {
    "experiment",
    "methods",
    "partitions",
    "split_seed",
}


@dataclass(frozen=True)
class AlzheimerSuiteConfig:
    experiment: ExperimentConfig
    methods: tuple[str, ...]
    partitions: tuple[str, ...]
    split_seed: int


def _validate_partition_contract(partitions: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(partitions)
    if not selected:
        raise ValueError("at least one canonical Alzheimer partition is required")
    unknown = [partition for partition in selected if partition not in ALZHEIMER_PARTITIONS]
    if unknown:
        raise ValueError(
            "unknown partition; canonical Alzheimer partitions are "
            + ", ".join(ALZHEIMER_PARTITIONS)
        )
    canonical_subsequence = tuple(
        partition for partition in ALZHEIMER_PARTITIONS if partition in selected
    )
    if selected != canonical_subsequence:
        raise ValueError(
            "selected Alzheimer partitions must be unique and in canonical order"
        )
    return selected


def _resolve_data_path(config_path: Path, experiment: ExperimentConfig) -> ExperimentConfig:
    data_path = Path(str(experiment.data_path))
    if not data_path.is_absolute():
        data_path = config_path.parent / data_path
    return replace(experiment, data_path=str(data_path.resolve()))


def _validate_experiment(experiment: ExperimentConfig) -> None:
    if experiment.dataset_name != "alzheimer":
        raise ValueError("Alzheimer suite experiment dataset_name must be alzheimer")
    if experiment.dimension != 39 or experiment.feature_dimension != 39:
        raise ValueError("Alzheimer suite requires the exact 39-dimensional encoding")
    if experiment.task_type != "binary" or experiment.n_classes != 2:
        raise ValueError("Alzheimer suite requires the binary two-class task")
    if (experiment.negative_class, experiment.positive_class) != (0, 1):
        raise ValueError("Alzheimer suite Diagnosis classes must be ordered 0 then 1")
    if experiment.n_clients != 31:
        raise ValueError("Alzheimer suite requires the paper-aligned 31 clients")
    if not experiment.seeds:
        raise ValueError("Alzheimer suite requires at least one seed")
    if len({int(seed) for seed in experiment.seeds}) != len(experiment.seeds):
        raise ValueError("Alzheimer suite seeds must be unique")
    if any(int(seed) < 0 for seed in experiment.seeds):
        raise ValueError("Alzheimer suite seeds must be nonnegative integers")


def load_alzheimer_suite_config(
    path: Path, *, allow_legacy_fixture: bool = False
) -> FrozenAlzheimerSuiteConfig | V2FrozenAlzheimerSuiteConfig:
    """Load the shared frozen Alzheimer production contract."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    protocol_version = raw.get("protocol_version")
    if protocol_version == "alzheimer_v2":
        suite = load_v2_suite_config(path)
        return replace(suite, experiment=_resolve_data_path(path.resolve(), suite.experiment))
    if protocol_version is not None:
        raise ValueError(f"unsupported Alzheimer protocol_version {protocol_version!r}")
    if set(raw) != _SUITE_CONFIG_FIELDS:
        return load_frozen_alzheimer_suite_config(path)
    if not allow_legacy_fixture:
        raise ValueError("legacy fixture requires explicit allow_legacy_fixture=True")
    experiment = _resolve_data_path(path.resolve(), ExperimentConfig.from_dict(raw["experiment"]))
    _validate_experiment(experiment)
    methods = tuple(str(method) for method in raw["methods"])
    validate_alzheimer_method_contract(methods)
    return FrozenAlzheimerSuiteConfig(
        experiment=experiment,
        methods=methods,
        partitions=_validate_partition_contract(raw["partitions"]),
        split_seed=raw["split_seed"],
        protocol_role="legacy_scientific",
        evaluation_scope="outer_final_test_frozen",
        output_label="legacy_alzheimer_suite",
        method_parameters={},
        parameter_sources={},
        selection_provenance={},
        eligible_for_final_plots=True,
    )


def _is_v2_suite(
    suite: FrozenAlzheimerSuiteConfig | V2FrozenAlzheimerSuiteConfig,
) -> bool:
    return isinstance(suite, V2FrozenAlzheimerSuiteConfig)


def _suite_eligible_for_final_plots(
    suite: FrozenAlzheimerSuiteConfig | V2FrozenAlzheimerSuiteConfig,
) -> bool:
    if _is_v2_suite(suite):
        return suite.protocol_role != "package_smoke_only"
    return suite.eligible_for_final_plots


def _suite_selection_lock_sha256(
    suite: FrozenAlzheimerSuiteConfig | V2FrozenAlzheimerSuiteConfig,
) -> str:
    if _is_v2_suite(suite):
        return str(suite.selection_provenance["selection_lock_sha256"])
    return str(suite.selection_provenance.get("reviewed_selection_lock_sha256", ""))


def _apply_v2_final_method_parameters(
    config: ExperimentConfig, *, suite: V2FrozenAlzheimerSuiteConfig, method: str
) -> ExperimentConfig:
    """Restore the frozen common v2 protocol after method defaults are applied."""
    common_names = (
        "objective_model",
        "regularization",
        "sigma_u",
        "query_noise_std",
        "initialization",
        "tau",
        "clip_norm",
    )
    sources = dict(config.parameter_sources)
    for name in common_names:
        sources[name] = suite.experiment.parameter_sources[name]
    common = {
        name: getattr(suite.experiment, name)
        for name in common_names
    }
    common.update(shared_initialization=False, beta=0.0)
    return apply_v2_method_parameters(
        replace(config, parameter_sources=sources, **common), suite=suite, method=method
    )


def _partition_parameters(partition: str) -> tuple[str, float]:
    try:
        return ALZHEIMER_PARTITION_SPECS[partition]
    except KeyError as error:
        raise ValueError(f"unknown canonical Alzheimer partition {partition!r}") from error


def _build_partition(
    labels: np.ndarray,
    *,
    n_clients: int,
    partition: str,
    seed: int,
) -> PartitionResult:
    mode, alpha = _partition_parameters(partition)
    return partition_labels(
        labels,
        n_clients,
        mode,
        np.random.default_rng(seed),
        alpha=alpha,
    )


def _dataset_metadata(
    *,
    dataset_sha256: str,
    dataset_path: str,
    train_count: int,
    test_count: int,
    partition_name: str,
    partition: PartitionResult,
    seed: int,
) -> dict[str, object]:
    return {
        "name": "alzheimer",
        "sha256": dataset_sha256,
        "source_path": dataset_path,
        "train_count": train_count,
        "test_count": test_count,
        "dimension": 39,
        "preprocessing": "standardize_plus_one_hot_no_pca",
        "partition": partition_name,
        "partition_mode": partition.mode,
        "dirichlet_alpha": partition.alpha,
        "partition_seed": seed,
        "label_histograms": partition.label_histograms.tolist(),
    }


def _load_completed_run(
    run_dir: Path,
    *,
    config: ExperimentConfig,
    dataset_sha256: str,
    partition_name: str,
    partition: PartitionResult,
    seed: int,
) -> RunResult | None:
    try:
        return load_run(
            run_dir,
            expected_config=config,
            expected_dataset_sha256=dataset_sha256,
            expected_partition=partition_name,
            expected_label_histograms=partition.label_histograms,
            expected_seed=seed,
            expected_stage=config.stage,
            expected_history_length=config.n_rounds + 1,
            expected_state_shape=(config.n_clients, config.dimension),
        )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ):
        return None


def _write_v2_raw_manifests(
    run_dir: Path, *, config: ExperimentConfig, result: RunResult
) -> None:
    """Attach versioned provenance without changing the persisted run config."""
    environment_path = Path(run_dir) / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["protocol_version"] = "alzheimer_v2"
    environment_path.write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    raw_manifest = {
        "status": "completed",
        "protocol_version": "alzheimer_v2",
        "stage": config.stage,
        "seed": result.metadata.get("seed"),
        "tau": config.tau,
        "gradient_oracle": result.metadata.get("gradient_oracle"),
        "config_sha256": hashlib.sha256((Path(run_dir) / "config.json").read_bytes()).hexdigest(),
    }
    (Path(run_dir) / "raw_manifest.json").write_text(
        json.dumps(raw_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _reject_smoke_provenance(
    result_provenance: Mapping[str, object] | None, *, allow_smoke: bool
) -> None:
    if (
        result_provenance is not None
        and result_provenance.get("protocol_role") == "package_smoke_only"
        and not allow_smoke
    ):
        raise ValueError("formal aggregation rejects package smoke provenance")


def _root_provenance(root: Path) -> Mapping[str, object]:
    manifest = json.loads((Path(root) / "suite_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("suite manifest must be a JSON object")
    return manifest


def _validated_root_provenance(
    result_root: Path, *, allow_smoke: bool
) -> Mapping[str, object]:
    result_root = Path(result_root)
    manifest = _root_provenance(result_root)
    _reject_smoke_provenance(manifest, allow_smoke=allow_smoke)
    required = {
        "status",
        "config_path",
        "config_sha256",
        "reviewed_selection_lock_sha256",
        "methods",
        "partitions",
        "seeds",
        "rounds",
        "protocol_role",
        "evaluation_scope",
        "eligible_for_final_plots",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError("suite manifest is missing " + ", ".join(missing))
    if manifest["status"] != "completed":
        raise ValueError("suite manifest must record completed raw runs")
    if not isinstance(manifest["eligible_for_final_plots"], bool):
        raise TypeError("suite manifest plot eligibility must be boolean")

    config_path = Path(str(manifest["config_path"]))
    if not config_path.is_file():
        raise FileNotFoundError("suite manifest frozen config does not exist")
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if manifest["config_sha256"] != config_sha256:
        raise ValueError("suite manifest frozen config SHA-256 does not match")
    suite = load_alzheimer_suite_config(config_path)
    is_v2 = _is_v2_suite(suite)
    if is_v2 and manifest.get("protocol_version") != "alzheimer_v2":
        raise ValueError("v2 suite manifest has no Alzheimer protocol version")
    if not is_v2 and "protocol_version" in manifest:
        raise ValueError("v1 suite manifest must not claim a v2 protocol version")
    methods = tuple(str(method) for method in manifest["methods"])
    validate_alzheimer_method_contract(methods)
    partitions = _validate_partition_contract(manifest["partitions"])
    raw_seeds = manifest["seeds"]
    if not isinstance(raw_seeds, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds
    ):
        raise TypeError("suite manifest seeds must be a JSON integer list")
    seeds = tuple(raw_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("suite manifest seeds must be unique nonnegative integers")
    rounds = manifest["rounds"]
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("suite manifest rounds must be a positive integer")

    expected = {
        "protocol_role": suite.protocol_role,
        "evaluation_scope": suite.evaluation_scope,
        "eligible_for_final_plots": _suite_eligible_for_final_plots(suite),
        "methods": suite.methods,
        "partitions": suite.partitions,
        "seeds": tuple(int(seed) for seed in suite.experiment.seeds),
        "rounds": suite.experiment.n_rounds,
        "reviewed_selection_lock_sha256": _suite_selection_lock_sha256(suite),
    }
    actual = {
        "protocol_role": manifest["protocol_role"],
        "evaluation_scope": manifest["evaluation_scope"],
        "eligible_for_final_plots": manifest["eligible_for_final_plots"],
        "methods": methods,
        "partitions": partitions,
        "seeds": seeds,
        "rounds": rounds,
        "reviewed_selection_lock_sha256": manifest[
            "reviewed_selection_lock_sha256"
        ],
    }
    if actual != expected:
        raise ValueError("suite manifest does not match its frozen configuration")
    return manifest


def _aggregate_run_directories(
    run_dirs: Iterable[Path],
    *,
    expected_stage: str,
    expected_history_length: int,
) -> dict[str, np.ndarray]:
    results = [
        load_run(
            Path(run_dir),
            expected_stage=expected_stage,
            expected_history_length=expected_history_length,
        )
        for run_dir in run_dirs
    ]
    if not results:
        raise ValueError("at least one completed raw Alzheimer run is required")
    history_names = tuple(results[0].history)
    if any(tuple(result.history) != history_names for result in results[1:]):
        raise ValueError("raw Alzheimer histories have inconsistent columns")

    aggregate: dict[str, np.ndarray] = {}
    expected_length = None
    for name in history_names:
        curves = [np.asarray(result.history[name], dtype=np.float64) for result in results]
        lengths = {curve.size for curve in curves}
        if len(lengths) != 1:
            raise ValueError("raw Alzheimer histories have inconsistent lengths")
        length = curves[0].size
        if expected_length is None:
            expected_length = length
            aggregate["round"] = np.arange(length, dtype=np.int64)
        elif length != expected_length:
            raise ValueError("raw Alzheimer history metrics have inconsistent lengths")
        stacked = np.vstack(curves)
        aggregate[f"{name}_mean"] = stacked.mean(axis=0)
        aggregate[f"{name}_std"] = stacked.std(axis=0, ddof=0)
    return aggregate


def _write_aggregate_manifest(
    result_root: Path,
    *,
    partition: str,
    provenance: Mapping[str, object],
) -> None:
    result_root = Path(result_root)
    manifest_path = result_root / "aggregates" / "aggregate_manifest.json"
    suite_sha256 = hashlib.sha256(
        (result_root / "suite_manifest.json").read_bytes()
    ).hexdigest()
    aggregate_manifest: dict[str, object] = {
        "status": "completed",
        "source_suite_manifest_sha256": suite_sha256,
        "protocol_role": provenance["protocol_role"],
        "eligible_for_final_plots": provenance["eligible_for_final_plots"],
        "partitions": {},
    }
    if provenance.get("protocol_version") == "alzheimer_v2":
        aggregate_manifest["protocol_version"] = "alzheimer_v2"
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            existing = None
        if (
            isinstance(existing, dict)
            and existing.get("source_suite_manifest_sha256") == suite_sha256
            and existing.get("protocol_role") == provenance["protocol_role"]
            and existing.get("eligible_for_final_plots")
            == provenance["eligible_for_final_plots"]
            and existing.get("protocol_version")
            == aggregate_manifest.get("protocol_version")
            and isinstance(existing.get("partitions"), dict)
        ):
            aggregate_manifest = existing
    artifact_records = {}
    for method in provenance["methods"]:
        relative_path = Path("aggregates") / partition / f"{method}.npz"
        artifact_path = result_root / relative_path
        artifact_records[str(method)] = {
            "path": relative_path.as_posix(),
            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        }
    aggregate_manifest["partitions"][partition] = {
        "methods": list(provenance["methods"]),
        "seeds": list(provenance["seeds"]),
        "artifacts": artifact_records,
    }
    manifest_path.write_text(
        json.dumps(aggregate_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def aggregate_alzheimer_runs(
    result_root: Path,
    *,
    partition: str,
    allow_smoke: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate the manifest-declared raw runs under one intrinsic result root."""
    result_root = Path(result_root)
    provenance = _validated_root_provenance(
        result_root,
        allow_smoke=allow_smoke,
    )
    _validate_partition_contract((partition,))
    if partition not in provenance["partitions"]:
        raise ValueError("requested partition is absent from the suite manifest")
    methods = tuple(str(method) for method in provenance["methods"])
    seeds = tuple(int(seed) for seed in provenance["seeds"])
    expected_history_length = int(provenance["rounds"]) + 1
    aggregates: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        run_dirs = [
            result_root / "raw" / partition / method / f"seed_{seed}"
            for seed in seeds
        ]
        aggregate = _aggregate_run_directories(
            run_dirs,
            expected_stage=method,
            expected_history_length=expected_history_length,
        )
        aggregates[method] = aggregate
        aggregate_path = result_root / "aggregates" / partition / f"{method}.npz"
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(aggregate_path, **aggregate)
    _write_aggregate_manifest(
        result_root,
        partition=partition,
        provenance=provenance,
    )
    return aggregates


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validated_aggregate_provenance(
    result_root: Path,
    *,
    partition: str,
    suite_provenance: Mapping[str, object],
) -> Mapping[str, object]:
    result_root = Path(result_root)
    manifest_path = result_root / "aggregates" / "aggregate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("aggregate manifest must be a JSON object")
    expected_suite_sha256 = hashlib.sha256(
        (result_root / "suite_manifest.json").read_bytes()
    ).hexdigest()
    if (
        manifest.get("status") != "completed"
        or manifest.get("source_suite_manifest_sha256") != expected_suite_sha256
        or manifest.get("protocol_role") != suite_provenance["protocol_role"]
        or manifest.get("eligible_for_final_plots")
        != suite_provenance["eligible_for_final_plots"]
    ):
        raise ValueError("aggregate manifest does not match suite provenance")
    if suite_provenance.get("protocol_version") == "alzheimer_v2" and (
        manifest.get("protocol_version") != "alzheimer_v2"
    ):
        raise ValueError("v2 aggregate manifest has no Alzheimer protocol version")
    partitions = manifest.get("partitions")
    if not isinstance(partitions, dict):
        raise TypeError("aggregate manifest partitions must be a JSON object")
    expected_partitions = tuple(str(name) for name in suite_provenance["partitions"])
    if set(partitions) != set(expected_partitions):
        raise ValueError(
            "aggregate manifest is incomplete; aggregate every declared "
            "partition before creating figures"
        )
    if partition not in partitions:
        raise ValueError("aggregate manifest has no requested partition")
    record = partitions[partition]
    if not isinstance(record, dict):
        raise TypeError("aggregate manifest partition record must be a JSON object")
    if (
        tuple(record.get("methods", ())) != tuple(suite_provenance["methods"])
        or tuple(record.get("seeds", ())) != tuple(suite_provenance["seeds"])
    ):
        raise ValueError("aggregate manifest scope does not match suite provenance")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("aggregate manifest artifacts must be a JSON object")
    for method in suite_provenance["methods"]:
        artifact = artifacts.get(method)
        relative_path = Path("aggregates") / partition / f"{method}.npz"
        if (
            not isinstance(artifact, dict)
            or artifact.get("path") != relative_path.as_posix()
        ):
            raise ValueError("aggregate manifest artifact path is invalid")
        artifact_path = result_root / relative_path
        if (
            not artifact_path.is_file()
            or hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            != artifact.get("sha256")
        ):
            raise ValueError("aggregate manifest artifact SHA-256 does not match")
    return manifest


def _plot_aggregate_arrays(
    partition: str,
    aggregates: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: Path,
) -> Path:
    """Render one raw-mean accuracy figure from already validated arrays."""
    _validate_partition_contract((partition,))
    if tuple(aggregates) != ALZHEIMER_METHOD_IDS:
        raise ValueError("accuracy figure requires the exact ordered seven methods")

    figure_dir = Path(output_dir) / partition
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict[str, object]] = []
    for method in ALZHEIMER_METHOD_IDS:
        aggregate = aggregates[method]
        rounds = np.asarray(aggregate["round"])
        mean = np.asarray(aggregate["accuracy_mean"])
        std = np.asarray(aggregate["accuracy_std"])
        if rounds.shape != mean.shape or mean.shape != std.shape:
            raise ValueError("accuracy aggregate arrays must have identical shapes")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("accuracy aggregate must contain finite values")
        for round_index, mean_value, std_value in zip(rounds, mean, std):
            source_rows.append(
                {
                    "method_id": method,
                    "method": DISPLAY_LABELS[method],
                    "round": int(round_index),
                    "mean": float(mean_value),
                    "std": float(std_value),
                }
            )
    _write_csv(
        figure_dir / "accuracy.source_data.csv",
        ("method_id", "method", "round", "mean", "std"),
        source_rows,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter

    titles = {
        "dir_0.3": r"Dirichlet non-IID ($\alpha=0.3$)",
        "dir_0.6": r"Dirichlet non-IID ($\alpha=0.6$)",
        "iid": "IID",
        "pathological": "Pathological non-IID",
    }
    figure, axis = plt.subplots(figsize=(8.2, 5.1), constrained_layout=True)
    for method in ALZHEIMER_METHOD_IDS:
        aggregate = aggregates[method]
        rounds = np.asarray(aggregate["round"])
        mean = np.asarray(aggregate["accuracy_mean"])
        std = np.asarray(aggregate["accuracy_std"])
        color, linestyle, marker = FIGURE_STYLES[method]
        axis.plot(
            rounds,
            mean,
            label=DISPLAY_LABELS[method],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=max(1, len(rounds) // 10),
            linewidth=FIGURE_LINEWIDTHS[method],
            markersize=5.0,
            zorder=4 if method == "spade_dfl" else 3,
        )
        axis.fill_between(
            rounds,
            mean - std,
            mean + std,
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )
    axis.set_xlim(0.0, float(np.max(rounds)))
    axis.set_xlabel("Communication round")
    axis.set_ylabel("Test accuracy")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set_title(titles[partition])
    axis.grid(True, color="#b0b0b0", alpha=0.45, linewidth=0.8)
    axis.legend(
        loc="best",
        frameon=True,
        framealpha=0.94,
        edgecolor="#cccccc",
        fontsize=8.5,
        ncol=2,
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            figure_dir / f"accuracy.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)
    (figure_dir / "accuracy.caption.txt").write_text(
        "Curves show the raw per-round seed mean; shadows show mean ± population "
        "standard deviation (ddof=0). No smoothing, interpolation, cumulative "
        "maximum, clipping, ranking manipulation, or curve post-processing was "
        "applied. DP-SGD x values are aligned outer optimization steps and do "
        "not represent network communication events.\n",
        encoding="utf-8",
    )
    return figure_dir


def plot_alzheimer_accuracy(
    result_root: Path,
    *,
    partition: str,
    allow_smoke: bool = False,
) -> Path:
    """Aggregate and plot one manifest-declared result-root partition."""
    result_root = Path(result_root)
    provenance = _validated_root_provenance(
        result_root,
        allow_smoke=allow_smoke,
    )
    if partition not in provenance["partitions"]:
        raise ValueError("requested partition is absent from the suite manifest")
    all_aggregates = {
        declared_partition: aggregate_alzheimer_runs(
            result_root,
            partition=str(declared_partition),
            allow_smoke=allow_smoke,
        )
        for declared_partition in provenance["partitions"]
    }
    aggregates = all_aggregates[partition]
    _validated_aggregate_provenance(
        result_root,
        partition=partition,
        suite_provenance=provenance,
    )
    figure_dir = _plot_aggregate_arrays(
        partition,
        aggregates,
        result_root / "figures",
    )
    artifacts = {}
    for filename in (
        "accuracy.png",
        "accuracy.pdf",
        "accuracy.svg",
        "accuracy.source_data.csv",
    ):
        path = figure_dir / filename
        artifacts[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    (figure_dir / "figure_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "source_suite_manifest_sha256": hashlib.sha256(
                    (result_root / "suite_manifest.json").read_bytes()
                ).hexdigest(),
                "source_aggregate_manifest_sha256": hashlib.sha256(
                    (
                        result_root
                        / "aggregates"
                        / "aggregate_manifest.json"
                    ).read_bytes()
                ).hexdigest(),
                "partition": partition,
                "protocol_role": provenance["protocol_role"],
                "eligible_for_final_plots": provenance[
                    "eligible_for_final_plots"
                ],
                "artifacts": artifacts,
                **(
                    {"protocol_version": "alzheimer_v2"}
                    if provenance.get("protocol_version") == "alzheimer_v2"
                    else {}
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return figure_dir


def run_alzheimer_suite(
    config_path: Path, output_root: Path, *, allow_smoke: bool = False
) -> Path:
    """Run or safely resume the declared Alzheimer comparison suite."""
    suite = load_alzheimer_suite_config(config_path)
    is_v2 = _is_v2_suite(suite)
    if suite.protocol_role == "package_smoke_only" and not allow_smoke:
        raise ValueError("package smoke requires explicit allow_smoke=True")
    output_root = Path(output_root).resolve()
    if is_v2 and suite.evaluation_scope == "outer_final_test_frozen":
        if not output_root.is_dir() and output_root.exists():
            raise ValueError("v2 one-shot formal suite requires an empty output root")
        if output_root.is_dir() and any(output_root.iterdir()):
            raise ValueError("v2 one-shot formal suite requires an empty output root")
    output_root.mkdir(parents=True, exist_ok=True)
    base = replace(
        suite.experiment,
        output_dir=str(output_root),
        seeds=tuple(int(seed) for seed in suite.experiment.seeds),
    )
    uses_inner_validation = suite.evaluation_scope == "inner_validation_smoke_only"
    split_builder = build_validation_split if uses_inner_validation else build_final_split
    split = split_builder(Path(str(base.data_path)), split_seed=suite.split_seed)
    save_alzheimer_preprocessor(
        split.preprocessor, output_root / "alzheimer_preprocessor.npz"
    )
    partitions = {
        (partition_name, int(seed)): _build_partition(
            split.bundle.train_y,
            n_clients=base.n_clients,
            partition=partition_name,
            seed=int(seed),
        )
        for partition_name in suite.partitions
        for seed in base.seeds
    }

    for partition_name in suite.partitions:
        mode, alpha = _partition_parameters(partition_name)
        for method in suite.methods:
            config = method_config(
                base,
                method,
                tuple(int(seed) for seed in base.seeds),
                mode,
                alpha,
            )
            if is_v2:
                config = _apply_v2_final_method_parameters(
                    config, suite=suite, method=method
                )
            else:
                config = apply_frozen_method_parameters(
                    config, suite=suite, method=method
                )
            for seed_value in base.seeds:
                seed = int(seed_value)
                partition = partitions[(partition_name, seed)]
                run_dir = (
                    output_root
                    / "raw"
                    / partition_name
                    / config.stage
                    / f"seed_{seed}"
                )
                result = _load_completed_run(
                    run_dir,
                    config=config,
                    dataset_sha256=split.bundle.sha256,
                    partition_name=partition_name,
                    partition=partition,
                    seed=seed,
                )
                if result is None:
                    result = run_method(
                        method,
                        config,
                        split.bundle,
                        partition,
                        seed,
                    )
                    metadata = _dataset_metadata(
                        dataset_sha256=split.bundle.sha256,
                        dataset_path=split.bundle.source_path,
                        train_count=int(split.bundle.train_x.shape[0]),
                        test_count=int(split.bundle.test_x.shape[0]),
                        partition_name=partition_name,
                        partition=partition,
                        seed=seed,
                    )
                    run_dir = save_run(
                        result,
                        config,
                        output_root / "raw" / partition_name,
                        metadata,
                    )
                if is_v2:
                    _write_v2_raw_manifests(run_dir, config=config, result=result)

    manifest = {
        "status": "completed",
        "config_path": str(Path(config_path).resolve()),
        "dataset_path": split.bundle.source_path,
        "dataset_sha256": split.bundle.sha256,
        "preprocessing": split.metadata["preprocessing"],
        "split_seed": suite.split_seed,
        "methods": list(suite.methods),
        "partitions": list(suite.partitions),
        "seeds": [int(seed) for seed in base.seeds],
        "rounds": base.n_rounds,
        "aggregation": "raw_seed_mean_and_population_std_ddof_0",
        "curve_postprocessing": "none",
        "dp_sgd_x_axis": "aligned_outer_optimization_step",
        "protocol_role": suite.protocol_role,
        "evaluation_scope": suite.evaluation_scope,
        "outer_final_test_accessed": not uses_inner_validation,
        "outer_final_test_firewall": (
            "build_validation_split selected for package smoke"
            if uses_inner_validation
            else "build_final_split selected for formal frozen evaluation"
        ),
        "eligible_for_final_plots": _suite_eligible_for_final_plots(suite),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "reviewed_selection_lock_sha256": _suite_selection_lock_sha256(suite),
        **({"protocol_version": "alzheimer_v2"} if is_v2 else {}),
    }
    (output_root / "suite_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    final_rows: list[dict[str, object]] = []
    for partition_name in suite.partitions:
        method_aggregates = aggregate_alzheimer_runs(
            output_root,
            partition=partition_name,
            allow_smoke=allow_smoke,
        )
        for method in suite.methods:
            aggregate = method_aggregates[method]
            final_rows.append(
                {
                    "partition": partition_name,
                    "method_id": method,
                    "method": DISPLAY_LABELS[method],
                    "seeds": len(base.seeds),
                    "rounds": base.n_rounds,
                    "final_accuracy_mean": float(aggregate["accuracy_mean"][-1]),
                    "final_accuracy_std": float(aggregate["accuracy_std"][-1]),
                }
            )

    # Every figure records the SHA-256 of one shared aggregate manifest. Finish
    # all declared partition aggregates first so that this source manifest is
    # immutable and identical for every standalone figure.
    for partition_name in suite.partitions:
        plot_alzheimer_accuracy(
            output_root,
            partition=partition_name,
            allow_smoke=allow_smoke,
        )

    _write_csv(
        output_root / "final_metrics.csv",
        (
            "partition",
            "method_id",
            "method",
            "seeds",
            "rounds",
            "final_accuracy_mean",
            "final_accuracy_std",
        ),
        final_rows,
    )
    return output_root
