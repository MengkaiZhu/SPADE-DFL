"""Frozen five-fold Alzheimer evaluation after the v4 validation gate."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .alzheimer_contract import ALZHEIMER_METHOD_IDS
from .alzheimer_data import (
    build_stratified_alzheimer_fold,
    save_alzheimer_preprocessor,
)
from .alzheimer_suite import (
    ALZHEIMER_PARTITIONS,
    DISPLAY_LABELS,
    _build_partition,
    _plot_aggregate_arrays,
)
from .alzheimer_tuning_protocol import (
    validate_final_run_protocol,
    validate_parameter_source_alignment,
)
from .alzheimer_v3_protocol import derive_v3_beta, validate_v3_common_config
from .alzheimer_v3_tuning import (
    V3Candidate,
    _candidate_config,
    _canonical_sha256,
    _load_v3_tuning_config,
    _write_csv,
    _write_json_atomic,
    v3_candidate_grid,
)
from .alzheimer_v4_native_budget import _v4_candidate_config
from .artifacts import load_run, save_run
from .config import ExperimentConfig, ParameterRecord
from .graph import build_graph
from .methods import run_method
from .partitioning import PartitionResult


@dataclass(frozen=True)
class CrossValidationContract:
    config_path: Path
    protocol_version: str
    protocol_role: str
    data_path: Path
    tuning_config_path: Path
    split_seed: int
    n_folds: int
    seeds: tuple[int, ...]
    rounds: int
    partitions: tuple[str, ...]
    locked_artifacts: Mapping[str, str]
    v3_selected: Mapping[str, object]
    v4_selected: Mapping[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_contract(path: Path) -> CrossValidationContract:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    common_fields = {
        "schema_version",
        "protocol_version",
        "split_seed",
        "n_folds",
        "seeds",
        "rounds",
        "partitions",
        "data_path",
        "v3_tuning_config",
        "locked_artifacts",
    }
    if not isinstance(raw, dict):
        raise TypeError("cross-validation config has invalid fields")
    protocol_version = raw.get("protocol_version")
    if protocol_version == "alzheimer_v4_cross_validation":
        expected = common_fields
        protocol_role = "local_validation"
        expected_seeds = [0, 1, 2]
    elif protocol_version == "alzheimer_v4_cross_validation_server":
        expected = common_fields | {"protocol_role"}
        protocol_role = "scientific_server"
        expected_seeds = list(range(30))
    else:
        raise ValueError("cross-validation protocol version is unsupported")
    if set(raw) != expected:
        raise ValueError("cross-validation config has invalid fields")
    if (
        raw["schema_version"] != 1
        or raw.get("protocol_role", protocol_role) != protocol_role
        or raw["split_seed"] != 1
        or raw["n_folds"] != 5
        or raw["seeds"] != expected_seeds
        or raw["rounds"] != 50
        or tuple(raw["partitions"]) != ALZHEIMER_PARTITIONS
    ):
        raise ValueError("cross-validation protocol differs from preregistration")
    locked_raw = raw["locked_artifacts"]
    if not isinstance(locked_raw, dict) or len(locked_raw) != 4:
        raise ValueError("cross-validation requires four locked artifacts")
    admitted: dict[str, dict[str, object]] = {}
    hashes: dict[str, str] = {}
    for relative, expected_sha in locked_raw.items():
        artifact = (config_path.parent / relative).resolve()
        if not artifact.is_file() or _sha256(artifact) != expected_sha:
            raise ValueError(f"locked cross-validation artifact mismatch: {relative}")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        admitted[relative] = payload
        hashes[str(artifact)] = expected_sha
    v3_selected = next(
        payload
        for relative, payload in admitted.items()
        if "parameter_validation_v3/selected_parameters.json"
        in relative.replace("\\", "/")
    )
    v4_selected = next(
        payload
        for relative, payload in admitted.items()
        if "parameter_validation_v4_native_budget/selected_parameters.json"
        in relative.replace("\\", "/")
    )
    if v4_selected.get("validation_gate_passed") is not True:
        raise ValueError("cross-validation requires a passing frozen v4 gate")
    return CrossValidationContract(
        config_path=config_path,
        protocol_version=str(protocol_version),
        protocol_role=protocol_role,
        data_path=(config_path.parent / raw["data_path"]).resolve(),
        tuning_config_path=(
            config_path.parent / raw["v3_tuning_config"]
        ).resolve(),
        split_seed=1,
        n_folds=5,
        seeds=tuple(int(seed) for seed in raw["seeds"]),
        rounds=50,
        partitions=ALZHEIMER_PARTITIONS,
        locked_artifacts=hashes,
        v3_selected=v3_selected,
        v4_selected=v4_selected,
    )


def frozen_v4_method_candidates(
    config_path: Path,
) -> dict[str, dict[str, object]]:
    """Return the exact seven candidate IDs and method-native tau values."""
    contract = _load_contract(config_path)
    combined = contract.v4_selected["combined_baseline_winners"]
    spade = contract.v4_selected["selected_spade_candidate"]
    return {
        method: {
            "candidate_id": (
                spade["candidate_id"]
                if method == "spade_dfl"
                else combined[method]["candidate_id"]
            ),
            "tau": (
                8
                if method == "spade_dfl"
                else 2
                if method in {"zo2_lt_admm_vr", "zo2_lt_admm"}
                else 8
            ),
        }
        for method in ALZHEIMER_METHOD_IDS
    }


def _selected_candidates(
    contract: CrossValidationContract,
) -> dict[str, V3Candidate]:
    by_id = {candidate.candidate_id: candidate for candidate in v3_candidate_grid()}
    frozen = frozen_v4_method_candidates(contract.config_path)
    selected: dict[str, V3Candidate] = {}
    for method, row in frozen.items():
        candidate = by_id[str(row["candidate_id"])]
        if candidate.method != method:
            raise ValueError("frozen candidate method mismatch")
        selected[method] = candidate
    return selected


def _final_method_config(
    base: ExperimentConfig,
    method: str,
    candidate: V3Candidate,
    partition_name: str,
) -> ExperimentConfig:
    if method in {"zo2_lt_admm_vr", "zo2_lt_admm"}:
        return _v4_candidate_config(base, candidate, partition_name)
    return _candidate_config(base, candidate, partition_name)


def _record_beta(
    config: ExperimentConfig, candidate: V3Candidate, seed: int
) -> ExperimentConfig:
    graph = build_graph(
        config.n_clients,
        topology=config.graph_topology,
        probability=config.graph_probability,
        seed=seed,
    )
    beta = derive_v3_beta(
        float(candidate.beta_factor),
        tau=config.tau,
        lambda_u=float(graph.laplacian_eigenvalues[-1]),
        rho=config.rho,
    )
    sources = dict(config.parameter_sources)
    sources["beta"] = ParameterRecord(
        "beta",
        beta,
        "TheoryDerived",
        "SPADE-DFL Eq. (31): c_beta/(tau*lambda_u*rho)",
    )
    return replace(config, beta=beta, parameter_sources=sources)


def _execute_cv_run(
    method: str,
    candidate: V3Candidate,
    config: ExperimentConfig,
    bundle: Any,
    partition: PartitionResult,
    fold: int,
    partition_name: str,
    seed: int,
    run_root: Path,
) -> dict[str, object]:
    try:
        if method == "spade_dfl":
            config = _record_beta(config, candidate, seed)
        if method in {"zo2_lt_admm_vr", "zo2_lt_admm"}:
            validate_parameter_source_alignment(config)
        else:
            validate_v3_common_config(config, method)
        validate_final_run_protocol(
            config, method=method, bundle=bundle, partition=partition
        )
        result = run_method(method, config, bundle, partition, seed)
        metadata = {
            "name": "alzheimer",
            "sha256": bundle.sha256,
            "source_path": bundle.source_path,
            "test_role": "stratified_cross_validation_fold",
            "protocol_version": "alzheimer_v4_cross_validation",
            "candidate_id": candidate.candidate_id,
            "fold": fold,
            "partition": partition_name,
            "partition_mode": partition.mode,
            "dirichlet_alpha": partition.alpha,
            "partition_seed": seed,
            "label_histograms": partition.label_histograms.tolist(),
            "config_sha256": _canonical_sha256(config.to_dict()),
        }
        saved = save_run(result, config, run_root, metadata)
        expected = run_root / config.stage / f"seed_{seed}"
        if saved != expected:
            raise RuntimeError("cross-validation artifact path mismatch")
        histories = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in result.history.items()
        }
        finite = bool(
            np.all(np.isfinite(result.states))
            and all(np.all(np.isfinite(values)) for values in histories.values())
        )
        return {
            "status": "completed",
            "finite": finite,
            "accuracy": histories["accuracy"],
            "loss": histories["loss"],
            "stationarity_consensus": histories["stationarity_consensus"],
            "query_count": int(result.query_count),
            "transmitted_bytes": int(result.transmitted_bytes),
            "wall_time": float(result.wall_time),
            "config_sha256": _canonical_sha256(config.to_dict()),
        }
    except (
        ArithmeticError,
        FloatingPointError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        return {
            "status": "failed",
            "finite": False,
            "failure_reason": type(error).__name__,
            "failure_message": str(error),
        }


def _load_existing_cv_run(
    run_dir: Path,
    *,
    config: ExperimentConfig,
    bundle: Any,
    partition: PartitionResult,
    partition_name: str,
    seed: int,
) -> dict[str, object] | None:
    """Load one exact completed server run; reject partial/stale artifacts."""
    try:
        result = load_run(
            run_dir,
            expected_config=config,
            expected_dataset_sha256=bundle.sha256,
            expected_partition=partition_name,
            expected_label_histograms=partition.label_histograms,
            expected_seed=seed,
            expected_stage=config.stage,
            expected_history_length=config.n_rounds + 1,
            expected_state_shape=(config.n_clients, config.dimension),
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        return None
    histories = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in result.history.items()
    }
    if not all(np.all(np.isfinite(values)) for values in histories.values()):
        return None
    return {
        "status": "completed",
        "finite": True,
        "accuracy": histories["accuracy"],
        "loss": histories["loss"],
        "stationarity_consensus": histories["stationarity_consensus"],
        "query_count": int(result.query_count),
        "transmitted_bytes": int(result.transmitted_bytes),
        "wall_time": float(result.wall_time),
        "config_sha256": _canonical_sha256(config.to_dict()),
        "resumed": True,
    }


def _aggregate_curves(
    evidence: list[Mapping[str, object]],
    *,
    output: Path,
    repetitions: int,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in evidence:
        if row.get("status") != "completed" or row.get("finite") is not True:
            raise RuntimeError("cross-validation contains a failed or nonfinite run")
        grouped[(str(row["partition"]), str(row["method"]))].append(row)
    all_aggregates: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    final_rows: list[dict[str, object]] = []
    for partition_name in ALZHEIMER_PARTITIONS:
        method_aggregates: dict[str, dict[str, np.ndarray]] = {}
        for method in ALZHEIMER_METHOD_IDS:
            rows = grouped[(partition_name, method)]
            if len(rows) != repetitions:
                raise RuntimeError(
                    "each method/partition has an invalid fold-seed run count"
                )
            accuracy = np.stack(
                [np.asarray(row["accuracy"], dtype=np.float64) for row in rows]
            )
            loss = np.stack(
                [np.asarray(row["loss"], dtype=np.float64) for row in rows]
            )
            stationarity = np.stack(
                [
                    np.asarray(row["stationarity_consensus"], dtype=np.float64)
                    for row in rows
                ]
            )
            if accuracy.shape != (repetitions, 51):
                raise RuntimeError("cross-validation curves must have 51 points")
            aggregate = {
                "round": np.arange(51, dtype=np.int64),
                "accuracy_mean": accuracy.mean(axis=0),
                "accuracy_std": accuracy.std(axis=0, ddof=0),
                "loss_mean": loss.mean(axis=0),
                "loss_std": loss.std(axis=0, ddof=0),
                "stationarity_consensus_mean": stationarity.mean(axis=0),
                "stationarity_consensus_std": stationarity.std(axis=0, ddof=0),
            }
            method_aggregates[method] = aggregate
            aggregate_dir = output / "aggregates" / partition_name
            aggregate_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(aggregate_dir / f"{method}.npz", **aggregate)
            final_rows.append(
                {
                    "partition": partition_name,
                    "method_id": method,
                    "method": DISPLAY_LABELS[method],
                    "fold_seed_runs": repetitions,
                    "rounds": 50,
                    "final_accuracy_mean": float(aggregate["accuracy_mean"][-1]),
                    "final_accuracy_std": float(aggregate["accuracy_std"][-1]),
                    "query_count_total": sum(int(row["query_count"]) for row in rows),
                    "transmitted_bytes_total": sum(
                        int(row["transmitted_bytes"]) for row in rows
                    ),
                    "wall_time_total": sum(float(row["wall_time"]) for row in rows),
                }
            )
        all_aggregates[partition_name] = method_aggregates
    return all_aggregates, final_rows


def _scientific_acceptance(rows: list[Mapping[str, object]]) -> tuple[bool, dict[str, int]]:
    ranks: dict[str, int] = {}
    for partition in ALZHEIMER_PARTITIONS:
        selected = [row for row in rows if row["partition"] == partition]
        ordered = sorted(
            selected,
            key=lambda row: (
                -float(row["final_accuracy_mean"]),
                str(row["method_id"]),
            ),
        )
        ranks[partition] = next(
            index
            for index, row in enumerate(ordered, start=1)
            if row["method_id"] == "spade_dfl"
        )
    return all(rank == 1 for rank in ranks.values()), ranks


def run_v4_cross_validation(config_path: Path, output_root: Path) -> Path:
    """Run the frozen local or reference-replicated server evaluation."""
    output = Path(output_root).resolve()
    contract = _load_contract(config_path)
    if output.exists() and not output.is_dir():
        raise ValueError("cross-validation output root is nonempty")
    if (
        output.is_dir()
        and any(output.iterdir())
        and contract.protocol_role != "scientific_server"
    ):
        raise ValueError("cross-validation output root is nonempty")
    tuning = _load_v3_tuning_config(contract.tuning_config_path)
    base = replace(
        tuning.experiment,
        n_rounds=contract.rounds,
        seeds=contract.seeds,
        output_dir=str(output),
    )
    candidates = _selected_candidates(contract)
    expected_run_count = (
        contract.n_folds
        * len(contract.partitions)
        * len(ALZHEIMER_METHOD_IDS)
        * len(contract.seeds)
    )
    output.mkdir(parents=True, exist_ok=True)
    evidence: list[dict[str, object]] = []
    fold_audit: list[dict[str, object]] = []
    for fold in range(contract.n_folds):
        split = build_stratified_alzheimer_fold(
            contract.data_path,
            n_folds=contract.n_folds,
            fold_index=fold,
            split_seed=contract.split_seed,
        )
        save_alzheimer_preprocessor(
            split.preprocessor,
            output / "preprocessors" / f"fold_{fold}.npz",
        )
        fold_audit.append(
            {
                "fold": fold,
                "train_indices_sha256": hashlib.sha256(
                    np.asarray(split.model_indices, dtype="<i8").tobytes()
                ).hexdigest(),
                "test_indices_sha256": hashlib.sha256(
                    np.asarray(split.test_indices, dtype="<i8").tobytes()
                ).hexdigest(),
                "train_count": len(split.model_indices),
                "test_count": len(split.test_indices),
                "indices_disjoint": set(split.model_indices.tolist()).isdisjoint(
                    split.test_indices.tolist()
                ),
            }
        )
        partitions = {
            (partition_name, seed): _build_partition(
                split.bundle.train_y,
                n_clients=31,
                partition=partition_name,
                seed=seed,
            )
            for partition_name in contract.partitions
            for seed in contract.seeds
        }
        for partition_name in contract.partitions:
            for method in ALZHEIMER_METHOD_IDS:
                candidate = candidates[method]
                config = _final_method_config(
                    base, method, candidate, partition_name
                )
                for seed in contract.seeds:
                    run_root = (
                        output
                        / "raw"
                        / f"fold_{fold}"
                        / partition_name
                    )
                    exact_config = (
                        _record_beta(config, candidate, seed)
                        if method == "spade_dfl"
                        else config
                    )
                    resumed = (
                        _load_existing_cv_run(
                            run_root
                            / exact_config.stage
                            / f"seed_{seed}",
                            config=exact_config,
                            bundle=split.bundle,
                            partition=partitions[(partition_name, seed)],
                            partition_name=partition_name,
                            seed=seed,
                        )
                        if contract.protocol_role == "scientific_server"
                        else None
                    )
                    row = dict(
                        resumed
                        or _execute_cv_run(
                            method,
                            candidate,
                            config,
                            split.bundle,
                            partitions[(partition_name, seed)],
                            fold,
                            partition_name,
                            seed,
                            run_root,
                        )
                    )
                    row.update(
                        fold=fold,
                        partition=partition_name,
                        method=method,
                        seed=seed,
                    )
                    evidence.append(row)
                    _write_json_atomic(
                        output / "progress.json",
                        {
                            "status": "running",
                            "protocol_version": contract.protocol_version,
                            "completed_or_attempted_runs": len(evidence),
                            "expected_run_count": expected_run_count,
                            "last_fold": fold,
                            "last_partition": partition_name,
                            "last_method": method,
                            "last_seed": seed,
                            "last_status": row.get("status"),
                            "last_run_resumed": row.get("resumed", False),
                        },
                    )
    repetitions = contract.n_folds * len(contract.seeds)
    aggregates, final_rows = _aggregate_curves(
        evidence,
        output=output,
        repetitions=repetitions,
    )
    for partition_name in contract.partitions:
        _plot_aggregate_arrays(
            partition_name, aggregates[partition_name], output / "figures"
        )
    _write_csv(output / "final_metrics.csv", final_rows)
    scientific, ranks = _scientific_acceptance(final_rows)
    manifest = {
        "status": "completed",
        "protocol_version": contract.protocol_version,
        "protocol_role": contract.protocol_role,
        "expected_run_count": expected_run_count,
        "completed_run_count": sum(
            row.get("status") == "completed" for row in evidence
        ),
        "finite_run_count": sum(
            row.get("finite") is True for row in evidence
        ),
        "resumed_run_count": sum(
            row.get("resumed") is True for row in evidence
        ),
        "folds": contract.n_folds,
        "seeds": list(contract.seeds),
        "rounds": contract.rounds,
        "methods": list(ALZHEIMER_METHOD_IDS),
        "partitions": list(ALZHEIMER_PARTITIONS),
        "curve_postprocessing": "none",
        "aggregation": "raw_fold_seed_mean_and_population_std_ddof_0",
        "locked_artifacts": dict(contract.locked_artifacts),
        "fold_audit": fold_audit,
        "mechanical_acceptance": len(evidence) == expected_run_count
        and all(row.get("finite") is True for row in evidence),
        "scientific_acceptance": scientific,
        "spade_partition_ranks": ranks,
        "evaluation_limitation": (
            "Iterative stratified cross-validation on one cohort; not an "
            "independent external cohort."
        ),
    }
    _write_json_atomic(output / "suite_manifest.json", manifest)
    _write_json_atomic(
        output / "progress.json",
        {
            "status": "completed",
            "protocol_version": contract.protocol_version,
            "completed_or_attempted_runs": len(evidence),
            "expected_run_count": expected_run_count,
        },
    )
    return output
