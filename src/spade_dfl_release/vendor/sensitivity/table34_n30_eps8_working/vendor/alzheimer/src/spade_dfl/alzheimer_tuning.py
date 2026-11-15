"""Training-only, staged parameter validation for the Alzheimer suite."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .alzheimer_contract import ALZHEIMER_METHOD_IDS
from .alzheimer_data import build_validation_split
from .alzheimer_suite import (
    _build_partition,
    _partition_parameters,
)
from .alzheimer_tuning_cache import (
    cache_matches_canonical,
    canonical_evidence,
    load_metrics_cache,
    valid_failed_cache,
)
from .alzheimer_tuning_protocol import (
    APPROVED_TUNING_PROTOCOL,
    protocol_candidate_values,
    validate_final_run_protocol,
)
from .alzheimer_validation_firewall import ValidationOnlySplitView
from .artifacts import load_run, save_run
from .config import ExperimentConfig, ParameterRecord
from .methods import method_config, run_method
from .objectives import build_objective
from .partitioning import PartitionResult
from .trainer import RunResult

ETA_MULTIPLIERS = APPROVED_TUNING_PROTOCOL.eta_multipliers
RHO_VALUES = APPROVED_TUNING_PROTOCOL.rho_values
BATCH_SIZES = APPROVED_TUNING_PROTOCOL.batch_sizes
MU_MULTIPLIERS = APPROVED_TUNING_PROTOCOL.mu_multipliers
BASELINE_METHODS = tuple(
    method for method in ALZHEIMER_METHOD_IDS if method != "spade_dfl"
)
RANKING_RULE = (
    "finite first; higher mean validation accuracy; lower mean validation "
    "loss; lower mean stationarity-consensus; lower total query count; "
    "declaration order only resolves an exact tie after every declared metric"
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    declaration_order: int
    stage_id: str
    phase: str
    method: str
    eta_multiplier: Optional[float] = None  # noqa: UP045 (Python 3.9)
    eta_value: Optional[float] = None  # noqa: UP045 (Python 3.9)
    rho: Optional[float] = None  # noqa: UP045 (Python 3.9)
    batch_size: Optional[int] = None  # noqa: UP045 (Python 3.9)
    mu_multiplier: Optional[float] = None  # noqa: UP045 (Python 3.9)
    mu_value: Optional[float] = None  # noqa: UP045 (Python 3.9)
    source_type: str = "Validation"
    source: str = ""
    center_source: str = ""
    beta_source_type: str = "TheoryDerived"
    beta_source: str = (
        "SPADE-DFL admissible LT-ADMM interval midpoint "
        "1.5/(tau*lambda_u*rho); never manually tried"
    )
    frozen_before: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class TuningConfig:
    experiment: ExperimentConfig
    split_seed: int
    partitions: tuple[str, ...]
    seeds: tuple[int, ...]
    rounds: int
    binding_protocol: bool
    config_path: Path


def _number_id(value: float) -> str:
    return format(value, "g").replace(".", "p")


def candidate_grid(dimension: int) -> list[Candidate]:
    """Return the complete approved, reference-centered validation grid."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    spade_eta_center = APPROVED_TUNING_PROTOCOL.eta_numerator / dimension
    spade_source = (
        "Approved training-only Validation grid centered on the local "
        "SPADE-DFL manuscript 鏂囩珷/SPADE-DFL.pdf"
    )
    beta_source = (
        "TheoryDerived midpoint of the SPADE-DFL admissible LT-ADMM interval "
        "1.5/(tau*lambda_u*rho), using the candidate graph and rho"
    )
    candidates: list[Candidate] = []

    def add(**values: Any) -> None:
        candidates.append(
            Candidate(
                declaration_order=len(candidates),
                beta_source=beta_source,
                **values,
            )
        )

    for multiplier in ETA_MULTIPLIERS:
        add(
            candidate_id=f"spade_eta_m{_number_id(multiplier)}",
            stage_id="spade_eta",
            phase="eta",
            method="spade_dfl",
            eta_multiplier=multiplier,
            eta_value=spade_eta_center * multiplier,
            source=spade_source,
            center_source="鏂囩珷/SPADE-DFL.pdf: eta0=1.5/d",
        )
    for rho in RHO_VALUES:
        add(
            candidate_id=f"spade_rho_{_number_id(rho)}",
            stage_id="spade_rho",
            phase="rho",
            method="spade_dfl",
            rho=rho,
            source=(
                "Approved training-only Validation grid; SPADE-DFL manuscript "
                "鏂囩珷/SPADE-DFL.pdf and LT-ADMM reference [13] "
                "鏂囩珷/13-LT-ADMM-VR.pdf, "
                "https://people.kth.se/~kallej/papers/"
                "Distributed%20_CDC2024_Ren.pdf"
            ),
            center_source="LT-ADMM/SPADE positive ADMM penalty reference center",
        )
    for batch_size in BATCH_SIZES:
        add(
            candidate_id=f"spade_batch_{batch_size}",
            stage_id="spade_batch",
            phase="batch",
            method="spade_dfl",
            batch_size=batch_size,
            source=(
                "Approved training-only Validation alternatives for the "
                "persistent-SAGA implementation; SAGA paper and supplied "
                "SPADE-DFL appendix (鏂囩珷/SPADE-DFL.pdf, Sec. X-A)"
            ),
            center_source="Persistent SAGA reference single-sample update",
        )
    for multiplier in MU_MULTIPLIERS:
        add(
            candidate_id=f"spade_mu_m{_number_id(multiplier)}",
            stage_id="spade_mu",
            phase="mu",
            method="spade_dfl",
            mu_multiplier=multiplier,
            mu_value=APPROVED_TUNING_PROTOCOL.mu_center * multiplier,
            source=spade_source,
            center_source="鏂囩珷/SPADE-DFL.pdf: mu0=3.5",
        )

    for method in BASELINE_METHODS:
        center = APPROVED_TUNING_PROTOCOL.baseline_eta_center(method)
        if method.startswith("zo2_lt_admm"):
            center_source = (
                "LT-ADMM reference [13], 鏂囩珷/13-LT-ADMM-VR.pdf and "
                "https://people.kth.se/~kallej/papers/"
                "Distributed%20_CDC2024_Ren.pdf"
            )
        elif method.startswith("zo2_dp_"):
            center_source = (
                "Abadi et al. DP-SGD, https://arxiv.org/abs/1607.00133"
            )
        else:
            center_source = "Existing exact method contract and its cited reference"
        for multiplier in ETA_MULTIPLIERS:
            add(
                candidate_id=(
                    f"{method}_eta_m{_number_id(multiplier)}"
                ),
                stage_id=f"baseline_eta:{method}",
                phase="baseline_eta",
                method=method,
                eta_multiplier=multiplier,
                eta_value=center * multiplier,
                source=(
                    "Approved training-only Validation multiplier grid; "
                    + center_source
                ),
                center_source=center_source,
            )
    return candidates


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _resolve_data_path(config_path: Path, experiment: ExperimentConfig) -> ExperimentConfig:
    data_path = Path(str(experiment.data_path))
    if not data_path.is_absolute():
        data_path = config_path.parent / data_path
    return replace(experiment, data_path=str(data_path.resolve()))


def _validate_declared_grid(raw: Mapping[str, object], binding: bool) -> None:
    declared = raw.get("candidate_values")
    if declared is None:
        if binding:
            raise ValueError("binding tuning config must declare candidate_values")
        return
    expected = protocol_candidate_values()
    if declared != expected:
        raise ValueError("candidate_values differ from the approved validation grid")


def load_tuning_config(path: Path) -> TuningConfig:
    """Load a tuning protocol and enforce the full binding real configuration."""
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("tuning configuration must be a JSON object")
    experiment_raw = raw.get("experiment")
    if not isinstance(experiment_raw, dict):
        raise TypeError("experiment must be a JSON object")
    binding = raw.get("binding_protocol", True)
    if not isinstance(binding, bool):
        raise TypeError("binding_protocol must be boolean")
    split_seed = _as_int(raw.get("split_seed"), "split_seed")
    rounds = _as_int(raw.get("rounds"), "rounds")
    partitions_raw = raw.get("partitions")
    seeds_raw = raw.get("seeds")
    if not isinstance(partitions_raw, list) or not isinstance(seeds_raw, list):
        raise TypeError("partitions and seeds must be JSON lists")
    partitions = tuple(str(value) for value in partitions_raw)
    seeds = tuple(_as_int(value, "seed") for value in seeds_raw)
    if len(set(partitions)) != len(partitions) or any(
        partition not in APPROVED_TUNING_PROTOCOL.partitions
        for partition in partitions
    ):
        raise ValueError("partitions must be unique canonical Alzheimer partitions")
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    _validate_declared_grid(raw, binding)

    experiment = _resolve_data_path(
        config_path, ExperimentConfig.from_dict(experiment_raw)
    )
    if experiment.dataset_name != "alzheimer":
        raise ValueError("tuning requires dataset_name=alzheimer")
    protocol = APPROVED_TUNING_PROTOCOL
    if (
        experiment.dimension != protocol.dimension
        or experiment.feature_dimension != protocol.dimension
    ):
        raise ValueError("tuning requires the exact 39-dimensional encoding")
    if experiment.n_clients != protocol.n_partition_clients:
        raise ValueError("tuning requires 31 clients")
    experiment = replace(experiment, n_rounds=rounds, seeds=seeds)

    if binding:
        expected_partitions = protocol.partitions
        if (
            rounds != protocol.rounds
            or partitions != expected_partitions
            or seeds != protocol.seeds
            or split_seed != protocol.split_seed
        ):
            raise ValueError(
                "binding protocol requires 50 rounds, all four canonical "
                "partitions, seeds 0/1/2, and split_seed=1"
            )
        exact_values = {
            "eta0": protocol.eta_center,
            "eta_exponent": protocol.eta_exponent,
            "mu0": protocol.mu_center,
            "mu_exponent": protocol.mu_exponent,
            "regularization": protocol.regularization,
            "sigma_u": protocol.sigma_u,
            "query_noise_std": protocol.query_noise_std,
            "graph_probability": protocol.graph_probability,
            "target_epsilon": protocol.target_epsilon,
            "target_delta": protocol.target_delta,
            "clip_norm": protocol.clip_norm,
            "rho": protocol.rho_center,
        }
        if (
            experiment.tau != protocol.tau
            or experiment.graph_topology != "erdos_renyi"
        ):
            raise ValueError("binding protocol requires tau=2 and Erdos-Renyi")
        for name, expected in exact_values.items():
            if not math.isclose(
                float(getattr(experiment, name)),
                expected,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"binding protocol has invalid {name}")
    return TuningConfig(
        experiment=experiment,
        split_seed=split_seed,
        partitions=partitions,
        seeds=seeds,
        rounds=rounds,
        binding_protocol=binding,
        config_path=config_path,
    )


def _sha256_indices(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _write_candidate_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    records = list(rows)
    fieldnames = (
        "candidate_id",
        "declaration_order",
        "stage_id",
        "phase",
        "method",
        "eta_multiplier",
        "eta_value",
        "rho",
        "batch_size",
        "mu_multiplier",
        "mu_value",
        "source_type",
        "source",
        "center_source",
        "beta_source_type",
        "beta_source",
        "frozen_before",
        "expected_run_count",
        "evaluated_run_count",
        "completed_run_count",
        "finite_run_count",
        "finite",
        "mean_validation_accuracy",
        "mean_validation_loss",
        "mean_stationarity_consensus",
        "total_query_count",
        "failure_reasons",
        "rank",
        "selection_status",
        "rejection_reason",
        "winner_candidate_id",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                key: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
                if key in fieldnames
            }
            for row in records
        )
    temporary.replace(path)


def rank_candidate_summaries(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Rank one stage exactly by the preregistered validation rule."""
    records = [dict(row) for row in rows]
    if not records:
        raise ValueError("at least one candidate summary is required")

    def key(row: Mapping[str, object]) -> tuple[object, ...]:
        finite = row.get("finite") is True
        return (
            0 if finite else 1,
            -float(row["mean_validation_accuracy"]) if finite else 0.0,
            float(row["mean_validation_loss"]) if finite else math.inf,
            float(row["mean_stationarity_consensus"]) if finite else math.inf,
            int(row["total_query_count"]) if finite else 2**63 - 1,
            int(row["declaration_order"]),
        )

    ranked = sorted(records, key=key)
    winner_id = (
        str(ranked[0]["candidate_id"])
        if ranked and ranked[0].get("finite") is True
        else ""
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["winner_candidate_id"] = winner_id
        if row["candidate_id"] == winner_id:
            row["selection_status"] = "selected"
            row["rejection_reason"] = ""
        elif row.get("finite") is not True:
            row["selection_status"] = "rejected"
            row["rejection_reason"] = "nonfinite_or_incomplete_runs"
        else:
            row["selection_status"] = "rejected"
            row["rejection_reason"] = "out_ranked_by_stage_winner"
    return ranked


def _candidate_config(
    base: ExperimentConfig,
    candidate: Candidate,
    *,
    partition_name: str,
) -> ExperimentConfig:
    mode, alpha = _partition_parameters(partition_name)
    config = method_config(
        base,
        candidate.method,
        tuple(int(seed) for seed in base.seeds),
        mode,
        alpha,
    )
    if candidate.method in APPROVED_TUNING_PROTOCOL.dp_methods:
        config = replace(
            config,
            clip_norm=base.clip_norm,
            target_epsilon=base.target_epsilon,
            target_delta=base.target_delta,
        )
    frozen = dict(candidate.frozen_before)
    if candidate.method == "spade_dfl":
        values: dict[str, object] = {
            "eta0": (
                APPROVED_TUNING_PROTOCOL.eta_numerator / base.dimension
            )
            * float(frozen.get("spade_eta_multiplier", 1.0)),
            "rho": float(
                frozen.get(
                    "spade_rho",
                    APPROVED_TUNING_PROTOCOL.rho_center,
                )
            ),
            "batch_size": int(frozen.get("spade_batch_size", 1)),
            "mu0": APPROVED_TUNING_PROTOCOL.mu_center
            * float(frozen.get("spade_mu_multiplier", 1.0)),
            "beta": 0.0,
        }
        if candidate.phase == "eta":
            values["eta0"] = candidate.eta_value
        elif candidate.phase == "rho":
            values["rho"] = candidate.rho
        elif candidate.phase == "batch":
            values["batch_size"] = candidate.batch_size
        elif candidate.phase == "mu":
            values["mu0"] = candidate.mu_value
        config = replace(config, **values)
    else:
        config = replace(config, eta0=float(candidate.eta_value), beta=0.0)
    sources = dict(config.parameter_sources)
    frozen_source = (
        "Frozen winner from the preceding staged training-only validation "
        "phase; immutable candidate record is in selected_parameters.json"
    )

    def validation_record(
        name: str,
        value: object,
        *,
        is_current_candidate: bool,
    ) -> None:
        sources[name] = ParameterRecord(
            name,
            value,
            candidate.source_type if is_current_candidate else "Validation",
            candidate.source if is_current_candidate else frozen_source,
        )

    if candidate.method == "spade_dfl":
        validation_record(
            "eta0",
            config.eta0,
            is_current_candidate=candidate.phase == "eta",
        )
        validation_record(
            "rho",
            config.rho,
            is_current_candidate=candidate.phase == "rho",
        )
        validation_record(
            "batch_size",
            config.batch_size,
            is_current_candidate=candidate.phase == "batch",
        )
        validation_record(
            "mu0",
            config.mu0,
            is_current_candidate=candidate.phase == "mu",
        )
    else:
        validation_record("eta0", config.eta0, is_current_candidate=True)
    if candidate.method in APPROVED_TUNING_PROTOCOL.dp_methods:
        dp_source = (
            "Binding Alzheimer DP baseline protocol: clip_norm=1, epsilon=8, "
            "delta=1e-5; Abadi et al., https://arxiv.org/abs/1607.00133"
        )
        for name in ("clip_norm", "target_epsilon", "target_delta"):
            sources[name] = ParameterRecord(
                name,
                getattr(config, name),
                "Reference",
                dp_source,
            )
    else:
        no_dp_source = (
            "Exact non-private method contract; DP release budget is inactive"
        )
        for name in ("target_epsilon", "target_delta"):
            sources[name] = ParameterRecord(
                name,
                getattr(config, name),
                "Assumed",
                no_dp_source,
            )
    sources["beta"] = ParameterRecord(
        "beta",
        "derived",
        candidate.beta_source_type,
        candidate.beta_source,
    )
    sources["validation_frozen_context"] = ParameterRecord(
        "validation_frozen_context",
        _canonical_frozen_context(candidate),
        "Validation",
        "Canonical preceding staged-winner IDs and frozen values",
    )
    return replace(config, parameter_sources=sources)


def _canonical_frozen_context(candidate: Candidate) -> dict[str, object]:
    return {
        name: value
        for name, value in sorted(
            candidate.frozen_before,
            key=lambda item: item[0],
        )
    }


def _run_fingerprint(
    candidate: Candidate,
    config: ExperimentConfig,
    dataset_sha256: str,
    partition_name: str,
    seed: int,
) -> str:
    payload = {
        "candidate_id": candidate.candidate_id,
        "config": config.to_dict(),
        "dataset_sha256": dataset_sha256,
        "partition": partition_name,
        "seed": seed,
        "frozen_before": _canonical_frozen_context(candidate),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_completed_result(
    run_dir: Path,
    *,
    config: ExperimentConfig,
    dataset_sha256: str,
    partition_name: str,
    partition: PartitionResult,
    seed: int,
) -> Optional[RunResult]:  # noqa: UP045 (Python 3.9)
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


def _metrics_from_result(
    result: RunResult,
    config: ExperimentConfig,
    bundle: Any,
) -> dict[str, object]:
    objective = build_objective(
        config.objective_model,
        config.regularization,
        feature_dimension=config.input_dimension,
        n_classes=config.n_classes,
    )
    average_state = np.asarray(result.states, dtype=np.float64).mean(axis=0)
    validation_loss = float(
        objective.value(
            average_state,
            bundle.test_x,
            bundle.test_y,
            u=1.0,
        )
    )
    history_arrays = [
        np.asarray(values, dtype=np.float64)
        for values in result.history.values()
    ]
    accuracy = float(np.asarray(result.history["accuracy"])[-1])
    stationarity = float(
        np.asarray(result.history["stationarity_consensus"])[-1]
    )
    finite = bool(
        np.all(np.isfinite(result.states))
        and all(np.all(np.isfinite(values)) for values in history_arrays)
        and np.isfinite(validation_loss)
        and np.isfinite(accuracy)
        and np.isfinite(stationarity)
    )
    return {
        "status": "completed",
        "finite": finite,
        "validation_accuracy": accuracy,
        "validation_loss": validation_loss,
        "stationarity_consensus": stationarity,
        "query_count": int(result.query_count),
        "beta": float(result.metadata.get("beta", 0.0)),
    }


def _execute_candidate_run(
    candidate: Candidate,
    config: ExperimentConfig,
    bundle: Any,
    partition: PartitionResult,
    seed: int,
    run_dir: Path,
) -> dict[str, object]:
    """Execute or reuse one auditable candidate/partition/seed run."""
    partition_name = run_dir.parents[1].name
    validate_final_run_protocol(
        config,
        method=candidate.method,
        bundle=bundle,
        partition=partition,
    )
    metrics_path = run_dir / "run_metrics.json"
    fingerprint = _run_fingerprint(
        candidate,
        config,
        bundle.sha256,
        partition_name,
        seed,
    )
    identity = {
        "fingerprint": fingerprint,
        "candidate_id": candidate.candidate_id,
        "partition": partition_name,
        "seed": seed,
        "artifact_path": str(run_dir.resolve()),
        "frozen_before": _canonical_frozen_context(candidate),
    }
    stored = load_metrics_cache(metrics_path) if metrics_path.is_file() else None
    result = _load_completed_result(
        run_dir,
        config=config,
        dataset_sha256=bundle.sha256,
        partition_name=partition_name,
        partition=partition,
        seed=seed,
    )
    if result is not None:
        evidence = canonical_evidence(
            _metrics_from_result(result, config, bundle),
            identity,
        )
        if not cache_matches_canonical(stored, evidence):
            _write_json(metrics_path, evidence)
        return evidence
    if valid_failed_cache(stored, identity):
        return dict(stored)

    try:
        result = run_method(candidate.method, config, bundle, partition, seed)
        metadata = {
            "name": "alzheimer",
            "sha256": bundle.sha256,
            "source_path": bundle.source_path,
            "train_count": int(bundle.train_x.shape[0]),
            "test_count": int(bundle.test_x.shape[0]),
            "test_role": "inner_validation_only",
            "dimension": APPROVED_TUNING_PROTOCOL.dimension,
            "preprocessing": "standardize_plus_one_hot_no_pca",
            "partition": partition_name,
            "partition_mode": partition.mode,
            "dirichlet_alpha": partition.alpha,
            "partition_seed": seed,
            "label_histograms": partition.label_histograms.tolist(),
        }
        saved_dir = save_run(
            result,
            config,
            run_dir.parent.parent,
            metadata,
        )
        if saved_dir != run_dir:
            raise RuntimeError("candidate artifact path did not match declaration")
        evidence = canonical_evidence(
            _metrics_from_result(result, config, bundle),
            identity,
        )
    except (
        ArithmeticError,
        FloatingPointError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        evidence = {
            "status": "failed",
            "finite": False,
            "validation_accuracy": None,
            "validation_loss": None,
            "stationarity_consensus": None,
            "query_count": 0,
            "beta": None,
            **identity,
            "failure_reason": type(error).__name__,
            "failure_message": str(error),
        }
    _write_json(metrics_path, evidence)
    return evidence


def _summarize_candidate(
    candidate: Candidate,
    run_evidence: list[Mapping[str, object]],
    expected_run_count: int,
) -> dict[str, object]:
    completed = [
        row for row in run_evidence if row.get("status") == "completed"
    ]
    finite_rows = [row for row in completed if row.get("finite") is True]
    finite = (
        len(run_evidence) == expected_run_count
        and len(finite_rows) == expected_run_count
    )
    failures = [
        {
            "partition": row.get("partition"),
            "seed": row.get("seed"),
            "reason": row.get(
                "failure_reason",
                "nonfinite_metrics"
                if row.get("status") == "completed"
                else "missing_or_failed_run",
            ),
            "message": row.get("failure_message", ""),
        }
        for row in run_evidence
        if row.get("status") != "completed" or row.get("finite") is not True
    ]
    if len(run_evidence) < expected_run_count:
        failures.append(
            {
                "partition": None,
                "seed": None,
                "reason": "missing_runs",
                "message": (
                    f"expected {expected_run_count}, observed {len(run_evidence)}"
                ),
            }
        )
    metrics: dict[str, object]
    if finite:
        metrics = {
            "mean_validation_accuracy": float(
                np.mean([float(row["validation_accuracy"]) for row in finite_rows])
            ),
            "mean_validation_loss": float(
                np.mean([float(row["validation_loss"]) for row in finite_rows])
            ),
            "mean_stationarity_consensus": float(
                np.mean(
                    [
                        float(row["stationarity_consensus"])
                        for row in finite_rows
                    ]
                )
            ),
            "total_query_count": int(
                sum(int(row["query_count"]) for row in finite_rows)
            ),
        }
    else:
        metrics = {
            "mean_validation_accuracy": None,
            "mean_validation_loss": None,
            "mean_stationarity_consensus": None,
            "total_query_count": 0,
        }
    row = asdict(candidate)
    row.update(
        {
            "frozen_before": dict(candidate.frozen_before),
            "expected_run_count": expected_run_count,
            "evaluated_run_count": len(run_evidence),
            "completed_run_count": len(completed),
            "finite_run_count": len(finite_rows),
            "finite": finite,
            "failure_reasons": failures,
            **metrics,
        }
    )
    return row


def _freeze_value(
    frozen: dict[str, object], winner: Mapping[str, object]
) -> None:
    phase = str(winner["phase"])
    method = str(winner["method"])
    stage_key = str(winner["stage_id"]).replace(":", "_")
    frozen[f"{stage_key}_winner_candidate_id"] = winner["candidate_id"]
    if phase == "eta":
        frozen["spade_eta_multiplier"] = winner["eta_multiplier"]
        frozen["spade_eta0"] = winner["eta_value"]
    elif phase == "rho":
        frozen["spade_rho"] = winner["rho"]
    elif phase == "batch":
        frozen["spade_batch_size"] = winner["batch_size"]
    elif phase == "mu":
        frozen["spade_mu_multiplier"] = winner["mu_multiplier"]
        frozen["spade_mu0"] = winner["mu_value"]
    elif phase == "baseline_eta":
        frozen[f"{method}_eta_multiplier"] = winner["eta_multiplier"]
        frozen[f"{method}_eta0"] = winner["eta_value"]


def freeze_selected_parameters(
    output_root: Path,
    *,
    frozen_values: Mapping[str, object],
    stage_winners: Iterable[Mapping[str, object]],
) -> Path:
    """Persist selected values and their immutable validation record IDs."""
    winners = [
        {
            "stage_id": row["stage_id"],
            "candidate_id": row["candidate_id"],
            "method": row["method"],
            "rank": row["rank"],
            "mean_validation_accuracy": row["mean_validation_accuracy"],
            "mean_validation_loss": row["mean_validation_loss"],
            "mean_stationarity_consensus": row[
                "mean_stationarity_consensus"
            ],
            "total_query_count": row["total_query_count"],
            "source_type": row["source_type"],
            "source": row["source"],
        }
        for row in stage_winners
    ]
    payload = {
        "status": "frozen",
        "selection_scope": "inner_training_validation_only",
        "ranking_rule": RANKING_RULE,
        "deterministic_tie_handling": (
            "Candidate declaration order resolves only an exact tie after "
            "finite status, accuracy, loss, stationarity-consensus, and queries."
        ),
        "selected_values": dict(frozen_values),
        "stage_winners": winners,
        "beta": {
            "value": "derived_per_graph_seed_and_selected_rho",
            "source_type": "TheoryDerived",
            "source": (
                "Midpoint 1.5/(tau*lambda_u*rho) of the SPADE-DFL "
                "admissible LT-ADMM interval; beta was never grid searched."
            ),
        },
    }
    return _write_json(Path(output_root) / "selected_parameters.json", payload)


def run_staged_validation(config_path: Path, output_root: Path) -> Path:
    """Run or safely resume the complete training-only staged validation."""
    tuning = load_tuning_config(config_path)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    loader_result = build_validation_split(
        Path(str(tuning.experiment.data_path)),
        split_seed=tuning.split_seed,
    )
    split = ValidationOnlySplitView.from_loader_result(loader_result)
    del loader_result
    base = replace(
        tuning.experiment,
        output_dir=str(output),
        n_rounds=tuning.rounds,
        seeds=tuning.seeds,
    )
    partitions = {
        (partition_name, seed): _build_partition(
            split.bundle.train_y,
            n_clients=base.n_clients,
            partition=partition_name,
            seed=seed,
        )
        for partition_name in tuning.partitions
        for seed in tuning.seeds
    }
    grid = candidate_grid(base.dimension)
    stage_ids = list(dict.fromkeys(candidate.stage_id for candidate in grid))
    expected_run_count_per_candidate = len(tuning.partitions) * len(tuning.seeds)
    expected_total_runs = len(grid) * expected_run_count_per_candidate
    all_rows: list[dict[str, object]] = []
    stage_winners: list[dict[str, object]] = []
    frozen: dict[str, object] = {}

    for stage_index, stage_id in enumerate(stage_ids, start=1):
        stage_candidates = [
            replace(
                candidate,
                frozen_before=tuple(frozen.items()),
            )
            for candidate in grid
            if candidate.stage_id == stage_id
        ]
        stage_rows = []
        for candidate_index, candidate in enumerate(stage_candidates, start=1):
            print(
                f"[validation] stage {stage_index}/{len(stage_ids)} "
                f"{stage_id}: candidate {candidate_index}/"
                f"{len(stage_candidates)} {candidate.candidate_id}",
                flush=True,
            )
            evidence = []
            for partition_name in tuning.partitions:
                config = _candidate_config(
                    base,
                    candidate,
                    partition_name=partition_name,
                )
                for seed in tuning.seeds:
                    run_dir = (
                        output
                        / "raw"
                        / candidate.candidate_id
                        / partition_name
                        / config.stage
                        / f"seed_{seed}"
                    )
                    row = _execute_candidate_run(
                        candidate,
                        config,
                        split.bundle,
                        partitions[(partition_name, seed)],
                        seed,
                        run_dir,
                    )
                    row = dict(row)
                    row.setdefault("candidate_id", candidate.candidate_id)
                    row.setdefault("partition", partition_name)
                    row.setdefault("seed", seed)
                    evidence.append(row)
            stage_rows.append(
                _summarize_candidate(
                    candidate,
                    evidence,
                    expected_run_count_per_candidate,
                )
            )
        ranked = rank_candidate_summaries(stage_rows)
        all_rows.extend(ranked)
        _write_candidate_csv(output / "candidates.csv", all_rows)
        winner = next(
            (
                row
                for row in ranked
                if row["selection_status"] == "selected"
            ),
            None,
        )
        if winner is None:
            manifest = _validation_manifest(
                tuning=tuning,
                split=split,
                grid=grid,
                rows=all_rows,
                status="failed_no_finite_stage_winner",
                expected_total_runs=expected_total_runs,
                stage_ids=stage_ids,
            )
            _write_json(output / "validation_manifest.json", manifest)
            raise RuntimeError(f"stage {stage_id} has no finite complete candidate")
        stage_winners.append(winner)
        _freeze_value(frozen, winner)

    freeze_selected_parameters(
        output,
        frozen_values=frozen,
        stage_winners=stage_winners,
    )
    manifest = _validation_manifest(
        tuning=tuning,
        split=split,
        grid=grid,
        rows=all_rows,
        status="completed",
        expected_total_runs=expected_total_runs,
        stage_ids=stage_ids,
    )
    _write_json(output / "validation_manifest.json", manifest)
    return output


def _validation_manifest(
    *,
    tuning: TuningConfig,
    split: Any,
    grid: list[Candidate],
    rows: list[Mapping[str, object]],
    status: str,
    expected_total_runs: int,
    stage_ids: list[str],
) -> dict[str, object]:
    model_indices = np.asarray(split.model_indices, dtype=np.int64)
    validation_indices = np.asarray(split.validation_indices, dtype=np.int64)
    firewall_evidence = split.firewall_evidence()
    return {
        "status": status,
        "config_path": str(tuning.config_path),
        "dataset_path": split.bundle.source_path,
        "dataset_sha256": split.bundle.sha256,
        "split_seed": tuning.split_seed,
        "rounds": tuning.rounds,
        "partitions": list(tuning.partitions),
        "seeds": list(tuning.seeds),
        "candidate_stage_order": stage_ids,
        "expected_candidate_count": len(grid),
        "completed_candidate_count": len(rows),
        "expected_run_count": expected_total_runs,
        "evaluated_run_count": sum(
            int(row["evaluated_run_count"]) for row in rows
        ),
        "completed_run_count": sum(
            int(row["completed_run_count"]) for row in rows
        ),
        "failed_run_count": sum(
            int(row["evaluated_run_count"]) - int(row["completed_run_count"])
            for row in rows
        ),
        "finite_run_count": sum(int(row["finite_run_count"]) for row in rows),
        "test_set_accessed": bool(
            firewall_evidence["forbidden_access_count"]
        ),
        "firewall_evidence": firewall_evidence,
        "metric_scope": "inner_16_percent_validation_only",
        "outer_final_test_metrics": "inaccessible_and_not_read",
        "split_audit": {
            "model_indices": model_indices.tolist(),
            "validation_indices": validation_indices.tolist(),
            "model_indices_sha256": _sha256_indices(model_indices),
            "validation_indices_sha256": _sha256_indices(validation_indices),
            "model_count": int(model_indices.size),
            "validation_count": int(validation_indices.size),
            "indices_disjoint": bool(
                set(model_indices.tolist()).isdisjoint(
                    validation_indices.tolist()
                )
            ),
        },
        "ranking_rule": RANKING_RULE,
        "deterministic_tie_handling": (
            "Declaration order is used only after an exact tie on every "
            "declared ranking metric."
        ),
        "histories": "raw_unmodified_per_run_artifacts",
        "curve_postprocessing": "none",
        "beta": "theory_derived_per_graph_seed_and_rho_not_tuned",
    }
