"""Guarded inner-validation parameter selection for Alzheimer v2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .alzheimer_contract import ALZHEIMER_METHOD_IDS
from .alzheimer_data import build_inner_validation_split
from .alzheimer_suite import _build_partition, _partition_parameters
from .alzheimer_tuning_protocol import validate_final_run_protocol
from .alzheimer_v2_protocol import (
    V2_PROTOCOL,
    V2SourceRecord,
    v2_candidate_values,
    validate_v2_common_config,
)
from .alzheimer_validation_firewall import ValidationOnlySplitView
from .artifacts import save_run
from .config import ExperimentConfig, ParameterRecord
from .methods import method_config, run_method
from .objectives import build_objective
from .partitioning import PartitionResult
from .trainer import RunResult


@dataclass(frozen=True)
class V2Candidate:
    """One source-backed, validation-only v2 candidate."""

    candidate_id: str
    declaration_order: int
    stage_id: str
    phase: str
    method: str
    source: str
    source_locator: str
    source_type: str
    source_center: bool = False
    eta_multiplier: float | None = None
    eta_value: float | None = None
    rho: float | None = None
    batch_size: int | None = None
    mu_multiplier: float | None = None
    mu_value: float | None = None
    conditional: bool = False
    frozen_before: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class V2TuningConfig:
    experiment: ExperimentConfig
    split_seed: int
    partitions: tuple[str, ...]
    seeds: tuple[int, ...]
    rounds: int
    binding_protocol: bool
    config_path: Path


_COMMON_RECORDS = {record.name: record for record in V2_PROTOCOL.common_source_records}
_CANDIDATE_RECORDS = {
    record.name: record for record in V2_PROTOCOL.candidate_source_records
}
_BASELINE_METHODS = tuple(method for method in ALZHEIMER_METHOD_IDS if method != "spade_dfl")
_CANONICAL_PARTITIONS = ("dir_0.3", "dir_0.6", "iid", "pathological")
_CANONICAL_SEEDS = (0, 1, 2)
_RANKING_RULE = (
    "finite complete runs; higher mean validation accuracy; higher worst-partition "
    "validation accuracy; lower validation loss; lower stationarity-consensus; "
    "lower function-query count; source-center preference after an exact metric tie"
)


def _number_id(value: float) -> str:
    return format(value, "g").replace(".", "p")


def _record(name: str) -> V2SourceRecord:
    try:
        return _CANDIDATE_RECORDS[name]
    except KeyError as error:
        raise ValueError(f"Alzheimer v2 candidate/parameter configuration lacks record {name!r}") from error


def _candidate(
    candidates: list[V2Candidate],
    *,
    record: V2SourceRecord,
    candidate_id: str,
    stage_id: str,
    phase: str,
    method: str,
    source_center: bool = False,
    **values: object,
) -> None:
    candidates.append(
        V2Candidate(
            candidate_id=candidate_id,
            declaration_order=len(candidates),
            stage_id=stage_id,
            phase=phase,
            method=method,
            source=record.source_locator,
            source_locator=record.source_locator,
            source_type=record.source_kind,
            source_center=source_center,
            **values,
        )
    )


def v2_candidate_grid() -> tuple[V2Candidate, ...]:
    """Return every unconditional candidate defined by the Alzheimer v2 candidate/parameter configuration."""
    values = v2_candidate_values()
    candidates: list[V2Candidate] = []
    eta_center = float(_record("spade_eta_center").value)
    for multiplier in values["spade_eta_multipliers"]:
        numeric = float(multiplier)
        _candidate(
            candidates,
            record=_record(f"spade_eta_multiplier:{numeric}"),
            candidate_id=f"spade_eta_m{_number_id(numeric)}",
            stage_id="spade_eta",
            phase="eta",
            method="spade_dfl",
            source_center=numeric == 1.0,
            eta_multiplier=numeric,
            eta_value=eta_center * numeric,
        )
    for rho in values["rho_values"]:
        numeric = float(rho)
        _candidate(
            candidates,
            record=_record(f"rho_candidate:{numeric}"),
            candidate_id=f"spade_rho_{_number_id(numeric)}",
            stage_id="spade_rho",
            phase="rho",
            method="spade_dfl",
            source_center=numeric == float(_record("rho_center").value),
            rho=numeric,
        )
    for batch_size in values["saga_batch_sizes"]:
        batch_size = int(batch_size)
        if batch_size == 12:
            continue
        _candidate(
            candidates,
            record=_record(f"saga_batch_size:{batch_size}"),
            candidate_id=f"spade_batch_{batch_size}",
            stage_id="spade_batch",
            phase="batch",
            method="spade_dfl",
            batch_size=batch_size,
        )
    mu_center = float(_record("spade_mu_center").value)
    for multiplier in values["mu_multipliers"]:
        numeric = float(multiplier)
        _candidate(
            candidates,
            record=_record(f"spade_mu_multiplier:{numeric}"),
            candidate_id=f"spade_mu_m{_number_id(numeric)}",
            stage_id="spade_mu",
            phase="mu",
            method="spade_dfl",
            source_center=numeric == 1.0,
            mu_multiplier=numeric,
            mu_value=mu_center * numeric,
        )
    for method in _BASELINE_METHODS:
        center = float(_record(f"baseline_eta_center:{method}").value)
        for multiplier in values["baseline_eta_multipliers"]:
            numeric = float(multiplier)
            _candidate(
                candidates,
                record=_record(f"baseline_eta_multiplier:{numeric}"),
                candidate_id=f"{method}_eta_m{_number_id(numeric)}",
                stage_id=f"baseline_eta:{method}",
                phase="baseline_eta",
                method=method,
                source_center=numeric == 1.0,
                eta_multiplier=numeric,
                eta_value=center * numeric,
            )
    return tuple(candidates)


def _batch_12_candidate(declaration_order: int) -> V2Candidate:
    record = _record("saga_batch_size:12")
    return V2Candidate(
        candidate_id="spade_batch_12",
        declaration_order=declaration_order,
        stage_id="spade_batch",
        phase="batch",
        method="spade_dfl",
        source=record.source_locator,
        source_locator=record.source_locator,
        source_type=record.source_kind,
        batch_size=12,
        conditional=True,
    )


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _resolve_data_path(config_path: Path, config: ExperimentConfig) -> ExperimentConfig:
    path = Path(str(config.data_path))
    if not path.is_absolute():
        path = config_path.parent / path
    return replace(config, data_path=str(path.resolve()))


def _load_v2_tuning_config(path: Path) -> V2TuningConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("experiment"), dict):
        raise TypeError("v2 tuning config must contain an experiment object")
    binding = raw.get("binding_protocol", True)
    if not isinstance(binding, bool):
        raise TypeError("binding_protocol must be boolean")
    partitions_raw = raw.get("partitions")
    seeds_raw = raw.get("seeds")
    if not isinstance(partitions_raw, list) or not isinstance(seeds_raw, list):
        raise TypeError("partitions and seeds must be lists")
    partitions = tuple(str(item) for item in partitions_raw)
    seeds = tuple(_as_int(item, "seed") for item in seeds_raw)
    rounds = _as_int(raw.get("rounds"), "rounds")
    split_seed = _as_int(raw.get("split_seed"), "split_seed")
    if not partitions or len(set(partitions)) != len(partitions) or any(
        item not in _CANONICAL_PARTITIONS for item in partitions
    ):
        raise ValueError("partitions must be unique canonical Alzheimer partitions")
    if not seeds or len(set(seeds)) != len(seeds) or any(item < 0 for item in seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    experiment = _resolve_data_path(
        config_path, ExperimentConfig.from_dict(raw["experiment"])
    )
    if experiment.dataset_name != "alzheimer" or experiment.dimension != 39:
        raise ValueError("v2 tuning requires the source-locked 39D Alzheimer data")
    if experiment.feature_dimension != 39 or experiment.n_clients != 31:
        raise ValueError("v2 tuning requires 39 features and 31 clients")
    experiment = replace(experiment, n_rounds=rounds, seeds=seeds)
    validate_v2_common_config(experiment)
    if binding and (
        partitions != _CANONICAL_PARTITIONS
        or seeds != _CANONICAL_SEEDS
        or rounds != 50
        or split_seed != 1
    ):
        raise ValueError("binding v2 tuning requires 50 rounds, four partitions, seeds 0/1/2, and split seed 1")
    return V2TuningConfig(
        experiment=experiment,
        split_seed=split_seed,
        partitions=partitions,
        seeds=seeds,
        rounds=rounds,
        binding_protocol=binding,
        config_path=config_path,
    )


def _common_parameter_sources(config: ExperimentConfig) -> dict[str, ParameterRecord]:
    sources = dict(config.parameter_sources)
    for name, record in _COMMON_RECORDS.items():
        config_name = "tau" if name == "local_training_tau" else name
        value = getattr(config, config_name)
        sources[config_name] = ParameterRecord(
            config_name, value, record.source_kind, record.source_locator
        )
    return sources


def _candidate_config(
    base: ExperimentConfig, candidate: V2Candidate, partition_name: str
) -> ExperimentConfig:
    mode, alpha = _partition_parameters(partition_name)
    config = method_config(base, candidate.method, tuple(base.seeds), mode, alpha)
    frozen = dict(candidate.frozen_before)
    values: dict[str, object] = {
        "objective_model": V2_PROTOCOL.objective_model,
        "regularization": V2_PROTOCOL.regularization,
        "sigma_u": V2_PROTOCOL.sigma_u,
        "query_noise_std": V2_PROTOCOL.query_noise_std,
        "initialization": V2_PROTOCOL.initialization,
        "shared_initialization": False,
        "tau": V2_PROTOCOL.local_training_tau,
        "clip_norm": base.clip_norm,
        "beta": 0.0,
    }
    if candidate.method == "spade_dfl":
        values.update(
            eta0=float(frozen.get("spade_eta0", _record("spade_eta_center").value)),
            rho=float(frozen.get("spade_rho", _record("rho_center").value)),
            batch_size=int(frozen.get("spade_batch_size", 1)),
            mu0=float(frozen.get("spade_mu0", _record("spade_mu_center").value)),
        )
        if candidate.phase == "eta":
            values["eta0"] = candidate.eta_value
        elif candidate.phase == "rho":
            values["rho"] = candidate.rho
        elif candidate.phase == "batch":
            values["batch_size"] = candidate.batch_size
        elif candidate.phase == "mu":
            values["mu0"] = candidate.mu_value
    else:
        values["eta0"] = candidate.eta_value
    config = replace(config, **values)
    sources = _common_parameter_sources(config)
    base_clip_source = base.parameter_sources.get(
        "clip_norm",
        ParameterRecord("clip_norm", base.clip_norm, "Reference", "v2 tuning base"),
    )
    sources["clip_norm"] = ParameterRecord(
        "clip_norm", config.clip_norm, base_clip_source.source_type, base_clip_source.source
    )
    candidate_value = (
        config.batch_size if candidate.phase == "batch" else config.eta0
        if candidate.phase in {"eta", "baseline_eta"} else config.rho
        if candidate.phase == "rho" else config.mu0
    )
    candidate_name = (
        "batch_size" if candidate.phase == "batch" else "eta0"
        if candidate.phase in {"eta", "baseline_eta"} else "rho"
        if candidate.phase == "rho" else "mu0"
    )
    sources[candidate_name] = ParameterRecord(
        candidate_name, candidate_value, candidate.source_type, candidate.source_locator
    )
    if candidate.method == "spade_dfl":
        frozen_source = "Frozen earlier v2 winner from guarded inner validation"
        for name in ("eta0", "rho", "batch_size", "mu0"):
            is_current = name == candidate_name
            sources[name] = ParameterRecord(
                name,
                getattr(config, name),
                candidate.source_type if is_current else "Validation",
                candidate.source_locator if is_current else frozen_source,
            )
    if candidate.method not in {"zo2_dp_sgd", "zo2_dp_fedavg"}:
        inactive_dp_source = (
            "Inactive DP release budget for this non-DP method in the exact "
            "seven-method Alzheimer v2 comparison"
        )
        for name in ("target_epsilon", "target_delta"):
            sources[name] = ParameterRecord(
                name, getattr(config, name), "ProtocolAssumption", inactive_dp_source
            )
    sources["v2_frozen_context"] = ParameterRecord(
        "v2_frozen_context", dict(candidate.frozen_before), "Validation", "Earlier v2 stage winner from guarded inner validation"
    )
    return replace(config, parameter_sources=sources)


def _metrics_from_result(
    result: RunResult, config: ExperimentConfig, bundle: Any
) -> dict[str, object]:
    objective = build_objective(
        config.objective_model,
        config.regularization,
        feature_dimension=config.input_dimension,
        n_classes=config.n_classes,
    )
    average_state = np.asarray(result.states, dtype=np.float64).mean(axis=0)
    validation_loss = float(objective.value(average_state, bundle.test_x, bundle.test_y, u=1.0))
    values = [np.asarray(item, dtype=np.float64) for item in result.history.values()]
    accuracy = float(np.asarray(result.history["accuracy"], dtype=np.float64)[-1])
    stationarity = float(np.asarray(result.history["stationarity_consensus"], dtype=np.float64)[-1])
    return {
        "status": "completed",
        "finite": bool(
            np.all(np.isfinite(result.states))
            and all(np.all(np.isfinite(item)) for item in values)
            and np.isfinite(validation_loss)
            and np.isfinite(accuracy)
            and np.isfinite(stationarity)
        ),
        "validation_accuracy": accuracy,
        "validation_loss": validation_loss,
        "stationarity_consensus": stationarity,
        "query_count": int(result.query_count),
    }


def _execute_candidate_run(
    candidate: V2Candidate,
    config: ExperimentConfig,
    bundle: Any,
    partition: PartitionResult,
    seed: int,
    run_dir: Path,
) -> dict[str, object]:
    """Execute one loss-only v2 candidate against the inner validation bundle."""
    validate_v2_common_config(config)
    validate_final_run_protocol(config, method=candidate.method, bundle=bundle, partition=partition)
    partition_name = run_dir.parents[1].name
    try:
        result = run_method(candidate.method, config, bundle, partition, seed)
        metadata = {
            "name": "alzheimer",
            "sha256": bundle.sha256,
            "source_path": bundle.source_path,
            "test_role": "inner_validation_only",
            "protocol_version": "alzheimer_v2",
            "partition": partition_name,
            "partition_mode": partition.mode,
            "dirichlet_alpha": partition.alpha,
            "partition_seed": seed,
            "label_histograms": partition.label_histograms.tolist(),
        }
        saved_dir = save_run(result, config, run_dir.parent.parent, metadata)
        if saved_dir != run_dir:
            raise RuntimeError("candidate artifact path did not match declaration")
        return _metrics_from_result(result, config, bundle)
    except (ArithmeticError, FloatingPointError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        return {
            "status": "failed",
            "finite": False,
            "validation_accuracy": None,
            "validation_loss": None,
            "stationarity_consensus": None,
            "query_count": 0,
            "failure_reason": type(error).__name__,
            "failure_message": str(error),
        }


def _summarize_candidate(
    candidate: V2Candidate, evidence: Sequence[Mapping[str, object]], expected: int
) -> dict[str, object]:
    completed = [row for row in evidence if row.get("status") == "completed"]
    finite_rows = [row for row in completed if row.get("finite") is True]
    finite = len(evidence) == expected and len(finite_rows) == expected
    by_partition: dict[str, list[float]] = defaultdict(list)
    for row in finite_rows:
        by_partition[str(row["partition"])].append(float(row["validation_accuracy"]))
    metrics: dict[str, object]
    if finite:
        metrics = {
            "mean_validation_accuracy": float(np.mean([float(row["validation_accuracy"]) for row in finite_rows])),
            "worst_partition_validation_accuracy": min(float(np.mean(items)) for items in by_partition.values()),
            "mean_validation_loss": float(np.mean([float(row["validation_loss"]) for row in finite_rows])),
            "mean_stationarity_consensus": float(np.mean([float(row["stationarity_consensus"]) for row in finite_rows])),
            "total_query_count": sum(int(row["query_count"]) for row in finite_rows),
        }
    else:
        metrics = {
            "mean_validation_accuracy": None,
            "worst_partition_validation_accuracy": None,
            "mean_validation_loss": None,
            "mean_stationarity_consensus": None,
            "total_query_count": 0,
        }
    summary = asdict(candidate)
    summary.update(
        {
            "frozen_before": dict(candidate.frozen_before),
            "expected_run_count": expected,
            "evaluated_run_count": len(evidence),
            "completed_run_count": len(completed),
            "finite_run_count": len(finite_rows),
            "finite": finite,
            "failure_reasons": [
                {"partition": row.get("partition"), "seed": row.get("seed"), "reason": row.get("failure_reason", "nonfinite_metrics")}
                for row in evidence
                if row.get("status") != "completed" or row.get("finite") is not True
            ],
            **metrics,
        }
    )
    return summary


def _ranking_key(row: Mapping[str, object]) -> tuple[object, ...]:
    finite = row.get("finite") is True
    if not finite:
        return (1, 0.0, 0.0, math.inf, math.inf, 2**63 - 1, 1, int(row["declaration_order"]))
    return (
        0,
        -float(row["mean_validation_accuracy"]),
        -float(row["worst_partition_validation_accuracy"]),
        float(row["mean_validation_loss"]),
        float(row["mean_stationarity_consensus"]),
        int(row["total_query_count"]),
        0 if row.get("source_center") is True else 1,
        int(row["declaration_order"]),
    )


def _rank_stage(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    ranked = sorted((dict(row) for row in rows), key=_ranking_key)
    winner_id = str(ranked[0]["candidate_id"]) if ranked and ranked[0].get("finite") is True else ""
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
        row["winner_candidate_id"] = winner_id
        row["selection_status"] = "selected" if row["candidate_id"] == winner_id else "rejected"
        row["rejection_reason"] = "" if row["candidate_id"] == winner_id else (
            "nonfinite_or_incomplete_runs" if row.get("finite") is not True else "out_ranked_by_stage_winner"
        )
    return ranked


def select_v2_winners(rows: Sequence[Mapping[str, object]]) -> dict:
    """Apply the predeclared deterministic ranking independently per stage."""
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    stages_by_method: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        stage = str(row["stage_id"])
        groups[stage].append(row)
        stages_by_method[str(row["method"])].add(stage)
    winners: dict[str, dict[str, object]] = {}
    for stage, stage_rows in groups.items():
        ranked = _rank_stage(stage_rows)
        winner = next((row for row in ranked if row["selection_status"] == "selected"), None)
        if winner is None:
            continue
        method = str(winner["method"])
        key = method if len(stages_by_method[method]) == 1 else stage
        winners[key] = winner
    return winners


def _freeze_winner(frozen: dict[str, object], winner: Mapping[str, object]) -> None:
    phase = str(winner["phase"])
    if phase == "eta":
        frozen["spade_eta0"] = winner["eta_value"]
    elif phase == "rho":
        frozen["spade_rho"] = winner["rho"]
    elif phase == "batch":
        frozen["spade_batch_size"] = winner["batch_size"]
    elif phase == "mu":
        frozen["spade_mu0"] = winner["mu_value"]
    else:
        frozen[f"{winner['method']}_eta0"] = winner["eta_value"]
    frozen[f"{winner['stage_id']}_winner_candidate_id".replace(":", "_")] = winner["candidate_id"]


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _run_stage(
    *, candidates: Sequence[V2Candidate], base: ExperimentConfig, split: ValidationOnlySplitView,
    partitions: Mapping[tuple[str, int], PartitionResult], tuning: V2TuningConfig,
    output: Path, frozen: Mapping[str, object],
) -> list[dict[str, object]]:
    expected = len(tuning.partitions) * len(tuning.seeds)
    summaries: list[dict[str, object]] = []
    for declared in candidates:
        candidate = replace(declared, frozen_before=tuple(sorted(frozen.items())))
        evidence: list[dict[str, object]] = []
        for partition_name in tuning.partitions:
            config = _candidate_config(base, candidate, partition_name)
            for seed in tuning.seeds:
                run_dir = output / "raw" / candidate.candidate_id / partition_name / config.stage / f"seed_{seed}"
                row = dict(_execute_candidate_run(candidate, config, split.bundle, partitions[(partition_name, seed)], seed, run_dir))
                row.update(candidate_id=candidate.candidate_id, partition=partition_name, seed=seed)
                evidence.append(row)
        summaries.append(_summarize_candidate(candidate, evidence, expected))
    return _rank_stage(summaries)


def _batch_12_gate(ranked: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Admit midpoint batch 12 only after complete finite 8/16 brackets."""
    bracket_rows = list(ranked[:2])
    bracket_ids = [str(row.get("candidate_id", "")) for row in bracket_rows]
    eligible = (
        len(bracket_rows) == 2
        and set(bracket_ids) == {"spade_batch_8", "spade_batch_16"}
        and all(
            row.get("finite") is True
            and int(row.get("completed_run_count", 0))
            == int(row.get("expected_run_count", -1))
            and int(row.get("finite_run_count", 0))
            == int(row.get("expected_run_count", -1))
            for row in bracket_rows
        )
    )
    return {"eligible": eligible, "bracket_candidate_ids": bracket_ids}


def _sha256_indices(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def run_v2_validation(config_path: Path, output_root: Path) -> Path:
    """Run v2 selection solely through the guarded inner-validation view."""
    tuning = _load_v2_tuning_config(config_path)
    output = Path(output_root).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("v2 formal output root already exists and is nonempty")
    output.mkdir(parents=True, exist_ok=True)
    loader_result = build_inner_validation_split(
        Path(str(tuning.experiment.data_path)), split_seed=tuning.split_seed
    )
    split = ValidationOnlySplitView.from_loader_result(loader_result)
    del loader_result
    base = replace(tuning.experiment, n_rounds=tuning.rounds, seeds=tuning.seeds, output_dir=str(output))
    partitions = {
        (name, seed): _build_partition(split.bundle.train_y, n_clients=base.n_clients, partition=name, seed=seed)
        for name in tuning.partitions for seed in tuning.seeds
    }
    grid = v2_candidate_grid()
    stage_order = list(dict.fromkeys(item.stage_id for item in grid))
    all_rows: list[dict[str, object]] = []
    stage_winners: list[dict[str, object]] = []
    frozen: dict[str, object] = {}
    batch_12_gate: dict[str, object] = {"eligible": False, "bracket_candidate_ids": []}
    for stage_id in stage_order:
        stage_candidates = [item for item in grid if item.stage_id == stage_id]
        ranked = _run_stage(candidates=stage_candidates, base=base, split=split, partitions=partitions, tuning=tuning, output=output, frozen=frozen)
        if stage_id == "spade_batch":
            batch_12_gate = _batch_12_gate(ranked)
            if batch_12_gate["eligible"]:
                refined = _run_stage(candidates=[_batch_12_candidate(len(grid))], base=base, split=split, partitions=partitions, tuning=tuning, output=output, frozen=frozen)
                ranked = _rank_stage([*ranked, *refined])
        all_rows.extend(ranked)
        winner = next((row for row in ranked if row["selection_status"] == "selected"), None)
        if winner is None:
            raise RuntimeError(f"v2 stage {stage_id} has no finite complete candidate")
        stage_winners.append(winner)
        _freeze_winner(frozen, winner)
        _write_csv(output / "candidates.csv", all_rows)
    firewall = split.firewall_evidence()
    selected = {
        "status": "frozen",
        "selection_scope": "inner_training_validation_only",
        "ranking_rule": _RANKING_RULE,
        "batch_12_gate": batch_12_gate,
        "selected_values": frozen,
        "stage_winners": stage_winners,
    }
    _write_json(output / "selected_parameters.json", selected)
    manifest = {
        "status": "completed",
        "test_set_accessed": bool(firewall["forbidden_access_count"]),
        "forbidden_access_count": firewall["forbidden_access_count"],
        "firewall_evidence": firewall,
        "objective_model": V2_PROTOCOL.objective_model,
        "initialization": V2_PROTOCOL.initialization,
        "rounds": tuning.rounds,
        "partitions": list(tuning.partitions),
        "seeds": list(tuning.seeds),
        "expected_candidate_count": len(all_rows),
        "expected_run_count": sum(int(row["expected_run_count"]) for row in all_rows),
        "evaluated_run_count": sum(int(row["evaluated_run_count"]) for row in all_rows),
        "completed_run_count": sum(int(row["completed_run_count"]) for row in all_rows),
        "finite_run_count": sum(int(row["finite_run_count"]) for row in all_rows),
        "candidate_stage_order": stage_order,
        "split_audit": {
            "model_indices_sha256": _sha256_indices(split.model_indices),
            "validation_indices_sha256": _sha256_indices(split.validation_indices),
            "indices_disjoint": set(split.model_indices.tolist()).isdisjoint(split.validation_indices.tolist()),
        },
    }
    _write_json(output / "validation_manifest.json", manifest)
    return output
