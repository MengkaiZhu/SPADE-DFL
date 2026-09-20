"""Guarded ten-seed joint validation for Alzheimer SPADE-DFL v3."""

from __future__ import annotations

import csv
import hashlib
import itertools
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
from .alzheimer_v3_protocol import (
    V3_PROTOCOL,
    derive_v3_beta,
    v3_baseline_multiplier_values,
    v3_spade_joint_values,
    validate_v3_common_config,
)
from .alzheimer_validation_firewall import ValidationOnlySplitView
from .artifacts import save_run
from .config import ExperimentConfig, ParameterRecord
from .graph import build_graph
from .methods import method_config, run_method
from .metrics import paper_stationarity_consensus
from .objectives import build_objective
from .partitioning import PartitionResult
from .trainer import RunResult

_PARTITIONS = ("dir_0.3", "dir_0.6", "iid", "pathological")
_BASELINE_CENTERS = {
    "zo2_lt_admm_vr": 0.1,
    "zo2_lt_admm": 0.1,
    "zo2_dp_sgd": 0.05,
    "zo2_dp_fedavg": 0.05,
    "one_point_dsg": 0.05,
    "one_point_dsgt": 1.5 / 39,
}
_RANKING_RULE = (
    "complete finite runs; partition win count; minimum partition accuracy "
    "margin; mean partition accuracy margin; mean validation accuracy; "
    "validation loss; paper-average stationarity-consensus; function-query "
    "count; source-center preference; declaration order"
)


@dataclass(frozen=True)
class V3Candidate:
    """One preregistered v3 candidate."""

    candidate_id: str
    declaration_order: int
    method: str
    source_type: str
    source_locator: str
    source_center: bool = False
    eta_multiplier: float | None = None
    eta_value: float | None = None
    mu_value: float | None = None
    beta_factor: float | None = None


@dataclass(frozen=True)
class V3TuningConfig:
    experiment: ExperimentConfig
    split_seed: int
    partitions: tuple[str, ...]
    seeds: tuple[int, ...]
    rounds: int
    binding_protocol: bool
    config_path: Path


def _number_id(value: float) -> str:
    return format(value, ".10g").replace("-", "n").replace(".", "p")


def v3_candidate_grid() -> tuple[V3Candidate, ...]:
    """Materialize the exact 18 SPADE and 30 baseline candidates."""
    candidates: list[V3Candidate] = []
    spade = v3_spade_joint_values()
    for eta_multiplier, mu_value, beta_factor in itertools.product(
        spade["eta_multipliers"],
        spade["mu_values"],
        spade["beta_factors"],
    ):
        candidates.append(
            V3Candidate(
                candidate_id=(
                    f"spade_eta_m{_number_id(eta_multiplier)}"
                    f"_mu_{_number_id(mu_value)}"
                    f"_beta_{_number_id(beta_factor)}"
                ),
                declaration_order=len(candidates),
                method="spade_dfl",
                source_type="Validation",
                source_locator=(
                    "Alzheimer v3 design section 5, preregistered joint grid"
                ),
                source_center=(
                    eta_multiplier == 0.25
                    and mu_value == 3.5
                    and beta_factor == 1.5
                ),
                eta_multiplier=float(eta_multiplier),
                eta_value=float((1.5 / 39) * eta_multiplier),
                mu_value=float(mu_value),
                beta_factor=float(beta_factor),
            )
        )
    for method, multipliers in v3_baseline_multiplier_values().items():
        center = _BASELINE_CENTERS[method]
        for multiplier in multipliers:
            candidates.append(
                V3Candidate(
                    candidate_id=f"{method}_eta_m{_number_id(multiplier)}",
                    declaration_order=len(candidates),
                    method=method,
                    source_type="Validation",
                    source_locator=(
                        "Alzheimer v3 design section 5, exact baseline grid"
                    ),
                    source_center=multiplier == 1.0,
                    eta_multiplier=float(multiplier),
                    eta_value=float(center * multiplier),
                )
            )
    return tuple(candidates)


def _baseline_ranking_key(row: Mapping[str, object]) -> tuple[object, ...]:
    if row.get("finite") is not True:
        return (
            1,
            0.0,
            0.0,
            math.inf,
            math.inf,
            2**63 - 1,
            1,
            int(row["declaration_order"]),
        )
    return (
        0,
        -float(row["mean_validation_accuracy"]),
        -float(row["worst_partition_validation_accuracy"]),
        float(row["mean_validation_loss"]),
        float(row["mean_paper_stationarity_consensus"]),
        int(row["total_query_count"]),
        0 if row.get("source_center") is True else 1,
        int(row["declaration_order"]),
    )


def _rank_baselines(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    winners: dict[str, dict[str, object]] = {}
    for method in ALZHEIMER_METHOD_IDS:
        if method == "spade_dfl":
            continue
        ranked = sorted(
            (dict(row) for row in grouped[method]), key=_baseline_ranking_key
        )
        if not ranked or ranked[0].get("finite") is not True:
            raise RuntimeError(f"v3 baseline {method} has no complete finite candidate")
        winner_id = str(ranked[0]["candidate_id"])
        for rank, row in enumerate(ranked, start=1):
            row["rank_within_method"] = rank
            row["winner_candidate_id"] = winner_id
            row["selection_status"] = (
                "selected" if row["candidate_id"] == winner_id else "rejected"
            )
        winners[method] = ranked[0]
    return winners


def _best_baseline_by_partition(
    baseline_winners: Mapping[str, Mapping[str, object]],
) -> dict[str, float]:
    if set(baseline_winners) != set(ALZHEIMER_METHOD_IDS) - {"spade_dfl"}:
        raise ValueError("baseline winners must cover all six baselines")
    return {
        partition: max(
            float(winner["partition_accuracy_means"][partition])
            for winner in baseline_winners.values()
            if winner.get("finite") is True
        )
        for partition in _PARTITIONS
    }


def rank_v3_spade_candidates(
    rows: Sequence[Mapping[str, object]],
    baseline_winners: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Rank SPADE candidates against frozen baseline winners."""
    comparator = _best_baseline_by_partition(baseline_winners)
    enriched: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        finite = row.get("finite") is True
        partition_means = row.get("partition_accuracy_means", {})
        complete = (
            finite
            and isinstance(partition_means, Mapping)
            and set(partition_means) == set(_PARTITIONS)
        )
        if complete:
            margins = {
                partition: float(partition_means[partition])
                - comparator[partition]
                for partition in _PARTITIONS
            }
            row.update(
                partition_margins=margins,
                partition_win_count=sum(
                    margin > 0.0 for margin in margins.values()
                ),
                minimum_partition_margin=min(margins.values()),
                mean_partition_margin=float(np.mean(list(margins.values()))),
            )
        else:
            row.update(
                finite=False,
                partition_margins={},
                partition_win_count=-1,
                minimum_partition_margin=-math.inf,
                mean_partition_margin=-math.inf,
            )
        enriched.append(row)

    def key(row: Mapping[str, object]) -> tuple[object, ...]:
        if row.get("finite") is not True:
            return (1, 0, 0.0, 0.0, 0.0, math.inf, math.inf, 2**63 - 1, 1, int(row["declaration_order"]))
        return (
            0,
            -int(row["partition_win_count"]),
            -float(row["minimum_partition_margin"]),
            -float(row["mean_partition_margin"]),
            -float(row["mean_validation_accuracy"]),
            float(row["mean_validation_loss"]),
            float(row["mean_paper_stationarity_consensus"]),
            int(row["total_query_count"]),
            0 if row.get("source_center") is True else 1,
            int(row["declaration_order"]),
        )

    ranked = sorted(enriched, key=key)
    winner_id = (
        str(ranked[0]["candidate_id"])
        if ranked and ranked[0].get("finite") is True
        else ""
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["winner_candidate_id"] = winner_id
        row["selection_status"] = (
            "selected" if row["candidate_id"] == winner_id else "rejected"
        )
    return ranked


def validation_gate(selected: Mapping[str, object]) -> bool:
    """Require four present, finite, and strictly positive partition margins."""
    margins = selected.get("partition_margins")
    return (
        isinstance(margins, Mapping)
        and set(margins) == set(_PARTITIONS)
        and all(
            isinstance(margins[name], (float, int))
            and math.isfinite(float(margins[name]))
            and float(margins[name]) > 0.0
            for name in _PARTITIONS
        )
    )


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _resolve_data_path(
    config_path: Path, config: ExperimentConfig
) -> ExperimentConfig:
    data_path = Path(str(config.data_path))
    if not data_path.is_absolute():
        data_path = config_path.parent / data_path
    return replace(config, data_path=str(data_path.resolve()))


def _load_v3_tuning_config(path: Path) -> V3TuningConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("experiment"), dict):
        raise TypeError("v3 tuning config must contain an experiment object")
    partitions_raw = raw.get("partitions")
    seeds_raw = raw.get("seeds")
    if not isinstance(partitions_raw, list) or not isinstance(seeds_raw, list):
        raise TypeError("v3 partitions and seeds must be lists")
    partitions = tuple(str(value) for value in partitions_raw)
    seeds = tuple(_as_int(value, "seed") for value in seeds_raw)
    split_seed = _as_int(raw.get("split_seed"), "split_seed")
    rounds = _as_int(raw.get("rounds"), "rounds")
    binding = raw.get("binding_protocol", True)
    if not isinstance(binding, bool):
        raise TypeError("binding_protocol must be boolean")
    if binding and (
        partitions != _PARTITIONS
        or seeds != V3_PROTOCOL.validation_seeds
        or split_seed != 1
        or rounds != V3_PROTOCOL.rounds
    ):
        raise ValueError(
            "binding v3 validation requires four canonical partitions, "
            "seeds 0-9, split seed 1, and 50 rounds"
        )
    experiment = _resolve_data_path(
        config_path, ExperimentConfig.from_dict(raw["experiment"])
    )
    experiment = replace(experiment, n_rounds=rounds, seeds=seeds)
    if (
        experiment.dataset_name != "alzheimer"
        or experiment.dimension != 39
        or experiment.feature_dimension != 39
    ):
        raise ValueError("v3 tuning requires the audited 39D Alzheimer encoding")
    validate_v3_common_config(experiment, "spade_dfl")
    return V3TuningConfig(
        experiment=experiment,
        split_seed=split_seed,
        partitions=partitions,
        seeds=seeds,
        rounds=rounds,
        binding_protocol=binding,
        config_path=config_path,
    )


def _record(
    sources: dict[str, ParameterRecord],
    name: str,
    value: object,
    source_type: str,
    source: str,
) -> None:
    sources[name] = ParameterRecord(name, value, source_type, source)


def _candidate_config(
    base: ExperimentConfig,
    candidate: V3Candidate,
    partition_name: str,
) -> ExperimentConfig:
    mode, alpha = _partition_parameters(partition_name)
    config = method_config(
        base, candidate.method, tuple(base.seeds), mode, alpha
    )
    common: dict[str, object] = {
        "objective_model": V3_PROTOCOL.objective_model,
        "regularization": V3_PROTOCOL.regularization,
        "sigma_u": V3_PROTOCOL.sigma_u,
        "query_noise_std": V3_PROTOCOL.query_noise_std,
        "initialization": V3_PROTOCOL.initialization,
        "shared_initialization": False,
        "tau": V3_PROTOCOL.tau,
        "rho": V3_PROTOCOL.rho,
        "beta": 0.0,
        "clip_norm": base.clip_norm,
    }
    if candidate.method == "spade_dfl":
        common.update(
            eta0=candidate.eta_value,
            mu0=candidate.mu_value,
            batch_size=V3_PROTOCOL.saga_batch_size,
            vr_mode=V3_PROTOCOL.vr_mode,
        )
    else:
        common["eta0"] = candidate.eta_value
    config = replace(config, **common)
    sources = dict(config.parameter_sources)
    common_sources = {
        "objective_model": (
            "Reference",
            "Reference [13], Sec. IV nonconvex logistic objective",
        ),
        "regularization": (
            "Reference",
            "Reference [13], Sec. IV epsilon=0.01",
        ),
        "sigma_u": (
            "ProtocolAssumption",
            "SPADE Assumption 3 zero-variance special case",
        ),
        "query_noise_std": (
            "ProtocolAssumption",
            "SPADE Assumption 3 zero-variance special case",
        ),
        "initialization": (
            "ProtocolAssumption",
            "Alzheimer v3 design section 4 shared-zero protocol",
        ),
        "tau": ("Reference", "Reference [13], Fig. 1 tau=8"),
        "rho": ("Reference", "Reference [13], Table V penalty center"),
    }
    for name, (source_type, source) in common_sources.items():
        _record(sources, name, getattr(config, name), source_type, source)
    _record(
        sources,
        "dimension",
        config.dimension,
        "ProtocolAssumption",
        (
            "Alzheimer dataset preprocessing: 15 standardized continuous, "
            "15 binary, 8 one-hot categorical, and 1 intercept feature"
        ),
    )
    _record(
        sources,
        "feature_dimension",
        config.feature_dimension,
        "ProtocolAssumption",
        "Alzheimer preprocessing output dimension verified from train-only transform",
    )
    _record(
        sources,
        "n_rounds",
        config.n_rounds,
        "ProtocolAssumption",
        "Alzheimer supplemental communication-round budget frozen at 50",
    )
    _record(
        sources,
        "seeds",
        config.seeds,
        "Reference" if len(config.seeds) == 10 else "ProtocolAssumption",
        (
            "Reference [13], Sec. IV ten Monte Carlo repetitions"
            if len(config.seeds) == 10
            else "Alzheimer local verification uses seeds 0, 1, and 2"
        ),
    )
    _record(
        sources,
        "clip_norm",
        config.clip_norm,
        "Validation",
        (
            "Alzheimer v2 frozen validation contract retains clipping norm "
            "C=1 without outer-test adjustment"
        ),
    )
    _record(
        sources,
        "eta0",
        config.eta0,
        candidate.source_type,
        candidate.source_locator,
    )
    if candidate.method == "spade_dfl":
        _record(
            sources,
            "mu0",
            config.mu0,
            candidate.source_type,
            candidate.source_locator,
        )
        _record(
            sources,
            "batch_size",
            config.batch_size,
            "Validation",
            "Alzheimer v2 guarded midpoint selection between 8 and 16",
        )
        _record(
            sources,
            "vr_mode",
            config.vr_mode,
            "SPADE-DFL",
            "SPADE-DFL Sec. X-A persistent SAGA variant",
        )
    if candidate.method not in {"zo2_dp_sgd", "zo2_dp_fedavg"}:
        for name in ("target_epsilon", "target_delta"):
            _record(
                sources,
                name,
                getattr(config, name),
                "ProtocolAssumption",
                "Inactive privacy budget for non-DP method",
            )
    config = replace(config, parameter_sources=sources)
    validate_v3_common_config(config, candidate.method)
    return config


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    validation_loss = float(
        objective.value(average_state, bundle.test_x, bundle.test_y, u=1.0)
    )
    accuracy = float(
        np.asarray(result.history["accuracy"], dtype=np.float64)[-1]
    )
    stationarity = paper_stationarity_consensus(
        np.asarray(result.history["stationarity"], dtype=np.float64),
        np.asarray(result.history["consensus_error"], dtype=np.float64),
    )
    history_values = [
        np.asarray(values, dtype=np.float64)
        for values in result.history.values()
    ]
    finite = bool(
        np.all(np.isfinite(result.states))
        and all(np.all(np.isfinite(values)) for values in history_values)
        and np.isfinite(validation_loss)
        and np.isfinite(accuracy)
        and np.isfinite(stationarity)
    )
    return {
        "status": "completed",
        "finite": finite,
        "validation_accuracy": accuracy,
        "validation_loss": validation_loss,
        "paper_stationarity_consensus": stationarity,
        "query_count": int(result.query_count),
        "config_sha256": _canonical_sha256(config.to_dict()),
    }


def _execute_candidate_run(
    candidate: V3Candidate,
    config: ExperimentConfig,
    bundle: Any,
    partition: PartitionResult,
    seed: int,
    run_root: Path,
) -> dict[str, object]:
    """Execute one declared run and save its unaltered history."""
    try:
        if candidate.method == "spade_dfl":
            graph = build_graph(
                config.n_clients,
                topology=config.graph_topology,
                probability=config.graph_probability,
                seed=seed,
            )
            lambda_u = float(graph.laplacian_eigenvalues[-1])
            beta = derive_v3_beta(
                float(candidate.beta_factor),
                tau=config.tau,
                lambda_u=lambda_u,
                rho=config.rho,
            )
            sources = dict(config.parameter_sources)
            _record(
                sources,
                "beta",
                beta,
                "TheoryDerived",
                "SPADE-DFL Eq. (31): c_beta/(tau*lambda_u*rho)",
            )
            config = replace(config, beta=beta, parameter_sources=sources)
        validate_v3_common_config(config, candidate.method)
        validate_final_run_protocol(
            config,
            method=candidate.method,
            bundle=bundle,
            partition=partition,
        )
        result = run_method(candidate.method, config, bundle, partition, seed)
        metadata = {
            "name": "alzheimer",
            "sha256": bundle.sha256,
            "source_path": bundle.source_path,
            "test_role": "inner_validation_only",
            "protocol_version": "alzheimer_v3",
            "candidate_id": candidate.candidate_id,
            "partition": run_root.name,
            "partition_mode": partition.mode,
            "dirichlet_alpha": partition.alpha,
            "partition_seed": seed,
            "label_histograms": partition.label_histograms.tolist(),
            "config_sha256": _canonical_sha256(config.to_dict()),
            "beta_factor": candidate.beta_factor,
        }
        saved = save_run(result, config, run_root, metadata)
        expected = run_root / config.stage / f"seed_{seed}"
        if saved != expected:
            raise RuntimeError("v3 candidate artifact path mismatch")
        return _metrics_from_result(result, config, bundle)
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
            "validation_accuracy": None,
            "validation_loss": None,
            "paper_stationarity_consensus": None,
            "query_count": 0,
            "config_sha256": "",
            "failure_reason": type(error).__name__,
            "failure_message": str(error),
        }


def _summarize_candidate(
    candidate: V3Candidate,
    evidence: Sequence[Mapping[str, object]],
    expected: int,
) -> dict[str, object]:
    finite_rows = [
        row
        for row in evidence
        if row.get("status") == "completed" and row.get("finite") is True
    ]
    finite = len(evidence) == expected and len(finite_rows) == expected
    by_partition: dict[str, list[float]] = defaultdict(list)
    for row in finite_rows:
        by_partition[str(row["partition"])].append(
            float(row["validation_accuracy"])
        )
    if finite and set(by_partition) == set(_PARTITIONS):
        partition_means = {
            partition: float(np.mean(by_partition[partition]))
            for partition in _PARTITIONS
        }
        metrics: dict[str, object] = {
            "partition_accuracy_means": partition_means,
            "mean_validation_accuracy": float(
                np.mean([float(row["validation_accuracy"]) for row in finite_rows])
            ),
            "worst_partition_validation_accuracy": min(
                partition_means.values()
            ),
            "mean_validation_loss": float(
                np.mean([float(row["validation_loss"]) for row in finite_rows])
            ),
            "mean_paper_stationarity_consensus": float(
                np.mean(
                    [
                        float(row["paper_stationarity_consensus"])
                        for row in finite_rows
                    ]
                )
            ),
            "total_query_count": sum(
                int(row["query_count"]) for row in finite_rows
            ),
        }
    else:
        metrics = {
            "partition_accuracy_means": {},
            "mean_validation_accuracy": None,
            "worst_partition_validation_accuracy": None,
            "mean_validation_loss": None,
            "mean_paper_stationarity_consensus": None,
            "total_query_count": 0,
        }
    summary = asdict(candidate)
    summary.update(
        finite=finite,
        expected_run_count=expected,
        evaluated_run_count=len(evidence),
        completed_run_count=sum(
            row.get("status") == "completed" for row in evidence
        ),
        finite_run_count=len(finite_rows),
        config_sha256_values=sorted(
            {
                str(row["config_sha256"])
                for row in finite_rows
                if row.get("config_sha256")
            }
        ),
        **metrics,
    )
    return summary


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".part")
    staging.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    staging.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = sorted({name for row in rows for name in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for name, value in row.items()
                }
            )


def _sha256_indices(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<i8").tobytes()
    ).hexdigest()


def run_v3_validation(config_path: Path, output_root: Path) -> Path:
    """Run the one-shot, structurally test-free v3 validation."""
    output = Path(output_root).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("v3 validation output root is nonempty")
    tuning = _load_v3_tuning_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    loader_result = build_inner_validation_split(
        Path(str(tuning.experiment.data_path)),
        split_seed=tuning.split_seed,
    )
    split = ValidationOnlySplitView.from_loader_result(loader_result)
    del loader_result
    partitions = {
        (partition_name, seed): _build_partition(
            split.bundle.train_y,
            n_clients=V3_PROTOCOL.n_clients,
            partition=partition_name,
            seed=seed,
        )
        for partition_name in tuning.partitions
        for seed in tuning.seeds
    }
    expected_per_candidate = len(tuning.partitions) * len(tuning.seeds)
    summaries: list[dict[str, object]] = []
    evidence_count = 0
    for candidate in v3_candidate_grid():
        evidence: list[dict[str, object]] = []
        for partition_name in tuning.partitions:
            config = _candidate_config(
                tuning.experiment, candidate, partition_name
            )
            for seed in tuning.seeds:
                row = dict(
                    _execute_candidate_run(
                        candidate,
                        config,
                        split.bundle,
                        partitions[(partition_name, seed)],
                        seed,
                        output
                        / "raw"
                        / candidate.candidate_id
                        / partition_name,
                    )
                )
                row.update(
                    candidate_id=candidate.candidate_id,
                    method=candidate.method,
                    partition=partition_name,
                    seed=seed,
                )
                evidence.append(row)
                evidence_count += 1
        summaries.append(
            _summarize_candidate(
                candidate, evidence, expected_per_candidate
            )
        )

    baseline_rows = [
        row for row in summaries if row["method"] != "spade_dfl"
    ]
    baseline_winners = _rank_baselines(baseline_rows)
    spade_ranked = rank_v3_spade_candidates(
        [row for row in summaries if row["method"] == "spade_dfl"],
        baseline_winners,
    )
    selected = spade_ranked[0]
    gate_passed = validation_gate(selected)
    _write_csv(output / "candidates.csv", [*baseline_rows, *spade_ranked])
    _write_json_atomic(
        output / "baseline_winners.json",
        {"status": "frozen", "winners": baseline_winners},
    )
    _write_json_atomic(
        output / "selected_parameters.json",
        {
            "status": "frozen" if gate_passed else "validation_gate_failed",
            "selection_scope": "inner_training_validation_only",
            "ranking_rule": _RANKING_RULE,
            "validation_gate_passed": gate_passed,
            "selected_spade_candidate": selected,
            "baseline_winners": baseline_winners,
        },
    )
    firewall = split.firewall_evidence()
    completed = sum(
        int(row["completed_run_count"]) for row in summaries
    )
    finite = sum(int(row["finite_run_count"]) for row in summaries)
    manifest = {
        "status": "completed",
        "protocol_version": "alzheimer_v3",
        "test_set_accessed": bool(firewall["forbidden_access_count"]),
        "forbidden_access_count": firewall["forbidden_access_count"],
        "firewall_evidence": firewall,
        "objective_model": V3_PROTOCOL.objective_model,
        "initialization": V3_PROTOCOL.initialization,
        "rounds": tuning.rounds,
        "partitions": list(tuning.partitions),
        "seeds": list(tuning.seeds),
        "expected_candidate_count": len(summaries),
        "expected_run_count": len(summaries) * expected_per_candidate,
        "evaluated_run_count": evidence_count,
        "completed_run_count": completed,
        "finite_run_count": finite,
        "split_audit": {
            "model_indices_sha256": _sha256_indices(split.model_indices),
            "validation_indices_sha256": _sha256_indices(
                split.validation_indices
            ),
            "indices_disjoint": set(split.model_indices.tolist()).isdisjoint(
                split.validation_indices.tolist()
            ),
        },
    }
    _write_json_atomic(output / "validation_manifest.json", manifest)
    return output
