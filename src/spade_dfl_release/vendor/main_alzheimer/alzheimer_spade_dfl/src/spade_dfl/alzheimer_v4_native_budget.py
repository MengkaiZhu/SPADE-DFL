"""Method-native LT budget amendment for Alzheimer SPADE-DFL v4."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .alzheimer_data import build_inner_validation_split
from .alzheimer_suite import _build_partition
from .alzheimer_tuning_protocol import (
    validate_final_run_protocol,
    validate_parameter_source_alignment,
)
from .alzheimer_v3_tuning import (
    V3Candidate,
    V3TuningConfig,
    _baseline_ranking_key,
    _candidate_config,
    _canonical_sha256,
    _load_v3_tuning_config,
    _metrics_from_result,
    _sha256_indices,
    _summarize_candidate,
    _write_csv,
    _write_json_atomic,
    v3_candidate_grid,
)
from .alzheimer_validation_firewall import ValidationOnlySplitView
from .artifacts import save_run
from .config import ExperimentConfig, ParameterRecord
from .methods import run_method
from .partitioning import PartitionResult

_LT_METHODS = ("zo2_lt_admm_vr", "zo2_lt_admm")
_PARTITIONS = ("dir_0.3", "dir_0.6", "iid", "pathological")
_LT_TAU = 2
_LT_TAU_SOURCE = (
    "Reference [13], Sec. IV numerical algorithm comparisons use tau=2"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def v4_lt_candidate_grid() -> tuple[V3Candidate, ...]:
    """Reuse exactly the ten preregistered v3 LT learning-rate candidates."""
    return tuple(
        candidate
        for candidate in v3_candidate_grid()
        if candidate.method in _LT_METHODS
    )


def _load_v4_config(
    path: Path,
) -> tuple[V3TuningConfig, dict[str, object]]:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("binding_protocol") is not True:
        raise ValueError("v4 requires a binding protocol object")
    expected_fields = {
        "binding_protocol",
        "v3_tuning_config",
        "v3_result_root",
        "v3_artifact_sha256",
        "selected_spade_candidate_id",
        "lt_tau",
        "lt_tau_source",
    }
    if set(raw) != expected_fields:
        raise ValueError("v4 config fields do not match the frozen schema")
    if raw["lt_tau"] != _LT_TAU or raw["lt_tau_source"] != _LT_TAU_SOURCE:
        raise ValueError("v4 LT tau must be the Reference-[13] value 2")
    tuning_path = config_path.parent / str(raw["v3_tuning_config"])
    tuning = _load_v3_tuning_config(tuning_path)
    result_root = config_path.parent / str(raw["v3_result_root"])
    result_root = result_root.resolve()
    hashes = raw["v3_artifact_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != {
        "validation_manifest.json",
        "selected_parameters.json",
        "baseline_winners.json",
    }:
        raise ValueError("v4 requires all three frozen v3 artifact hashes")
    artifacts: dict[str, dict[str, object]] = {}
    for name, expected_sha in hashes.items():
        artifact = result_root / name
        if not artifact.is_file() or _sha256(artifact) != str(expected_sha):
            raise ValueError(f"v4 locked v3 artifact mismatch: {name}")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"v3 artifact {name} must contain an object")
        artifacts[name] = payload
    manifest = artifacts["validation_manifest.json"]
    selected = artifacts["selected_parameters.json"]
    if (
        manifest.get("status") != "completed"
        or manifest.get("completed_run_count") != 1920
        or manifest.get("finite_run_count") != 1920
        or manifest.get("test_set_accessed") is not False
    ):
        raise ValueError("v4 cannot admit an incomplete or test-accessing v3 run")
    spade = selected.get("selected_spade_candidate")
    if (
        not isinstance(spade, dict)
        or spade.get("candidate_id") != raw["selected_spade_candidate_id"]
    ):
        raise ValueError("v4 selected SPADE candidate does not match locked v3")
    context = {
        "config_path": config_path,
        "v3_result_root": result_root,
        "artifact_hashes": dict(hashes),
        "v3_manifest": manifest,
        "v3_selected": selected,
    }
    return tuning, context


def _v4_candidate_config(
    base: ExperimentConfig,
    candidate: V3Candidate,
    partition_name: str,
) -> ExperimentConfig:
    config = _candidate_config(base, candidate, partition_name)
    sources = dict(config.parameter_sources)
    sources["tau"] = ParameterRecord(
        "tau", _LT_TAU, "Reference", _LT_TAU_SOURCE
    )
    config = replace(config, tau=_LT_TAU, beta=0.0, parameter_sources=sources)
    validate_parameter_source_alignment(config)
    if (
        config.stage != candidate.method
        or candidate.method not in _LT_METHODS
        or config.tau != _LT_TAU
        or config.n_clients != 31
        or config.objective_model != "lt_admm_nonconvex_logistic"
        or config.regularization != 0.01
        or config.sigma_u != 0.0
        or config.query_noise_std != 0.0
        or config.initialization != "shared_zero"
    ):
        raise ValueError("v4 LT candidate departed from the frozen amendment")
    return config


def _execute_v4_candidate_run(
    candidate: V3Candidate,
    config: ExperimentConfig,
    bundle: Any,
    partition: PartitionResult,
    seed: int,
    run_root: Path,
) -> dict[str, object]:
    try:
        validate_parameter_source_alignment(config)
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
            "protocol_version": "alzheimer_v4_native_budget",
            "candidate_id": candidate.candidate_id,
            "partition": run_root.name,
            "partition_mode": partition.mode,
            "dirichlet_alpha": partition.alpha,
            "partition_seed": seed,
            "label_histograms": partition.label_histograms.tolist(),
            "config_sha256": _canonical_sha256(config.to_dict()),
        }
        saved = save_run(result, config, run_root, metadata)
        expected = run_root / config.stage / f"seed_{seed}"
        if saved != expected:
            raise RuntimeError("v4 candidate artifact path mismatch")
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


def _rank_v4_lt(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    winners: dict[str, dict[str, object]] = {}
    for method in _LT_METHODS:
        ranked = sorted(
            (dict(row) for row in grouped[method]), key=_baseline_ranking_key
        )
        if not ranked or ranked[0].get("finite") is not True:
            raise RuntimeError(f"v4 {method} has no complete finite candidate")
        winner_id = str(ranked[0]["candidate_id"])
        for rank, row in enumerate(ranked, start=1):
            row["rank_within_method"] = rank
            row["winner_candidate_id"] = winner_id
            row["selection_status"] = (
                "selected" if row["candidate_id"] == winner_id else "rejected"
            )
        winners[method] = ranked[0]
    return winners


def _combined_gate(
    spade: Mapping[str, object],
    baselines: Mapping[str, Mapping[str, object]],
) -> tuple[bool, dict[str, float]]:
    spade_means = spade["partition_accuracy_means"]
    margins = {
        partition: float(spade_means[partition])
        - max(
            float(row["partition_accuracy_means"][partition])
            for row in baselines.values()
        )
        for partition in _PARTITIONS
    }
    passed = all(
        math.isfinite(margin) and margin > 0.0 for margin in margins.values()
    )
    return passed, margins


def run_v4_native_budget_validation(
    config_path: Path, output_root: Path
) -> Path:
    """Run the one-shot 400-run method-native LT validation amendment."""
    output = Path(output_root).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("v4 validation output root is nonempty")
    tuning, context = _load_v4_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    loader_result = build_inner_validation_split(
        Path(str(tuning.experiment.data_path)), split_seed=tuning.split_seed
    )
    split = ValidationOnlySplitView.from_loader_result(loader_result)
    del loader_result
    partitions = {
        (partition_name, seed): _build_partition(
            split.bundle.train_y,
            n_clients=31,
            partition=partition_name,
            seed=seed,
        )
        for partition_name in tuning.partitions
        for seed in tuning.seeds
    }
    expected_per_candidate = len(tuning.partitions) * len(tuning.seeds)
    summaries: list[dict[str, object]] = []
    for candidate in v4_lt_candidate_grid():
        evidence: list[dict[str, object]] = []
        for partition_name in tuning.partitions:
            config = _v4_candidate_config(
                tuning.experiment, candidate, partition_name
            )
            for seed in tuning.seeds:
                row = dict(
                    _execute_v4_candidate_run(
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
        summaries.append(
            _summarize_candidate(
                candidate, evidence, expected_per_candidate
            )
        )
    lt_winners = _rank_v4_lt(summaries)
    v3_selected = context["v3_selected"]
    spade = v3_selected["selected_spade_candidate"]
    v3_baselines = v3_selected["baseline_winners"]
    combined_baselines = {
        method: (
            lt_winners[method]
            if method in _LT_METHODS
            else v3_baselines[method]
        )
        for method in v3_baselines
    }
    gate_passed, margins = _combined_gate(spade, combined_baselines)
    _write_csv(output / "candidates.csv", summaries)
    _write_json_atomic(
        output / "selected_parameters.json",
        {
            "status": "frozen" if gate_passed else "validation_gate_failed",
            "protocol_version": "alzheimer_v4_native_budget",
            "validation_gate_passed": gate_passed,
            "selected_spade_candidate": spade,
            "selected_lt_winners": lt_winners,
            "combined_baseline_winners": combined_baselines,
            "partition_margins": margins,
            "v3_artifact_sha256": context["artifact_hashes"],
        },
    )
    firewall = split.firewall_evidence()
    manifest = {
        "status": "completed",
        "protocol_version": "alzheimer_v4_native_budget",
        "test_set_accessed": bool(firewall["forbidden_access_count"]),
        "forbidden_access_count": firewall["forbidden_access_count"],
        "firewall_evidence": firewall,
        "expected_candidate_count": len(summaries),
        "expected_run_count": len(summaries) * expected_per_candidate,
        "evaluated_run_count": sum(
            int(row["evaluated_run_count"]) for row in summaries
        ),
        "completed_run_count": sum(
            int(row["completed_run_count"]) for row in summaries
        ),
        "finite_run_count": sum(
            int(row["finite_run_count"]) for row in summaries
        ),
        "partitions": list(tuning.partitions),
        "seeds": list(tuning.seeds),
        "rounds": tuning.rounds,
        "lt_tau": _LT_TAU,
        "lt_tau_source": _LT_TAU_SOURCE,
        "v3_artifact_sha256": context["artifact_hashes"],
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
