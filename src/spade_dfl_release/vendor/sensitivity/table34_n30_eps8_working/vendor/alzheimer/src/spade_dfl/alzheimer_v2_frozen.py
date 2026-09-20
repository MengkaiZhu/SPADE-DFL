"""Fail-closed frozen configuration contract for the Alzheimer v2 suite."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .alzheimer_contract import ALZHEIMER_METHOD_IDS, validate_alzheimer_method_contract
from .alzheimer_v2_protocol import V2_PROTOCOL, validate_v2_common_config
from .config import ExperimentConfig, ParameterRecord

_LOCK_NAME = "alzheimer_v2_selection_lock.json"
_CONFIG_NAMES = (
    "alzheimer_v2_base.json",
    "alzheimer_v2_local_3seed_50round.json",
    "alzheimer_v2_server_30seed_50round.json",
    "alzheimer_v2_package_smoke.json",
)
_PARTITIONS = ("dir_0.3", "dir_0.6", "iid", "pathological")
_ALLOWED_SOURCE_TYPES = {
    "Reference",
    "TheoryDerived",
    "Validation",
    "ProtocolAssumption",
}
_REQUIRED_TOP_LEVEL = {
    "schema_version",
    "protocol_version",
    "protocol_role",
    "evaluation_scope",
    "output_label",
    "methods",
    "partitions",
    "seeds",
    "rounds",
    "split_seed",
    "experiment",
    "method_parameters",
    "parameter_sources",
    "config_source_records",
    "selection_provenance",
}
_BASELINE_STAGES = {
    "zo2_lt_admm_vr": "baseline_eta:zo2_lt_admm_vr",
    "zo2_lt_admm": "baseline_eta:zo2_lt_admm",
    "zo2_dp_sgd": "baseline_eta:zo2_dp_sgd",
    "zo2_dp_fedavg": "baseline_eta:zo2_dp_fedavg",
    "one_point_dsg": "baseline_eta:one_point_dsg",
    "one_point_dsgt": "baseline_eta:one_point_dsgt",
}


@dataclass(frozen=True)
class FrozenAlzheimerSuiteConfig:
    """Validated v2 configuration plus its locked provenance."""

    experiment: ExperimentConfig
    methods: tuple[str, ...]
    partitions: tuple[str, ...]
    seeds: tuple[int, ...]
    rounds: int
    split_seed: int
    protocol_role: str
    evaluation_scope: str
    output_label: str
    method_parameters: Mapping[str, Mapping[str, Any]]
    parameter_sources: Mapping[str, Mapping[str, Any]]
    selection_provenance: Mapping[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read required v2 JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"v2 JSON artifact must be an object: {path}")
    return value


def _record(name: str, value: Any, source_type: str, source: str) -> dict[str, Any]:
    if source_type not in _ALLOWED_SOURCE_TYPES or not source:
        raise ValueError(f"invalid source record for {name}")
    return {
        "record_id": f"v2:{name}",
        "value": value,
        "source_type": source_type,
        "source": source,
    }


def _as_parameter_record(name: str, raw: Mapping[str, Any]) -> ParameterRecord:
    _validate_source_record(name, raw)
    return ParameterRecord(
        name=name,
        value=raw["value"],
        source_type=str(raw["source_type"]),
        source=str(raw["source"]),
    )


def _validate_source_record(name: str, raw: object) -> None:
    if not isinstance(raw, Mapping):
        raise TypeError(f"source record for {name} must be an object")
    if raw.get("source_type") not in _ALLOWED_SOURCE_TYPES:
        raise ValueError(f"source record for {name} has an invalid source_type")
    if not raw.get("source") or "value" not in raw:
        raise ValueError(f"source record for {name} is incomplete")


def _relative_path(target: Path, directory: Path) -> str:
    try:
        return os.path.relpath(str(target.resolve()), str(directory.resolve()))
    except ValueError:
        # Pytest temp directories can be on a different Windows drive.
        return str(target.resolve())


def _resolve_lock_artifact(lock_path: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("v2 selection lock artifact path is missing")
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else (lock_path.parent / candidate).resolve()


def _protocol_record_map() -> dict[str, Any]:
    values = {record.name: record for record in V2_PROTOCOL.common_source_records}
    return {
        "objective_model": values["objective_model"],
        "regularization": values["regularization"],
        "sigma_u": values["sigma_u"],
        "query_noise_std": values["query_noise_std"],
        "initialization": values["initialization"],
        "tau": values["local_training_tau"],
    }


def _tuning_source(name: str, value: Any, tuning_sources: Mapping[str, Any]) -> dict[str, Any]:
    raw = tuning_sources.get(name)
    if isinstance(raw, Mapping) and raw.get("source"):
        source_type = str(raw.get("source_type", "ProtocolAssumption"))
        if source_type == "Assumed":
            source_type = "ProtocolAssumption"
        if source_type in _ALLOWED_SOURCE_TYPES:
            return _record(name, value, source_type, str(raw["source"]))
    return _record(
        name,
        value,
        "ProtocolAssumption",
        "Binding v2 value from configs/alzheimer_v2_tuning.json",
    )


def _winner_map(selected: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if selected.get("status") != "frozen":
        raise ValueError("selected parameters must be frozen")
    rows = selected.get("stage_winners")
    if not isinstance(rows, list):
        raise TypeError("selected parameters must contain stage_winners")
    winners: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("selection_status") != "selected":
            raise ValueError("selected parameters contain an unselected winner")
        stage = row.get("stage_id")
        if not isinstance(stage, str) or stage in winners:
            raise ValueError("selected parameters contain invalid winner stages")
        if not row.get("candidate_id") or row.get("source_type") != "Validation":
            raise ValueError("selected winner must have a Validation source")
        winners[stage] = row
    expected = {"spade_eta", "spade_rho", "spade_batch", "spade_mu", *_BASELINE_STAGES.values()}
    if set(winners) != expected:
        raise ValueError("selected parameters do not cover the exact v2 stages")
    return winners


def _selected_value(
    values: Mapping[str, Any], winners: Mapping[str, Mapping[str, Any]], key: str, stage: str
) -> tuple[Any, dict[str, Any]]:
    winner = winners[stage]
    winner_key = f"{stage.replace(':', '_')}_winner_candidate_id"
    if values.get(winner_key) != winner["candidate_id"] or key not in values:
        raise ValueError(f"selected value for {stage} is inconsistent with its winner")
    value = values[key]
    return value, _record(
        key,
        value,
        "Validation",
        str(winner["source"]),
    )


def _selected_method_parameters(
    selected: Mapping[str, Any],
) -> tuple[
    dict[str, tuple[Any, dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, Mapping[str, Any]],
]:
    """Derive every selected operational value from the admitted artifact."""
    winners = _winner_map(selected)
    values = selected.get("selected_values")
    if not isinstance(values, Mapping):
        raise TypeError("selected parameters lack selected_values")
    spade = {
        field: _selected_value(values, winners, key, stage)
        for field, key, stage in (
            ("eta0", "spade_eta0", "spade_eta"),
            ("rho", "spade_rho", "spade_rho"),
            ("batch_size", "spade_batch_size", "spade_batch"),
            ("mu0", "spade_mu0", "spade_mu"),
        )
    }
    methods: dict[str, dict[str, Any]] = {}
    for method, stage in _BASELINE_STAGES.items():
        value, record = _selected_value(values, winners, f"{method}_eta0", stage)
        methods[method] = {"eta0": value, "eta0_source": record}
    methods["spade_dfl"] = {
        "eta0": spade["eta0"][0], "eta0_source": spade["eta0"][1],
        "rho": spade["rho"][0], "rho_source": spade["rho"][1],
        "batch_size": spade["batch_size"][0], "batch_size_source": spade["batch_size"][1],
        "mu0": spade["mu0"][0], "mu0_source": spade["mu0"][1],
    }
    return spade, methods, winners


def _validate_validation_inputs(manifest: Mapping[str, Any], selected: Mapping[str, Any]) -> None:
    if manifest.get("status") != "completed":
        raise ValueError("validation manifest is not completed")
    if manifest.get("test_set_accessed") is not False:
        raise ValueError("validation manifest accessed the final test set")
    if manifest.get("forbidden_access_count") != 0:
        raise ValueError("validation manifest firewall is not zero-access")
    if manifest.get("objective_model") != V2_PROTOCOL.objective_model:
        raise ValueError("validation manifest objective does not match v2 protocol")
    if manifest.get("initialization") != V2_PROTOCOL.initialization:
        raise ValueError("validation manifest initialization does not match v2 protocol")
    if selected.get("selection_scope") != "inner_training_validation_only":
        raise ValueError("selected parameters are not validation-only")
    gate = selected.get("batch_12_gate")
    if not isinstance(gate, Mapping) or gate.get("eligible") is not True:
        raise ValueError("v2 batch-12 selection lacks its validation gate")


def _build_common_experiment(tuning: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_experiment = tuning.get("experiment")
    if not isinstance(raw_experiment, Mapping):
        raise TypeError("v2 tuning config must contain an experiment")
    experiment = dict(raw_experiment)
    tuning_sources = experiment.pop("parameter_sources", {})
    experiment.pop("output_dir", None)
    experiment.pop("seeds", None)
    if not isinstance(tuning_sources, Mapping):
        raise TypeError("v2 tuning parameter_sources must be an object")
    sources = {
        name: _tuning_source(name, value, tuning_sources)
        for name, value in experiment.items()
    }
    for name, record in _protocol_record_map().items():
        if experiment.get(name) != record.value:
            raise ValueError(f"tuning config {name} conflicts with the v2 protocol")
        sources[name] = _record(name, record.value, record.source_kind, record.source_locator)
    return experiment, sources


def _suite_payload(
    *,
    role: str,
    output_label: str,
    seeds: tuple[int, ...],
    partitions: tuple[str, ...],
    rounds: int,
    split_seed: int,
    experiment_base: Mapping[str, Any],
    sources_base: Mapping[str, Any],
    method_parameters: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    experiment = dict(experiment_base)
    sources = dict(sources_base)
    experiment["n_rounds"] = rounds
    sources["n_rounds"] = _record(
        "n_rounds", rounds, "ProtocolAssumption", "Approved v2 fair 50-round protocol"
    )
    return {
        "schema_version": 2,
        "protocol_version": "alzheimer_v2",
        "protocol_role": role,
        "evaluation_scope": "inner_validation_smoke_only" if role == "package_smoke_only" else "outer_final_test_frozen",
        "output_label": output_label,
        "methods": list(ALZHEIMER_METHOD_IDS),
        "partitions": list(partitions),
        "seeds": list(seeds),
        "rounds": rounds,
        "split_seed": split_seed,
        "experiment": experiment,
        "method_parameters": method_parameters,
        "parameter_sources": sources,
        "config_source_records": {
            "schema_version": _record("schema_version", 2, "ProtocolAssumption", "Alzheimer v2 frozen config schema"),
            "rounds": _record("rounds", rounds, "ProtocolAssumption", "Approved v2 fair 50-round protocol"),
            "split_seed": _record("split_seed", split_seed, "ProtocolAssumption", "v2 tuning split protocol"),
            "seeds": _record("suite_seeds", list(seeds), "Reference" if len(seeds) == 30 else "ProtocolAssumption", "Reference [20], Sec. 5.1 uses 30 instances" if len(seeds) == 30 else "Approved v2 local/smoke seed scale"),
        },
        "selection_provenance": provenance,
    }


def freeze_v2_configs(
    *, selected_path: Path, manifest_path: Path, tuning_config_path: Path, output_dir: Path
) -> dict[str, Path]:
    """Hash-lock approved validation inputs and derive all v2 configurations."""
    selected_path, manifest_path, tuning_config_path = (
        Path(selected_path).resolve(),
        Path(manifest_path).resolve(),
        Path(tuning_config_path).resolve(),
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = (output_dir / _LOCK_NAME, *(output_dir / name for name in _CONFIG_NAMES))
    if any(path.exists() for path in targets):
        raise ValueError("v2 freeze output already contains a locked configuration")

    selected_bytes, manifest_bytes = selected_path.read_bytes(), manifest_path.read_bytes()
    selected, manifest = json.loads(selected_bytes), json.loads(manifest_bytes)
    if not isinstance(selected, Mapping) or not isinstance(manifest, Mapping):
        raise TypeError("v2 validation artifacts must be JSON objects")
    _validate_validation_inputs(manifest, selected)
    selected_values, methods, winners = _selected_method_parameters(selected)

    tuning = _read_object(tuning_config_path)
    if tuning.get("binding_protocol") is not True or tuning.get("rounds") != 50:
        raise ValueError("v2 tuning config is not the approved binding 50-round protocol")
    if tuple(tuning.get("partitions", ())) != _PARTITIONS or tuple(tuning.get("seeds", ())) != (0, 1, 2):
        raise ValueError("v2 tuning config does not use the approved local protocol")
    experiment, sources = _build_common_experiment(tuning)
    for field, (value, record) in selected_values.items():
        experiment[field], sources[field] = value, record
    lock = {
        "schema_version": 1,
        "protocol_version": "alzheimer_v2",
        "selected_parameters_path": _relative_path(selected_path, output_dir),
        "selected_parameters_sha256": hashlib.sha256(selected_bytes).hexdigest(),
        "validation_manifest_path": _relative_path(manifest_path, output_dir),
        "validation_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    lock_path = output_dir / _LOCK_NAME
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance = {
        "selection_lock": _LOCK_NAME,
        "selection_lock_sha256": _sha256(lock_path),
        "selected_parameters_sha256": lock["selected_parameters_sha256"],
        "validation_manifest_sha256": lock["validation_manifest_sha256"],
        "winner_candidate_ids": {stage: row["candidate_id"] for stage, row in winners.items()},
    }
    base = _suite_payload(role="scientific_base", output_label="alzheimer_v2_base", seeds=(0, 1, 2), partitions=_PARTITIONS, rounds=50, split_seed=int(tuning["split_seed"]), experiment_base=experiment, sources_base=sources, method_parameters=methods, provenance=provenance)
    local = _suite_payload(role="scientific_local", output_label="alzheimer_v2_local_3seed_50round", seeds=(0, 1, 2), partitions=_PARTITIONS, rounds=50, split_seed=int(tuning["split_seed"]), experiment_base=experiment, sources_base=sources, method_parameters=methods, provenance=provenance)
    server = _suite_payload(role="scientific_server", output_label="alzheimer_v2_server_30seed_50round", seeds=tuple(range(30)), partitions=_PARTITIONS, rounds=50, split_seed=int(tuning["split_seed"]), experiment_base=experiment, sources_base=sources, method_parameters=methods, provenance=provenance)
    smoke = _suite_payload(role="package_smoke_only", output_label="alzheimer_v2_package_smoke", seeds=(0,), partitions=("iid",), rounds=1, split_seed=int(tuning["split_seed"]), experiment_base=experiment, sources_base=sources, method_parameters=methods, provenance=provenance)
    payloads = dict(zip(_CONFIG_NAMES, (base, local, server, smoke)))
    for name, payload in payloads.items():
        (output_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {_LOCK_NAME: lock_path, **{name: output_dir / name for name in _CONFIG_NAMES}}


def _validate_lock(
    config_path: Path, provenance: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = config_path.parent / _LOCK_NAME
    lock = _read_object(lock_path)
    expected = {
        "schema_version", "protocol_version", "selected_parameters_path",
        "selected_parameters_sha256", "validation_manifest_path", "validation_manifest_sha256",
    }
    if set(lock) != expected or lock.get("schema_version") != 1 or lock.get("protocol_version") != "alzheimer_v2":
        raise ValueError("v2 selection lock has an invalid schema")
    if provenance.get("selection_lock_sha256") != _sha256(lock_path):
        raise ValueError("v2 configuration does not match its selection-lock SHA")
    admitted: dict[str, dict[str, Any]] = {}
    for label in ("selected_parameters", "validation_manifest"):
        artifact = _resolve_lock_artifact(lock_path, lock[f"{label}_path"])
        if not artifact.is_file() or _sha256(artifact) != lock[f"{label}_sha256"]:
            raise ValueError(f"locked {label} SHA does not match its approved artifact")
        if provenance.get(f"{label}_sha256") != lock[f"{label}_sha256"]:
            raise ValueError(f"v2 configuration {label} SHA does not match its lock")
        admitted[label] = _read_object(artifact)
    return admitted["selected_parameters"], admitted["validation_manifest"]


def _validate_role(raw: Mapping[str, Any], seeds: tuple[int, ...], partitions: tuple[str, ...]) -> None:
    roles = {
        "scientific_base": ((0, 1, 2), _PARTITIONS, 50, "outer_final_test_frozen"),
        "scientific_local": ((0, 1, 2), _PARTITIONS, 50, "outer_final_test_frozen"),
        "scientific_server": (tuple(range(30)), _PARTITIONS, 50, "outer_final_test_frozen"),
        "package_smoke_only": ((0,), ("iid",), 1, "inner_validation_smoke_only"),
    }
    contract = roles.get(raw.get("protocol_role"))
    if contract is None or (seeds, partitions, raw.get("rounds"), raw.get("evaluation_scope")) != contract:
        raise ValueError("v2 frozen configuration violates its role contract")


def load_v2_suite_config(path: Path) -> FrozenAlzheimerSuiteConfig:
    """Strictly load a v2 suite config and re-check its two input hashes."""
    path = Path(path).resolve()
    raw = _read_object(path)
    if set(raw) != _REQUIRED_TOP_LEVEL:
        raise ValueError("v2 frozen configuration has unknown or missing fields")
    if raw.get("schema_version") != 2 or raw.get("protocol_version") != "alzheimer_v2":
        raise ValueError("unsupported v2 frozen configuration")
    methods = tuple(raw["methods"])
    validate_alzheimer_method_contract(methods)
    seeds = tuple(int(seed) for seed in raw["seeds"])
    partitions = tuple(raw["partitions"])
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds) or partitions not in {_PARTITIONS, ("iid",)}:
        raise ValueError("v2 configuration has invalid seeds or partitions")
    _validate_role(raw, seeds, partitions)
    provenance = raw["selection_provenance"]
    if not isinstance(provenance, Mapping):
        raise TypeError("v2 configuration lacks selection provenance")
    locked_selected, locked_manifest = _validate_lock(path, provenance)
    _validate_validation_inputs(locked_manifest, locked_selected)
    expected_spade_values, expected_method_parameters, _ = _selected_method_parameters(
        locked_selected
    )
    experiment_raw = raw["experiment"]
    sources = raw["parameter_sources"]
    if not isinstance(experiment_raw, Mapping) or not isinstance(sources, Mapping):
        raise TypeError("v2 configuration experiment or sources are invalid")
    if set(experiment_raw) != set(sources):
        raise ValueError("each v2 experiment field requires one source record")
    for name, (value, record) in expected_spade_values.items():
        if experiment_raw.get(name) != value or sources.get(name) != record:
            raise ValueError(
                f"v2 experiment disagrees with locked selected {name}"
            )
    if experiment_raw.get("n_rounds") != raw["rounds"]:
        raise ValueError("v2 experiment differs from suite round contract")
    for name, value in experiment_raw.items():
        record = sources[name]
        _validate_source_record(name, record)
        if record["value"] != value:
            raise ValueError(f"v2 source record disagrees with {name}")
    experiment = ExperimentConfig.from_dict({
        **experiment_raw,
        "parameter_sources": {name: _as_parameter_record(name, source) for name, source in sources.items()},
    })
    experiment = replace(experiment, seeds=seeds)
    if experiment.n_rounds != raw["rounds"]:
        raise ValueError("v2 ExperimentConfig differs from frozen suite contract")
    method_parameters = raw["method_parameters"]
    if not isinstance(method_parameters, Mapping) or tuple(method_parameters) != methods:
        raise ValueError("v2 method parameters must cover exactly seven methods")
    if method_parameters != expected_method_parameters:
        raise ValueError("v2 method parameters disagree with locked selected values")
    for method, values in method_parameters.items():
        if not isinstance(values, Mapping) or "eta0" not in values:
            raise ValueError(f"v2 method {method} lacks selected eta0")
        for name, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                source = values.get(f"{name}_source")
                _validate_source_record(f"{method}.{name}", source)
                if source["value"] != value:
                    raise ValueError(f"v2 source record disagrees with {method}.{name}")
    config_source_records = raw["config_source_records"]
    if not isinstance(config_source_records, Mapping):
        raise TypeError("v2 config_source_records must be an object")
    expected_config_values = {
        "schema_version": raw["schema_version"],
        "rounds": raw["rounds"],
        "split_seed": raw["split_seed"],
        "seeds": raw["seeds"],
    }
    if set(config_source_records) != set(expected_config_values):
        raise ValueError("v2 config_source_records do not cover every numeric run field")
    for name, record in config_source_records.items():
        _validate_source_record(name, record)
        if record["value"] != expected_config_values[name]:
            raise ValueError(f"v2 config source record disagrees with {name}")
    return FrozenAlzheimerSuiteConfig(
        experiment=experiment, methods=methods, partitions=partitions, seeds=seeds,
        rounds=int(raw["rounds"]), split_seed=int(raw["split_seed"]),
        protocol_role=str(raw["protocol_role"]), evaluation_scope=str(raw["evaluation_scope"]),
        output_label=str(raw["output_label"]), method_parameters=method_parameters,
        parameter_sources=sources, selection_provenance=provenance,
    )


def apply_v2_method_parameters(
    config: ExperimentConfig, *, suite: FrozenAlzheimerSuiteConfig, method: str
) -> ExperimentConfig:
    """Restore validation-selected values and their source records for one method."""
    if method not in suite.methods:
        raise ValueError("method is outside the canonical v2 suite")
    values = suite.method_parameters[method]
    updates = {name: values[name] for name in ("eta0", "rho", "batch_size", "mu0") if name in values}
    sources = dict(config.parameter_sources)
    for name, value in updates.items():
        record = values[f"{name}_source"]
        sources[name] = _as_parameter_record(name, record)
        if sources[name].value != value:
            raise ValueError(f"v2 method source disagrees with {method}.{name}")
    frozen = replace(config, parameter_sources=sources, **updates)
    validate_v2_common_config(frozen)
    return frozen


def final_v2_method_config(suite: FrozenAlzheimerSuiteConfig, method: str) -> ExperimentConfig:
    """Build the final common v2 configuration for one canonical method."""
    return apply_v2_method_parameters(
        replace(suite.experiment, stage=method), suite=suite, method=method
    )
