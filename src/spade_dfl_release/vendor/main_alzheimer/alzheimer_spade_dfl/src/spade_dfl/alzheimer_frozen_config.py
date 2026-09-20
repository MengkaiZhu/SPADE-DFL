"""Shared executable contract for frozen Alzheimer suite configurations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .alzheimer_contract import ALZHEIMER_METHOD_IDS, validate_alzheimer_method_contract
from .alzheimer_tuning_protocol import APPROVED_TUNING_PROTOCOL
from .config import ExperimentConfig, ParameterRecord

ALZHEIMER_PARTITIONS = ("dir_0.3", "dir_0.6", "iid", "pathological")
_COMMON_FIELDS = {
    "schema_version",
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
    "selection_provenance",
}
_SCIENTIFIC_ROLES = {"scientific_base", "scientific_local", "scientific_server"}


@dataclass(frozen=True)
class FrozenAlzheimerSuiteConfig:
    experiment: ExperimentConfig
    methods: tuple[str, ...]
    partitions: tuple[str, ...]
    split_seed: int
    protocol_role: str
    evaluation_scope: str
    output_label: str
    method_parameters: Mapping[str, Mapping[str, Any]]
    parameter_sources: Mapping[str, Mapping[str, Any]]
    selection_provenance: Mapping[str, Any]
    eligible_for_final_plots: bool

    @property
    def frozen(self) -> FrozenAlzheimerSuiteConfig:
        """Compatibility-facing name that makes frozen provenance explicit."""
        return self


def _read_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Alzheimer frozen configuration must be a JSON object")
    return raw


def required_experiment_parameter_names(config: Mapping[str, Any]) -> set[str]:
    experiment = config.get("experiment")
    if not isinstance(experiment, Mapping):
        raise TypeError("formal configuration must contain an experiment object")
    return set(experiment)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_payload_sha256(
    methods: Mapping[str, Any], winners: Mapping[str, Any]
) -> str:
    encoded = json.dumps(
        {"method_parameters": methods, "winner_candidate_ids": winners},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _winner_map(selected: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if selected.get("status") != "frozen":
        raise ValueError("selected parameters status must be frozen")
    rows = selected.get("stage_winners")
    if not isinstance(rows, list):
        raise TypeError("selected parameters must contain stage winners")
    return {str(row["stage_id"]): row for row in rows}


def _validation_record(value: Any, winner: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_id": f"validation:{winner['candidate_id']}",
        "value": value,
        "source_type": "Validation",
        "source": winner["source"],
    }


def _protocol_record(name: str, value: Any) -> dict[str, Any]:
    return {
        "record_id": f"protocol:alzheimer_tuning.experiment.{name}",
        "value": value,
        "source_type": "ProtocolAssumption",
        "source": "Binding Alzheimer formal-suite protocol in configs/alzheimer_tuning.json",
    }


def _selected(
    values: Mapping[str, Any], winners: Mapping[str, Mapping[str, Any]], key: str, stage: str
) -> tuple[Any, Mapping[str, Any]]:
    winner = winners[stage]
    winner_key = f"{stage.replace(':', '_')}_winner_candidate_id"
    if values.get(winner_key) != winner["candidate_id"]:
        raise ValueError(f"selected winner ID for {stage} is inconsistent")
    return values[key], winner


def _resolve_data_path(path: Path, experiment: ExperimentConfig) -> ExperimentConfig:
    data_path = Path(str(experiment.data_path))
    if not data_path.is_absolute():
        data_path = path.parent / data_path
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
    if not experiment.seeds or any(int(seed) < 0 for seed in experiment.seeds):
        raise ValueError("Alzheimer suite requires nonnegative seeds")


def _validate_partitions(partitions: object) -> tuple[str, ...]:
    if not isinstance(partitions, list):
        raise TypeError("partitions must be a JSON list")
    selected = tuple(str(value) for value in partitions)
    canonical = tuple(item for item in ALZHEIMER_PARTITIONS if item in selected)
    if not selected or selected != canonical:
        raise ValueError("partitions must be unique and in canonical Alzheimer order")
    return selected


def _record(name: str, raw: Mapping[str, Any]) -> ParameterRecord:
    return ParameterRecord(
        name=name,
        value=raw["value"],
        source_type=str(raw["source_type"]),
        source=str(raw["source"]),
    )


def load_frozen_alzheimer_suite_config(path: Path) -> FrozenAlzheimerSuiteConfig:
    """Load the sole production schema used by both freezer and suite runner."""
    path = Path(path)
    raw = _read_object(path)
    role = raw.get("protocol_role")
    expected = set(_COMMON_FIELDS)
    if role == "package_smoke_only":
        expected.add("eligible_for_final_plots")
    if set(raw) != expected:
        missing = sorted(expected - set(raw))
        extra = sorted(set(raw) - expected)
        raise ValueError(
            "invalid frozen Alzheimer configuration fields: "
            + "; ".join(
                (["missing " + ", ".join(missing)] if missing else [])
                + (["unknown " + ", ".join(extra)] if extra else [])
            )
        )
    if role not in _SCIENTIFIC_ROLES | {"package_smoke_only"}:
        raise ValueError("unknown frozen Alzheimer protocol role")
    evaluation_scope = raw["evaluation_scope"]
    expected_scope = (
        "inner_validation_smoke_only"
        if role == "package_smoke_only"
        else "outer_final_test_frozen"
    )
    if evaluation_scope != expected_scope:
        raise ValueError("frozen Alzheimer evaluation scope does not match protocol role")
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported frozen Alzheimer schema version")
    if not isinstance(raw["methods"], list):
        raise TypeError("methods must be a JSON list")
    methods = tuple(str(method) for method in raw["methods"])
    validate_alzheimer_method_contract(methods)
    partitions = _validate_partitions(raw["partitions"])
    if not isinstance(raw["seeds"], list) or not raw["seeds"]:
        raise ValueError("frozen config requires at least one seed")
    seeds = tuple(int(seed) for seed in raw["seeds"])
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("frozen config seeds must be unique nonnegative integers")
    if isinstance(raw["split_seed"], bool) or not isinstance(raw["split_seed"], int):
        raise TypeError("split_seed must be a nonnegative integer")
    if not isinstance(raw["experiment"], Mapping):
        raise TypeError("experiment must be a JSON object")
    if not isinstance(raw["parameter_sources"], Mapping):
        raise TypeError("parameter_sources must be a JSON object")
    if set(raw["experiment"]) - set(raw["parameter_sources"]):
        raise ValueError("every frozen experiment value requires a source record")
    experiment_raw = dict(raw["experiment"])
    experiment_raw["seeds"] = list(seeds)
    experiment_raw["parameter_sources"] = {
        name: _record(name, source)
        for name, source in raw["parameter_sources"].items()
        if name in experiment_raw and isinstance(source, Mapping)
    }
    experiment = _resolve_data_path(path.resolve(), ExperimentConfig.from_dict(experiment_raw))
    if experiment.n_rounds != raw["rounds"]:
        raise ValueError("rounds must equal experiment.n_rounds")
    _validate_experiment(experiment)
    method_parameters = raw["method_parameters"]
    if not isinstance(method_parameters, Mapping) or tuple(method_parameters) != methods:
        raise ValueError("method_parameters must cover the exact ordered methods")
    provenance = raw["selection_provenance"]
    if not isinstance(provenance, Mapping) or not provenance.get("selected_parameters_sha256"):
        raise ValueError("frozen config requires selection provenance")
    lock_path = path.parent / "alzheimer_reviewed_selection_lock.json"
    lock = _read_object(lock_path)
    if provenance.get("reviewed_selection_lock_sha256") != _sha256(lock_path):
        raise ValueError("frozen config lock SHA does not match adjacent reviewed lock")
    if (
        provenance.get("selected_parameters_sha256") != lock.get("selected_parameters_sha256")
        or provenance.get("validation_manifest", {}).get("sha256")
        != lock.get("validation_manifest_sha256")
    ):
        raise ValueError("frozen config artifact SHA does not match reviewed lock")
    winners = provenance.get("winner_candidate_ids", {})
    if winners.get("baseline_eta:zo2_dp_sgd") != "zo2_dp_sgd_eta_m0p25":
        raise ValueError("frozen config does not retain corrected DP-SGD winner")
    if _canonical_payload_sha256(method_parameters, winners) != lock.get(
        "canonical_payload_sha256"
    ):
        raise ValueError("frozen method payload does not match reviewed lock")
    eligible = bool(raw.get("eligible_for_final_plots", role != "package_smoke_only"))
    scientific_contracts = {
        "scientific_base": ((0, 1, 2), ALZHEIMER_PARTITIONS),
        "scientific_local": ((0, 1, 2), ALZHEIMER_PARTITIONS),
        "scientific_server": (tuple(range(30)), ALZHEIMER_PARTITIONS),
    }
    if role in scientific_contracts:
        expected_seeds, expected_partitions = scientific_contracts[role]
        if (
            seeds != expected_seeds
            or partitions != expected_partitions
            or raw["rounds"] != 50
            or not eligible
        ):
            raise ValueError("scientific frozen role does not meet its exact contract")
    for method, values in method_parameters.items():
        if "eta0" not in values or "eta0_source" not in values:
            raise ValueError(f"{method} is missing frozen eta or source")
    spade_values = method_parameters["spade_dfl"]
    if not all(name in spade_values for name in ("rho", "rho_source", "batch_size", "batch_size_source", "mu0", "mu0_source", "beta", "beta_source")):
        raise ValueError("SPADE-DFL frozen parameters or sources are incomplete")
    if role == "package_smoke_only" and (
        eligible or raw["rounds"] != 1 or partitions != ("iid",) or seeds != (0,)
    ):
        raise ValueError("package smoke requires one IID round, seed 0, and no plot eligibility")
    return FrozenAlzheimerSuiteConfig(
        experiment=experiment,
        methods=methods,
        partitions=partitions,
        split_seed=raw["split_seed"],
        protocol_role=role,
        evaluation_scope=str(evaluation_scope),
        output_label=str(raw["output_label"]),
        method_parameters=method_parameters,
        parameter_sources=raw["parameter_sources"],
        selection_provenance=provenance,
        eligible_for_final_plots=eligible,
    )


def _source_for(
    suite: FrozenAlzheimerSuiteConfig, method_values: Mapping[str, Any], name: str
) -> ParameterRecord:
    if name == "n_clients":
        return ParameterRecord(
            name=name,
            value=method_values["optimizer_n_clients"],
            source_type="ProtocolAssumption",
            source="Frozen method optimizer-client semantics",
        )
    raw = method_values.get(f"{name}_source", suite.parameter_sources[name])
    if not isinstance(raw, Mapping):
        raise TypeError(f"frozen source for {name} must be an object")
    return _record(name, raw)


def apply_frozen_method_parameters(
    config: ExperimentConfig,
    *,
    suite: FrozenAlzheimerSuiteConfig,
    method: str,
) -> ExperimentConfig:
    """Restore selected values after generic method configuration mutation."""
    if not suite.method_parameters:
        return config
    method_values = suite.method_parameters[method]
    values = {
        name: method_values[name]
        for name in ("eta0", "rho", "batch_size", "mu0", "clip_norm", "target_epsilon", "target_delta")
        if name in method_values
    }
    if "optimizer_n_clients" in method_values:
        values["n_clients"] = method_values["optimizer_n_clients"]
    sources = dict(config.parameter_sources)
    for name, value in values.items():
        if name == "n_clients":
            sources[name] = _source_for(suite, method_values, "n_clients")
        else:
            sources[name] = _source_for(suite, method_values, name)
        if sources[name].value != value:
            raise ValueError(f"frozen source for {method}.{name} disagrees with value")
    frozen = replace(config, parameter_sources=sources, **values)
    for name, value in values.items():
        if getattr(frozen, name) != value:
            raise ValueError(f"generic method configuration overwrote frozen {method}.{name}")
    if method == "zo2_dp_sgd" and (
        frozen.n_clients != 1 or method_values.get("partition_population_clients") != 31
    ):
        raise ValueError("DP-SGD requires one optimizer over the 31-client partition")
    return frozen


def freeze_alzheimer_suite_configs(
    *, selected_path: Path, manifest_path: Path, tuning_config_path: Path,
    output_dir: Path, reviewed_lock_path: Path | None = None,
) -> dict[str, Path]:
    """Generate all frozen suite JSON through this sole production contract."""
    selected_path, manifest_path = Path(selected_path), Path(manifest_path)
    lock_path = Path(reviewed_lock_path or (
        Path(__file__).resolve().parents[2] / "configs" / "alzheimer_reviewed_selection_lock.json"
    ))
    lock = _read_object(lock_path)
    selected_bytes = selected_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(selected_bytes).hexdigest() != lock.get("selected_parameters_sha256"):
        raise ValueError("reviewed selected-parameters SHA does not match lock")
    if hashlib.sha256(manifest_bytes).hexdigest() != lock.get("validation_manifest_sha256"):
        raise ValueError("reviewed validation-manifest SHA does not match lock")
    selected = json.loads(selected_bytes)
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "completed" or manifest.get("test_set_accessed") is not False:
        raise ValueError("validation manifest is not the completed guarded artifact")
    if manifest.get("firewall_evidence", {}).get("forbidden_access_count") != 0:
        raise ValueError("validation manifest firewall is not zero-access")
    if selected.get("selection_scope") != lock.get("selection_scope"):
        raise ValueError("reviewed selection scope does not match lock")
    tuning = _read_object(Path(tuning_config_path))
    values, winners = selected["selected_values"], _winner_map(selected)
    eta, eta_winner = _selected(values, winners, "spade_eta0", "spade_eta")
    rho, rho_winner = _selected(values, winners, "spade_rho", "spade_rho")
    batch, batch_winner = _selected(values, winners, "spade_batch_size", "spade_batch")
    mu, mu_winner = _selected(values, winners, "spade_mu0", "spade_mu")
    experiment = dict(tuning["experiment"])
    experiment.pop("seeds"); experiment.pop("output_dir")
    experiment.update(eta0=eta, rho=rho, batch_size=batch, mu0=mu, n_rounds=tuning["rounds"])
    sources = {name: _protocol_record(name, value) for name, value in experiment.items()}
    for name, value, winner in (("eta0", eta, eta_winner), ("rho", rho, rho_winner), ("batch_size", batch, batch_winner), ("mu0", mu, mu_winner)):
        sources[name] = _validation_record(value, winner)
    sources["beta"] = {"record_id": "theory:spade_beta_midpoint", "value": experiment["beta"], "source_type": "TheoryDerived", "source": selected["beta"]["source"]}
    method_stages = {
        "zo2_lt_admm_vr": "baseline_eta:zo2_lt_admm_vr", "zo2_lt_admm": "baseline_eta:zo2_lt_admm",
        "zo2_dp_sgd": "baseline_eta:zo2_dp_sgd", "zo2_dp_fedavg": "baseline_eta:zo2_dp_fedavg",
        "one_point_dsg": "baseline_eta:one_point_dsg", "one_point_dsgt": "baseline_eta:one_point_dsgt",
    }
    methods: dict[str, dict[str, Any]] = {}
    for method, stage in method_stages.items():
        key = f"{method}_eta0"; value, winner = _selected(values, winners, key, stage)
        methods[method] = {"eta0": value, "eta0_source": _validation_record(value, winner), "optimizer_n_clients": 31, "partition_population_clients": 31}
    methods["spade_dfl"] = {
        "eta0": eta, "eta0_source": _validation_record(eta, eta_winner), "rho": rho,
        "rho_source": _validation_record(rho, rho_winner), "batch_size": batch,
        "batch_size_source": _validation_record(batch, batch_winner), "mu0": mu,
        "mu0_source": _validation_record(mu, mu_winner), "beta": selected["beta"]["value"],
        "beta_source": {"record_id": "theory:spade_beta_midpoint", "value": selected["beta"]["value"], "source_type": "TheoryDerived", "source": selected["beta"]["source"]},
        "optimizer_n_clients": 31, "partition_population_clients": 31,
    }
    methods["zo2_dp_sgd"].update(optimizer_n_clients=1, optimizer_scope="pooled_31_client_partition", clip_norm=experiment["clip_norm"], target_epsilon=experiment["target_epsilon"], target_delta=experiment["target_delta"])
    methods["zo2_dp_fedavg"].update(optimizer_scope="31_client_federated_aggregation", clip_norm=experiment["clip_norm"], target_epsilon=experiment["target_epsilon"], target_delta=experiment["target_delta"])
    winner_ids = {stage: row["candidate_id"] for stage, row in winners.items()}
    canonical_digest = _canonical_payload_sha256(methods, winner_ids)
    if canonical_digest != lock.get("canonical_payload_sha256"):
        raise ValueError("frozen method payload does not match reviewed lock")
    provenance = {"selected_parameters": selected_path.name, "selected_parameters_sha256": _sha256(selected_path), "winner_candidate_ids": winner_ids, "reviewed_selection_lock_sha256": _sha256(lock_path), "canonical_payload_sha256": canonical_digest, "validation_manifest": {"artifact": manifest_path.name, "sha256": _sha256(manifest_path), "status": manifest["status"], "test_set_accessed": manifest["test_set_accessed"], "forbidden_access_count": 0}}
    base = {"schema_version": 1, "protocol_role": "scientific_base", "evaluation_scope": "outer_final_test_frozen", "output_label": "alzheimer_formal_suite", "methods": list(ALZHEIMER_METHOD_IDS), "partitions": list(APPROVED_TUNING_PROTOCOL.partitions), "seeds": list(APPROVED_TUNING_PROTOCOL.seeds), "rounds": APPROVED_TUNING_PROTOCOL.rounds, "split_seed": tuning["split_seed"], "experiment": experiment, "method_parameters": methods, "parameter_sources": sources, "selection_provenance": provenance}
    local = dict(base); local.update(protocol_role="scientific_local", output_label="alzheimer_local_3seed_50round")
    server = dict(base); server.update(protocol_role="scientific_server", output_label="alzheimer_server_30seed_50round", seeds=list(range(30)))
    smoke_experiment = dict(experiment); smoke_experiment["n_rounds"] = 1
    smoke_sources = dict(sources); smoke_sources["n_rounds"] = _protocol_record("package_smoke_only.rounds", 1)
    smoke = dict(base); smoke.update(protocol_role="package_smoke_only", evaluation_scope="inner_validation_smoke_only", output_label="alzheimer_package_smoke", eligible_for_final_plots=False, partitions=["iid"], seeds=[0], rounds=1, experiment=smoke_experiment, parameter_sources=smoke_sources)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    payloads = {"alzheimer_base.json": base, "alzheimer_local_3seed_50round.json": local, "alzheimer_server_30seed_50round.json": server, "alzheimer_package_smoke.json": smoke}
    for name, payload in payloads.items():
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {name: output / name for name in payloads}
