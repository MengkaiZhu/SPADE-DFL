"""Fail-closed protocol for the MNIST 6/7 SPADE-DFL paper experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


METHOD_IDS = (
    "zo2_lt_admm_vr",
    "zo2_lt_admm",
    "zo2_dp_sgd",
    "zo2_dp_fedavg",
    "one_point_dsg",
    "one_point_dsgt",
    "spade_dfl_private",
)
PARTITION_IDS = (
    "iid",
    "dir_0.6",
    "dir_0.3",
    "pathological",
    "dir_0.1",
    "pathological_pure",
)
DISPLAY_LABELS = MappingProxyType(
    {
        "zo2_lt_admm_vr": "LT-ADMM-VR",
        "zo2_lt_admm": "LT-ADMM",
        "zo2_dp_sgd": "DP-SGD",
        "zo2_dp_fedavg": "DP-FedAvg",
        "one_point_dsg": "1P-DSG",
        "one_point_dsgt": "1P-DSGT",
        "spade_dfl_private": "SPADE-DFL",
    }
)

_ALLOWED_SOURCE_TYPES = {
    "Reference",
    "SPADE-DFL",
    "TheoryDerived",
    "Validation",
    "Assumed",
}
_PRIVATE_METHODS = {"zo2_dp_sgd", "zo2_dp_fedavg", "spade_dfl_private"}
_EXPECTED_PARAMETER_VALUES = {
    "zo2_lt_admm_vr": {"eta0": 0.1, "batch_size": 1, "tau": 2},
    "zo2_lt_admm": {"eta0": 0.1, "batch_size": 1, "tau": 2},
    "zo2_dp_sgd": {
        "eta0": 0.05,
        "batch_size": 31,
        "clip_norm": 4.0,
        "target_epsilon": 8.0,
        "target_delta": 1e-5,
    },
    "zo2_dp_fedavg": {
        "eta0": 0.001,
        "batch_size": 1,
        "clip_norm": 1.0,
        "local_epochs": 1,
        "target_epsilon": 8.0,
        "target_delta": 1e-5,
    },
    "one_point_dsg": {
        "eta0": 0.05,
        "eta_exponent": 0.75,
        "mu0": 0.8,
        "mu_exponent": 0.25,
    },
    "one_point_dsgt": {
        "eta0": 0.15,
        "eta_exponent": 0.51,
        "mu0": 3.5,
        "mu_exponent": 0.17,
        "sigma_u": 0.01,
    },
    "spade_dfl_private": {
        "eta0": 0.01,
        "eta_exponent": 0.51,
        "mu0": 3.5,
        "mu_exponent": 0.17,
        "rho": 1.0,
        "tau": 2,
        "batch_size": 16,
        "clip_norm": 1.0,
        "clip_exponent": 0.51,
        "vr_mode": "persistent_saga",
        "target_epsilon": 8.0,
        "target_delta": 1e-5,
    },
}


class ProtocolError(ValueError):
    """Raised when an experiment file departs from the frozen protocol."""


@dataclass(frozen=True)
class FrozenProtocol:
    base: Mapping[str, object]
    suite: Mapping[str, object]
    method_parameters: Mapping[str, Mapping[str, object]]
    provenance_sha256: str


def load_and_validate_protocol(
    base_path: Path, suite_path: Path, parameters_path: Path
) -> FrozenProtocol:
    """Load and validate the three files before any official-test execution."""

    base = _read_mapping(base_path, "base configuration")
    suite = _read_mapping(suite_path, "suite configuration")
    frozen = _read_mapping(parameters_path, "frozen parameter artifact")
    _reject_proxyfl(base, "base configuration")
    _reject_proxyfl(suite, "suite configuration")
    _reject_proxyfl(frozen, "frozen parameter artifact")
    _validate_base(base)
    _validate_suite(suite)
    methods = _validate_parameters(frozen)
    digest = hashlib.sha256(Path(parameters_path).read_bytes()).hexdigest()
    return FrozenProtocol(
        base=MappingProxyType(dict(base)),
        suite=MappingProxyType(dict(suite)),
        method_parameters=MappingProxyType(methods),
        provenance_sha256=digest,
    )


def _read_mapping(path: Path, name: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot read {name}: {path}") from error
    if not isinstance(payload, dict):
        raise ProtocolError(f"{name} must be a JSON object")
    return payload


def _validate_base(base: Mapping[str, object]) -> None:
    required = {
        "dataset_name": "mnist",
        "task_type": "binary",
        "negative_class": 6,
        "positive_class": 7,
        "dimension": 10,
        "n_clients": 1000,
        "n_rounds": 50,
        "graph_topology": "ring",
        "initialization": "shared_zero",
        "objective_model": "paper_sigmoid",
        "target_epsilon": 8.0,
        "target_delta": 1e-5,
    }
    for name, expected in required.items():
        if base.get(name) != expected:
            raise ProtocolError(f"base {name} must be {expected!r}")
    if base.get("data_path") != "data/mnist.npz":
        raise ProtocolError("base data_path must be data/mnist.npz")


def _validate_suite(suite: Mapping[str, object]) -> None:
    if tuple(suite.get("methods", ())) != METHOD_IDS:
        raise ProtocolError("suite method order must match the frozen seven-method order")
    if tuple(suite.get("partitions", ())) != PARTITION_IDS:
        raise ProtocolError("suite partition order must match the frozen heterogeneity order")
    seeds = tuple(suite.get("seeds", ()))
    status = suite.get("inference_status")
    if status == "formal_30_paired_seeds":
        if seeds != tuple(range(30)):
            raise ProtocolError("formal seeds must be integers 0 through 29")
        if "n_rounds_override" in suite:
            raise ProtocolError("formal suite cannot override the 50-round horizon")
    elif status == "layout_and_runtime_preview_only":
        if seeds != (0, 1, 2):
            raise ProtocolError("preview seeds must be exactly 0, 1, and 2")
        if suite.get("n_rounds_override") != 3:
            raise ProtocolError("preview n_rounds_override must be exactly 3")
    else:
        raise ProtocolError("suite inference_status is not recognized")
    if suite.get("comparison_environment") != "mnist67_strong_dp_all_black_box":
        raise ProtocolError("suite must declare the MNIST 6/7 strong-DP black-box environment")


def _validate_parameters(
    frozen: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    if frozen.get("status") != "accepted":
        raise ProtocolError("frozen parameter status must be accepted")
    if frozen.get("official_test_used_for_selection") is not False:
        raise ProtocolError("official test data cannot be used for parameter selection")
    raw_methods = frozen.get("method_parameters")
    if not isinstance(raw_methods, dict) or tuple(raw_methods) != METHOD_IDS:
        raise ProtocolError("frozen method order must match the seven-method protocol")
    validated: dict[str, Mapping[str, object]] = {}
    for method in METHOD_IDS:
        record = raw_methods[method]
        if not isinstance(record, dict):
            raise ProtocolError(f"invalid frozen record for {method}")
        parameters = record.get("parameters")
        provenance = record.get("provenance")
        if not isinstance(parameters, dict) or not parameters:
            raise ProtocolError(f"missing parameters for {method}")
        if not isinstance(provenance, dict) or set(provenance) != set(parameters):
            raise ProtocolError(f"missing provenance for {method}")
        for name, expected in _EXPECTED_PARAMETER_VALUES[method].items():
            if parameters.get(name) != expected:
                if method in _PRIVATE_METHODS and name == "target_epsilon":
                    raise ProtocolError(f"{method} must use epsilon=8")
                raise ProtocolError(
                    f"{method}/{name} must equal the frozen value {expected!r}"
                )
        for name, source in provenance.items():
            if not isinstance(source, dict):
                raise ProtocolError(f"invalid provenance for {method}/{name}")
            if source.get("source_type") not in _ALLOWED_SOURCE_TYPES:
                raise ProtocolError(f"invalid source type for {method}/{name}")
            if not isinstance(source.get("source"), str) or not source["source"].strip():
                raise ProtocolError(f"empty source locator for {method}/{name}")
            if "proxyfl" in source["source"].lower():
                raise ProtocolError(f"ProxyFL provenance is forbidden for {method}/{name}")
        validated[method] = MappingProxyType(dict(record))
    return validated


def _reject_proxyfl(payload: object, name: str) -> None:
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    if "proxyfl" in serialized:
        raise ProtocolError(f"ProxyFL provenance is forbidden in {name}")
