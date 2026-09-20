"""Immutable source contract for the Alzheimer SPADE-DFL v3 experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .alzheimer_contract import ALZHEIMER_METHOD_IDS
from .config import ExperimentConfig

_SOURCE_KINDS = frozenset(
    {"Reference", "TheoryDerived", "Validation", "ProtocolAssumption"}
)
_DESIGN = (
    "docs/superpowers/specs/"
    "2026-07-26-alzheimer-v3-robust-validation-design.md"
)
_SPADE_VALUES = {
    "eta_multipliers": (0.25, 0.3125, 0.375),
    "mu_values": (3.5, 5.25, 7.0),
    "beta_factors": (1.5, 1.75),
}
_BASELINE_VALUES = {
    "zo2_lt_admm_vr": (0.125, 0.1875, 0.25, 0.375, 0.5),
    "zo2_lt_admm": (0.125, 0.1875, 0.25, 0.375, 0.5),
    "zo2_dp_sgd": (1.0, 1.5, 2.0, 3.0, 4.0),
    "zo2_dp_fedavg": (0.5, 0.75, 1.0, 1.5, 2.0),
    "one_point_dsg": (0.125, 0.1875, 0.25, 0.375, 0.5),
    "one_point_dsgt": (1.0, 1.5, 2.0, 3.0, 4.0),
}


@dataclass(frozen=True)
class V3SourceRecord:
    """One literal value and the evidence that admits it to v3."""

    name: str
    value: object
    source_kind: str
    source_locator: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("v3 source-record name must be nonempty")
        if self.source_kind not in _SOURCE_KINDS:
            raise ValueError(
                f"v3 source_kind must be one of {sorted(_SOURCE_KINDS)}"
            )
        if not self.source_locator:
            raise ValueError("v3 source locator must be nonempty")


@dataclass(frozen=True)
class V3Protocol:
    """Common values frozen before v3 validation."""

    objective_model: str = "lt_admm_nonconvex_logistic"
    regularization: float = 0.01
    sigma_u: float = 0.0
    query_noise_std: float = 0.0
    initialization: str = "shared_zero"
    n_clients: int = 31
    graph_topology: str = "erdos_renyi"
    graph_probability: float = 0.3
    rounds: int = 50
    tau: int = 8
    saga_batch_size: int = 12
    vr_mode: str = "persistent_saga"
    rho: float = 1.0
    validation_seeds: tuple[int, ...] = tuple(range(10))
    allowed_methods: tuple[str, ...] = ALZHEIMER_METHOD_IDS


V3_PROTOCOL = V3Protocol()


def v3_spade_joint_values() -> dict[str, tuple[float, ...]]:
    """Return a fresh copy of the preregistered 18-point SPADE grid."""
    return {name: tuple(values) for name, values in _SPADE_VALUES.items()}


def v3_baseline_multiplier_values() -> dict[str, tuple[float, ...]]:
    """Return the exact equally replicated baseline validation grids."""
    return {method: tuple(values) for method, values in _BASELINE_VALUES.items()}


def _validation_records(
    prefix: str, values: tuple[float, ...]
) -> tuple[V3SourceRecord, ...]:
    return tuple(
        V3SourceRecord(
            name=f"{prefix}:{value}",
            value=value,
            source_kind="Validation",
            source_locator=f"{_DESIGN}, section 5 source-bounded candidate set",
        )
        for value in values
    )


def v3_source_records() -> tuple[V3SourceRecord, ...]:
    """Return all common, reference-center, and validation-grid records."""
    records = [
        V3SourceRecord(
            "objective_model",
            V3_PROTOCOL.objective_model,
            "Reference",
            "Reference [13], Sec. IV nonconvex logistic objective",
        ),
        V3SourceRecord(
            "regularization",
            V3_PROTOCOL.regularization,
            "Reference",
            "Reference [13], Sec. IV sets epsilon=0.01",
        ),
        V3SourceRecord(
            "sigma_u",
            0.0,
            "ProtocolAssumption",
            "SPADE-DFL Assumption 3, zero-variance special case",
        ),
        V3SourceRecord(
            "query_noise_std",
            0.0,
            "ProtocolAssumption",
            "SPADE-DFL Assumption 3, zero-variance special case",
        ),
        V3SourceRecord(
            "initialization",
            "shared_zero",
            "ProtocolAssumption",
            f"{_DESIGN}, section 4 frozen common protocol",
        ),
        V3SourceRecord(
            "n_clients",
            31,
            "ProtocolAssumption",
            f"{_DESIGN}, section 4 retained isolated Alzheimer protocol",
        ),
        V3SourceRecord(
            "graph",
            ("erdos_renyi", 0.3),
            "ProtocolAssumption",
            f"{_DESIGN}, section 4 retained isolated Alzheimer protocol",
        ),
        V3SourceRecord(
            "rounds",
            50,
            "ProtocolAssumption",
            f"{_DESIGN}, section 4 communication budget",
        ),
        V3SourceRecord(
            "tau",
            8,
            "Reference",
            "Reference [13], Fig. 1 reports optimum tau=8",
        ),
        V3SourceRecord(
            "saga_batch_size",
            12,
            "Validation",
            "Alzheimer v2 guarded midpoint validation between batches 8 and 16",
        ),
        V3SourceRecord(
            "validation_seeds",
            tuple(range(10)),
            "Reference",
            "Reference [13], Sec. IV uses ten Monte Carlo repetitions",
        ),
        V3SourceRecord(
            "spade_eta_center",
            1.5 / 39,
            "Reference",
            "Reference [20], Sec. 5.1 eta0=1.5 with SPADE Eq. (10) cz=1/d",
        ),
        V3SourceRecord(
            "spade_mu_center",
            3.5,
            "Reference",
            "Reference [20], Sec. 5.1 mu0=3.5",
        ),
        V3SourceRecord(
            "rho",
            1.0,
            "Reference",
            "Reference [13], Table V penalty center",
        ),
        V3SourceRecord(
            "beta_formula",
            "c_beta/(tau*lambda_u*rho)",
            "TheoryDerived",
            "SPADE-DFL Eq. (31) admissible beta interval",
        ),
    ]
    records.extend(
        _validation_records(
            "spade_eta_multiplier", _SPADE_VALUES["eta_multipliers"]
        )
    )
    records.extend(
        _validation_records("spade_mu", _SPADE_VALUES["mu_values"])
    )
    records.extend(
        _validation_records(
            "spade_beta_factor", _SPADE_VALUES["beta_factors"]
        )
    )
    for method, values in _BASELINE_VALUES.items():
        records.extend(_validation_records(f"{method}_eta_multiplier", values))
    return tuple(records)


def derive_v3_beta(
    factor: float, *, tau: int, lambda_u: float, rho: float
) -> float:
    """Derive beta only for one of the two declared dimensionless factors."""
    if factor not in _SPADE_VALUES["beta_factors"]:
        raise ValueError("factor is not a declared v3 beta factor")
    if tau <= 0 or lambda_u <= 0 or rho <= 0:
        raise ValueError("tau, lambda_u, and rho must be positive")
    return float(factor / (tau * lambda_u * rho))


def validate_v3_common_config(config: ExperimentConfig, method: str) -> None:
    """Fail closed on any drift from the preregistered common protocol."""
    protocol = V3_PROTOCOL
    if method not in protocol.allowed_methods or config.stage != method:
        raise ValueError("v3 requires a canonical method and matching stage")
    expected_clients = 1 if method == "zo2_dp_sgd" else protocol.n_clients
    expected_batches = {
        "spade_dfl": 12,
        "zo2_lt_admm_vr": 1,
        "zo2_lt_admm": 1,
        "zo2_dp_sgd": 31,
        "zo2_dp_fedavg": 1,
        "one_point_dsg": 1,
        "one_point_dsgt": 1,
    }
    expected_vr_mode = (
        "persistent_saga"
        if method == "spade_dfl"
        else "full_refresh"
        if method in {"zo2_lt_admm_vr", "zo2_lt_admm"}
        else config.vr_mode
    )
    checks = (
        (
            config.objective_model == protocol.objective_model,
            "v3 objective is fixed",
        ),
        (
            math.isclose(
                config.regularization,
                protocol.regularization,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "v3 regularization is fixed",
        ),
        (config.sigma_u == protocol.sigma_u, "v3 sigma_u is fixed"),
        (
            config.query_noise_std == protocol.query_noise_std,
            "v3 query_noise_std is fixed",
        ),
        (
            config.initialization == protocol.initialization,
            "v3 initialization is fixed",
        ),
        (
            config.n_clients == expected_clients,
            "v3 requires 31 clients except centralized DP-SGD",
        ),
        (
            config.graph_topology == protocol.graph_topology,
            "v3 graph topology is fixed",
        ),
        (
            math.isclose(
                config.graph_probability,
                protocol.graph_probability,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "v3 graph probability is fixed",
        ),
        (config.n_rounds == protocol.rounds, "v3 requires 50 rounds"),
        (config.tau == protocol.tau, "v3 tau is fixed"),
        (
            config.batch_size == expected_batches[method],
            "v3 method batch is fixed",
        ),
        (
            config.vr_mode == expected_vr_mode,
            "v3 requires persistent SAGA for SPADE",
        ),
        (
            math.isclose(
                config.rho, protocol.rho, rel_tol=0.0, abs_tol=1e-15
            ),
            "v3 rho is fixed",
        ),
    )
    for valid, message in checks:
        if not valid:
            raise ValueError(message)
