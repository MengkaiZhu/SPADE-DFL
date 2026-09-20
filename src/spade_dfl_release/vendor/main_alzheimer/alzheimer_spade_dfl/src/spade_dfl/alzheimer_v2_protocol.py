"""Source-locked common contract for the isolated Alzheimer v2 experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .alzheimer_contract import ALZHEIMER_METHOD_IDS
from .config import ExperimentConfig

SOURCE_KINDS = frozenset({"Reference", "Validation", "ProtocolAssumption"})
VALIDATION_GRID_LOCATOR = (
    "docs/superpowers/specs/2026-07-26-alzheimer-v2-scientific-revision-design.md, "
    "§5 Training-validation selection"
)


@dataclass(frozen=True)
class V2SourceRecord:
    """One immutable source record for a v2 protocol value or candidate."""

    name: str
    value: object
    source_kind: str
    source_locator: str
    condition: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be nonempty")
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError(f"source_kind must be one of {sorted(SOURCE_KINDS)}")
        if not self.source_locator:
            raise ValueError("source_locator must be nonempty")


def _validation_records(
    prefix: str,
    values: tuple[float | int, ...],
) -> tuple[V2SourceRecord, ...]:
    return tuple(
        V2SourceRecord(
            f"{prefix}:{value}",
            value,
            "Validation",
            VALIDATION_GRID_LOCATOR,
        )
        for value in values
    )


COMMON_SOURCE_RECORDS = (
    V2SourceRecord(
        "objective_model",
        "lt_admm_nonconvex_logistic",
        "Reference",
        "Reference [13], Sec. IV nonconvex logistic objective",
    ),
    V2SourceRecord(
        "regularization",
        0.01,
        "Reference",
        "Reference [13], Sec. IV sets epsilon=0.01",
    ),
    V2SourceRecord(
        "sigma_u",
        0.0,
        "ProtocolAssumption",
        "文章/SPADE-DFL.pdf, Assumption 3",
    ),
    V2SourceRecord(
        "query_noise_std",
        0.0,
        "ProtocolAssumption",
        "文章/SPADE-DFL.pdf, Assumption 3",
    ),
    V2SourceRecord(
        "initialization",
        "shared_zero",
        "ProtocolAssumption",
        "docs/superpowers/specs/2026-07-26-alzheimer-v2-scientific-revision-design.md, "
        "§4 Fair v2 protocol",
    ),
    V2SourceRecord(
        "local_training_tau",
        8,
        "Reference",
        "Reference [13], Fig. 1 reports finite optimum at tau=8",
    ),
)

CANDIDATE_SOURCE_RECORDS = (
    V2SourceRecord(
        "spade_eta_center",
        1.5 / 39,
        "Reference",
        "src/spade_dfl/config.py:501 — Reference [20], Sec. 5.1 "
        "eta0=1.5, converted by cz=1/d for SPADE Eq. (10)",
    ),
    *_validation_records("spade_eta_multiplier", (0.25, 0.5, 1.0, 2.0)),
    V2SourceRecord(
        "rho_center",
        1.0,
        "Reference",
        VALIDATION_GRID_LOCATOR + " (Reference-[13] center 1.0)",
    ),
    *_validation_records("rho_candidate", (0.5, 1.0, 2.0)),
    *_validation_records("saga_batch_size", (1, 4, 8, 16)),
    V2SourceRecord(
        "saga_batch_size:12",
        12,
        "Validation",
        VALIDATION_GRID_LOCATOR,
        "Allowed only after 8 and 16 are the two best bracketed validation candidates",
    ),
    V2SourceRecord(
        "spade_mu_center",
        3.5,
        "Reference",
        "src/spade_dfl/config.py:511 — Reference [20], Sec. 5.1",
    ),
    *_validation_records("spade_mu_multiplier", (0.5, 1.0, 2.0)),
    *_validation_records("baseline_eta_multiplier", (0.25, 0.5, 1.0, 2.0)),
    V2SourceRecord(
        "baseline_eta_center:zo2_lt_admm_vr",
        0.1,
        "Reference",
        "Reference [13], Table V includes gamma=0.1",
    ),
    V2SourceRecord(
        "baseline_eta_center:zo2_lt_admm",
        0.1,
        "Reference",
        "Reference [13], Table V includes gamma=0.1",
    ),
    V2SourceRecord(
        "baseline_eta_center:zo2_dp_sgd",
        0.05,
        "Reference",
        "Reference [24] MNIST learning-rate peak",
    ),
    V2SourceRecord(
        "baseline_eta_center:zo2_dp_fedavg",
        0.05,
        "ProtocolAssumption",
        "src/spade_dfl/methods.py, method_config() zo2_dp_fedavg eta0 source record",
    ),
    V2SourceRecord(
        "baseline_eta_center:one_point_dsg",
        0.05,
        "Reference",
        "Reference [21], Sec. IV",
    ),
    V2SourceRecord(
        "baseline_eta_center:one_point_dsgt",
        1.5 / 39,
        "Reference",
        "Reference [20], Sec. 5.1 eta0=1.5, converted by cz=1/d",
    ),
)


@dataclass(frozen=True)
class V2Protocol:
    """Approved common values and their literal scientific provenance."""

    objective_model: str = "lt_admm_nonconvex_logistic"
    regularization: float = 0.01
    sigma_u: float = 0.0
    query_noise_std: float = 0.0
    initialization: str = "shared_zero"
    local_training_tau: int = 8
    objective_source: str = "Reference [13]"
    regularization_source: str = "Reference [13]"
    local_training_tau_source: str = "Reference [13]"
    zero_query_noise_source: str = (
        "SPADE Assumption 3 zero-variance special case"
    )
    initialization_source: str = "Common public protocol assumption"
    batch_12_source: str = (
        "Integer midpoint validation refinement between 8 and 16"
    )
    saga_batch_sizes: tuple[int, ...] = (1, 4, 8, 12, 16)
    local_training_admm_stages: tuple[str, ...] = (
        "spade_dfl",
        "zo2_lt_admm_vr",
        "zo2_lt_admm",
    )
    allowed_loss_only_stages: tuple[str, ...] = ALZHEIMER_METHOD_IDS
    analytic_gradient_stages: tuple[str, ...] = (
        "exact_gradient",
        "lt_admm_vr",
        "lt_admm",
        "dp_sgd",
        "dp_fedavg",
    )
    common_source_records: tuple[V2SourceRecord, ...] = COMMON_SOURCE_RECORDS
    candidate_source_records: tuple[V2SourceRecord, ...] = CANDIDATE_SOURCE_RECORDS


V2_PROTOCOL = V2Protocol()


def validate_v2_common_config(config: ExperimentConfig) -> None:
    """Reject a configuration that departs from the common v2 protocol."""
    protocol = V2_PROTOCOL
    if config.stage in protocol.analytic_gradient_stages:
        raise ValueError("v2 protocol forbids analytic-gradient stages")
    if config.stage not in protocol.allowed_loss_only_stages:
        raise ValueError("v2 protocol requires a canonical loss-only stage")
    if config.objective_model != protocol.objective_model:
        raise ValueError(
            "v2 objective_model must be " f"{protocol.objective_model!r}"
        )
    if not math.isclose(
        config.regularization,
        protocol.regularization,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "v2 regularization must be " f"{protocol.regularization!r}"
        )
    if config.sigma_u != protocol.sigma_u:
        raise ValueError(f"v2 sigma_u must be {protocol.sigma_u!r}")
    if config.query_noise_std != protocol.query_noise_std:
        raise ValueError(
            "v2 query_noise_std must be " f"{protocol.query_noise_std!r}"
        )
    if config.initialization != protocol.initialization:
        raise ValueError(
            "v2 initialization must be " f"{protocol.initialization!r}"
        )
    if (
        config.stage in protocol.local_training_admm_stages
        and config.tau != protocol.local_training_tau
    ):
        raise ValueError(
            "v2 local-training ADMM tau must be "
            f"{protocol.local_training_tau}"
        )


def v2_candidate_values() -> dict[str, tuple[float | int, ...] | str]:
    """Return the approved v2 validation grid and batch-12 provenance."""
    protocol = V2_PROTOCOL
    return {
        "spade_eta_multipliers": (0.25, 0.5, 1.0, 2.0),
        "rho_values": (0.5, 1.0, 2.0),
        "saga_batch_sizes": protocol.saga_batch_sizes,
        "mu_multipliers": (0.5, 1.0, 2.0),
        "baseline_eta_multipliers": (0.25, 0.5, 1.0, 2.0),
        "batch_12_source_type": "Validation",
        "batch_12_source": protocol.batch_12_source,
    }
