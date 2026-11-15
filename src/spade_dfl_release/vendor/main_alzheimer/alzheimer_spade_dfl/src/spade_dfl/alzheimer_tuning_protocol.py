"""Authoritative Alzheimer-local protocol and final run-boundary validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .config import ExperimentConfig
from .data import DatasetBundle
from .partitioning import PartitionResult


@dataclass(frozen=True)
class ApprovedAlzheimerTuningProtocol:
    dimension: int = 39
    n_partition_clients: int = 31
    rounds: int = 50
    split_seed: int = 1
    seeds: tuple[int, ...] = (0, 1, 2)
    partitions: tuple[str, ...] = (
        "dir_0.3",
        "dir_0.6",
        "iid",
        "pathological",
    )
    eta_numerator: float = 1.5
    eta_exponent: float = 0.51
    mu_center: float = 3.5
    mu_exponent: float = 0.17
    regularization: float = 0.1
    sigma_u: float = 0.01
    query_noise_std: float = 1.0
    graph_probability: float = 0.3
    tau: int = 2
    rho_center: float = 1.0
    clip_norm: float = 1.0
    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    eta_multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    rho_values: tuple[float, ...] = (0.5, 1.0, 2.0)
    batch_sizes: tuple[int, ...] = (1, 4, 8, 16)
    mu_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)
    dp_methods: tuple[str, ...] = ("zo2_dp_sgd", "zo2_dp_fedavg")
    centralized_methods: tuple[str, ...] = ("zo2_dp_sgd",)
    baseline_eta_centers: tuple[tuple[str, Optional[float]], ...] = (  # noqa: UP045
        ("zo2_lt_admm_vr", 0.1),
        ("zo2_lt_admm", 0.1),
        ("zo2_dp_sgd", 0.05),
        ("zo2_dp_fedavg", 0.05),
        ("one_point_dsg", 0.05),
        ("one_point_dsgt", None),
    )

    @property
    def eta_center(self) -> float:
        return self.eta_numerator / self.dimension

    def baseline_eta_center(self, method: str) -> float:
        centers = dict(self.baseline_eta_centers)
        try:
            value = centers[method]
        except KeyError as error:
            raise ValueError(f"unknown baseline method {method!r}") from error
        return self.eta_center if value is None else float(value)


APPROVED_TUNING_PROTOCOL = ApprovedAlzheimerTuningProtocol()


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return list(left) == list(right)
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    return left == right


def validate_parameter_source_alignment(config: ExperimentConfig) -> None:
    """Reject any final numeric/config value with stale source metadata."""
    for name, record in config.parameter_sources.items():
        if name == "beta" or not hasattr(config, name):
            continue
        actual = getattr(config, name)
        if not _same_value(record.value, actual):
            raise ValueError(
                f"parameter source {name}={record.value!r} does not match "
                f"final config value {actual!r}"
            )


def validate_final_run_protocol(
    config: ExperimentConfig,
    *,
    method: str,
    bundle: DatasetBundle,
    partition: PartitionResult,
) -> None:
    """Validate the final per-run config at the execution/fingerprint boundary."""
    protocol = APPROVED_TUNING_PROTOCOL
    if config.dataset_name != "alzheimer":
        raise ValueError("final tuning run must use the Alzheimer dataset")
    if config.dimension != protocol.dimension:
        raise ValueError("final tuning run has the wrong model dimension")
    if bundle.train_x.shape[1] != protocol.dimension:
        raise ValueError("final tuning bundle has the wrong feature dimension")
    if len(partition.indices) != protocol.n_partition_clients:
        raise ValueError(
            "final tuning partition must retain "
            f"all {protocol.n_partition_clients} clients"
        )
    assigned = np.concatenate(partition.indices)
    expected = np.arange(bundle.train_x.shape[0], dtype=np.int64)
    if not np.array_equal(np.sort(assigned), expected):
        raise ValueError(
            "final tuning partition must fingerprint the pooled training union"
        )
    expected_optimizer_clients = (
        1
        if method in protocol.centralized_methods
        else protocol.n_partition_clients
    )
    if config.n_clients != expected_optimizer_clients:
        raise ValueError(
            f"{method} requires {expected_optimizer_clients} logical "
            "optimizer client(s)"
        )
    if method in protocol.dp_methods:
        privacy_values = {
            "clip_norm": protocol.clip_norm,
            "target_epsilon": protocol.target_epsilon,
            "target_delta": protocol.target_delta,
        }
        for name, expected_value in privacy_values.items():
            if not _same_value(getattr(config, name), expected_value):
                raise ValueError(
                    f"{method} final {name} must equal {expected_value}"
                )
    validate_parameter_source_alignment(config)


def protocol_candidate_values() -> dict[str, list[Any]]:
    """Return the JSON-declared grid directly from the authoritative protocol."""
    protocol = APPROVED_TUNING_PROTOCOL
    return {
        "spade_eta_multipliers": list(protocol.eta_multipliers),
        "rho_values": list(protocol.rho_values),
        "saga_batch_sizes": list(protocol.batch_sizes),
        "mu_multipliers": list(protocol.mu_multipliers),
        "baseline_eta_multipliers": list(protocol.eta_multipliers),
    }
