from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict

from .config import ExperimentConfig, ParameterRecord
from .data import DatasetBundle
from .dp_baselines import run_dp_fedavg, run_dp_sgd
from .dsg import run_one_point_dsg
from .dsgt import run_one_point_dsgt
from .first_order_admm import run_first_order_admm
from .partitioning import PartitionResult
from .trainer import RunResult, run_experiment


@dataclass(frozen=True)
class MethodSpec:
    label: str
    color: str
    linestyle: str = "-"
    linewidth: float = 1.5
    marker: str = ""


METHOD_REGISTRY: Dict[str, MethodSpec] = {
    "spade_dfl": MethodSpec("SPADE-DFL", "#ef5350", "-", 2.5, "v"),
    "spade_dfl_private": MethodSpec(
        "Private SPADE-DFL", "#ef3b2c", ":", 2.5, "v"
    ),
    "multi_vr_admm": MethodSpec(
        "Non-private SPADE-DFL", "#99000d", "-", 2.4
    ),
    "lt_admm_vr": MethodSpec("LT-ADMM-VR", "#64b5f6"),
    "lt_admm": MethodSpec("LT-ADMM", "#90caf9", "-."),
    "dp_sgd": MethodSpec("DP-SGD", "#81c784"),
    "dp_fedavg": MethodSpec("DP-FedAvg", "#4db6ac"),
    "one_point_dsg": MethodSpec("1P-DSG", "#ffb74d"),
    "one_point_dsgt": MethodSpec("1P-DSGT", "#ff8a65"),
    "zo2_lt_admm_vr": MethodSpec(
        "2P-LT-ADMM-VR", "#3182bd", "-", 1.7, "o"
    ),
    "zo2_lt_admm": MethodSpec(
        "2P-LT-ADMM", "#6baed6", "--", 1.7, "s"
    ),
    "zo2_dp_sgd": MethodSpec(
        "2P-DP-SGD", "#31a354", "-.", 1.7, "^"
    ),
    "zo2_dp_fedavg": MethodSpec(
        "2P-DP-FedAvg", "#74c476", "--", 1.7, "D"
    ),
}


def _record(
    sources: dict[str, ParameterRecord],
    name: str,
    value,
    source_type: str,
    source: str,
) -> None:
    sources[name] = ParameterRecord(name, value, source_type, source)


def method_config(
    base: ExperimentConfig,
    method: str,
    seeds: tuple[int, ...],
    partition_mode: str,
    alpha: float,
) -> ExperimentConfig:
    if method not in METHOD_REGISTRY:
        raise ValueError(f"unknown comparison method {method}")
    values = {
        "stage": method,
        "seeds": tuple(seeds),
        "partition_mode": partition_mode,
        "dirichlet_alpha": alpha,
        "target_epsilon": 0.0,
        "target_delta": 0.0,
        "dp_noise_std": 0.0,
    }
    sources = dict(base.parameter_sources)
    if method.startswith("zo2_"):
        _record(
            sources,
            "stage",
            method,
            "Assumed",
            "Loss-only symmetric two-point black-box adaptation used by the "
            "seven-method CIFAR-10 comparison",
        )
    elif method == "one_point_dsg":
        _record(
            sources,
            "stage",
            method,
            "Reference",
            "Reference [21] one-point distributed stochastic gradient method",
        )
    elif method == "one_point_dsgt":
        _record(
            sources,
            "stage",
            method,
            "Reference",
            "Reference [20] one-point distributed gradient-tracking method",
        )
    elif method in {"spade_dfl", "spade_dfl_private", "multi_vr_admm"}:
        _record(
            sources,
            "stage",
            method,
            "SPADE-DFL",
            "SPADE-DFL Algorithm 1 with the Sec. X-A persistent-SAGA "
            "implementation variant",
        )
    else:
        _record(
            sources,
            "stage",
            method,
            "Reference" if method in {"lt_admm", "dp_sgd", "dp_fedavg"} else "Assumed",
            "Reference-protocol method stage"
            if method in {"lt_admm", "dp_sgd", "dp_fedavg"}
            else "Controlled comparison-method adaptation",
        )
    _record(
        sources,
        "seeds",
        list(seeds),
        "Reference" if len(seeds) == 30 else "Assumed",
        "Reference [20], Sec. 5.1 averages 30 simulation instances"
        if len(seeds) == 30
        else "Reduced local validation repetitions",
    )
    if partition_mode == "dirichlet_balanced" and alpha == 0.1:
        partition_source_type = "Reference"
        partition_source = (
            "Mai, La, and Zhang, IEEE TAI 5(11):5589–5604, 2024, "
            "doi:10.1109/TAI.2024.3430250; supplemental severe non-IID "
            "stress setting"
        )
    elif partition_mode == "dirichlet_balanced" and alpha in {0.3, 0.6}:
        partition_source_type = "Reference"
        partition_source = (
            "Wu et al., “Noise as a Signal: Synergistic Generalization and "
            "Aggregation in Decentralized Federated Learning,” IEEE "
            "Transactions on Mobile Computing manuscript TMC-2025-10-3153; "
            "provided manuscript; simulation uses Dirichlet alpha=0.3 and "
            "alpha=0.6"
        )
    elif partition_mode == "pathological":
        partition_source_type = "Assumed"
        partition_source = (
            "Binary pure-label stress-test adaptation of McMahan et al., "
            "AISTATS 2017"
        )
    else:
        partition_source_type = "Assumed"
        partition_source = "User-required supplemental distribution factor"
    _record(
        sources,
        "partition_mode",
        partition_mode,
        partition_source_type,
        partition_source,
    )
    _record(
        sources,
        "dirichlet_alpha",
        alpha,
        partition_source_type,
        partition_source,
    )
    if base.n_rounds == 500:
        _record(
            sources,
            "n_rounds",
            500,
            "Assumed",
            "Legacy extended horizon; not used by the 50-round reference-consistent profile",
        )

    if method in {"spade_dfl", "spade_dfl_private", "multi_vr_admm"}:
        values.update(
            batch_size=base.batch_size,
            vr_mode="persistent_saga",
            tau=base.tau,
            clip_norm=base.clip_norm,
        )
        _record(
            sources,
            "vr_mode",
            "persistent_saga",
            "SPADE-DFL",
            "Sec. X-A persistent SAGA memory without per-round full refresh",
        )
        _record(
            sources,
            "tau",
            base.tau,
            base.parameter_sources.get(
                "tau", ParameterRecord("tau", base.tau, "Validation", "training-holdout selection")
            ).source_type,
            base.parameter_sources.get(
                "tau", ParameterRecord("tau", base.tau, "Validation", "training-holdout selection")
            ).source,
        )
        _record(
            sources,
            "clip_norm",
            base.clip_norm,
            base.parameter_sources.get(
                "clip_norm",
                ParameterRecord("clip_norm", base.clip_norm, "Validation", "training-holdout selection"),
            ).source_type,
            base.parameter_sources.get(
                "clip_norm",
                ParameterRecord("clip_norm", base.clip_norm, "Validation", "training-holdout selection"),
            ).source,
        )
        if method == "spade_dfl_private":
            clip_exponent = base.clip_exponent
            values.update(
                target_epsilon=base.target_epsilon or 8.0,
                target_delta=base.target_delta or 1e-5,
                clip_exponent=clip_exponent,
            )
            _record(
                sources,
                "clip_exponent",
                clip_exponent,
                "Validation",
                "Preregistered training-holdout clipping-decay grid; SPADE does not give a numerical exponent",
            )
            _record(sources, "target_epsilon", values["target_epsilon"], "Reference", "Reference [24] MNIST reports epsilon=8")
            _record(
                sources,
                "target_delta",
                values["target_delta"],
                "Reference",
                "Reference [24] MNIST reports delta=1e-5",
            )
    elif method in {
        "lt_admm",
        "lt_admm_vr",
        "zo2_lt_admm",
        "zo2_lt_admm_vr",
    }:
        values.update(batch_size=1, eta0=0.1, eta_exponent=0.0, vr_mode="full_refresh")
        _record(sources, "batch_size", 1, "Reference", "Reference [13] Sec. IV")
        _record(
            sources,
            "eta0",
            0.1,
            "Reference" if method == "lt_admm" else "Assumed",
            "Reference [13] Table V for LT-ADMM; transferred to VR and 2P black-box adaptations as an explicit candidate",
        )
        _record(
            sources,
            "eta_exponent",
            0.0,
            "Reference" if method == "lt_admm" else "Assumed",
            "Reference [13] constant LT-ADMM step"
            if method == "lt_admm"
            else "Controlled transfer of the Reference [13] constant-step "
            "schedule to the VR/two-point comparison adaptation",
        )
        _record(
            sources,
            "vr_mode",
            "full_refresh",
            "Assumed",
            "Per-round full-refresh memory for the LT-ADMM-VR comparison; "
            "an inactive compatibility field for LT-ADMM without VR",
        )
    elif method in {"dp_sgd", "zo2_dp_sgd"}:
        values.update(
            n_clients=1,
            batch_size=31,
            eta0=0.05,
            eta_exponent=0.0,
            clip_norm=4.0,
            target_epsilon=8.0,
            target_delta=1e-5,
        )
        _record(
            sources,
            "stage",
            method,
            "Assumed" if method.startswith("zo2_") else "Reference",
            (
                "Loss-only two-point adaptation of the centralized DP-SGD "
                "protocol in Reference [24]"
                if method.startswith("zo2_")
                else "Reference [24] centralized DP-SGD training protocol"
            ),
        )
        _record(
            sources,
            "n_clients",
            1,
            "Assumed",
            "centralized DP-SGD protocol uses one logical optimizer and no client communication; exception to the 31-client decentralized suite",
        )
        _record(
            sources,
            "eta0",
            0.05,
            "Assumed" if method.startswith("zo2_") else "Reference",
            "Reference [24] MNIST learning-rate peak; transferred to the 2P black-box adaptation"
            if method.startswith("zo2_")
            else "Reference [24] MNIST learning-rate peak",
        )
        _record(
            sources,
            "eta_exponent",
            0.0,
            "Assumed" if method.startswith("zo2_") else "Reference",
            "Controlled transfer of the fixed Reference [24] learning-rate "
            "schedule to the two-point black-box adaptation"
            if method.startswith("zo2_")
            else "Reference [24] fixed learning-rate schedule",
        )
        _record(sources, "clip_norm", 4.0, "Reference", "Reference [24] MNIST reference clipping bound")
        _record(sources, "batch_size", 31, "Assumed", "Common low-dimensional local compute budget")
        _record(sources, "target_epsilon", 8.0, "Reference", "Reference [47] comparison budget")
        _record(sources, "target_delta", 1e-5, "Reference", "Reference [24] MNIST")
    elif method in {"dp_fedavg", "zo2_dp_fedavg"}:
        values.update(
            batch_size=1,
            local_epochs=1,
            eta0=0.05,
            eta_exponent=0.0,
            clip_norm=base.clip_norm,
            target_epsilon=base.target_epsilon or 8.0,
            target_delta=base.target_delta or 1e-5,
        )
        _record(sources, "target_epsilon", values["target_epsilon"], "Reference", "References [24] and [47] use epsilon=8")
        _record(sources, "target_delta", values["target_delta"], "Reference", "Reference [24] MNIST uses delta=1e-5")
        _record(
            sources,
            "clip_norm",
            base.clip_norm,
            "Assumed",
            "Controlled positive client-update clipping value inherited from "
            "the common base; frozen validation is SPADE-only and does not "
            "validate DP-FedAvg",
        )
        _record(
            sources,
            "local_epochs",
            1,
            "Assumed",
            "One local epoch is the smallest FedAvg communication-round implementation allowed by Reference [47] Algorithm 1",
        )
        _record(
            sources,
            "batch_size",
            1,
            "Assumed",
            "Single-sample symmetric two-point update for the common black-box oracle setting",
        )
        _record(
            sources,
            "eta0",
            0.05,
            "Assumed",
            "Controlled transfer of the Reference [24] MNIST learning-rate "
            "scale to the two-point DP-FedAvg adaptation; no DP-FedAvg "
            "CIFAR-10 validation selection is claimed",
        )
        _record(
            sources,
            "eta_exponent",
            0.0,
            "Assumed",
            "Reference [47] Algorithm 1 uses eta without a reported decay schedule",
        )
    elif method == "one_point_dsg":
        values.update(
            batch_size=1,
            eta0=0.05,
            eta_exponent=0.75,
            mu0=0.8,
            mu_exponent=0.25,
            sigma_u=0.0,
            query_noise_std=1.0,
        )
        _record(sources, "batch_size", 1, "Reference", "Reference [21] one query per client and iteration")
        _record(sources, "eta0", 0.05, "Reference", "Reference [21] Sec. IV")
        _record(sources, "eta_exponent", 0.75, "Reference", "Reference [21] Sec. IV")
        _record(sources, "mu0", 0.8, "Reference", "Reference [21] Sec. IV")
        _record(sources, "mu_exponent", 0.25, "Reference", "Reference [21] Sec. IV")
        _record(
            sources,
            "sigma_u",
            0.0,
            "Assumed",
            "Reference [21] comparison uses additive query noise rather than "
            "the Reference [20] multiplicative u factor; that factor is disabled",
        )
        _record(
            sources,
            "query_noise_std",
            1.0,
            "Reference",
            "Reference [21] MNIST experiment uses standard Gaussian query noise",
        )
    elif method == "one_point_dsgt":
        values.update(
            batch_size=1,
            eta0=1.5 / base.dimension,
            eta_exponent=0.51,
            mu0=3.5,
            mu_exponent=0.17,
            sigma_u=0.01,
            query_noise_std=1.0,
        )
        _record(sources, "batch_size", 1, "Reference", "Reference [20] one query per client and iteration")
        _record(sources, "eta0", 1.5 / base.dimension, "Reference", "Reference [20] eta0 converted by cz=1/d")
        _record(sources, "eta_exponent", 0.51, "Reference", "Reference [20] Sec. 5.1")
        _record(sources, "mu0", 3.5, "Reference", "Reference [20] Sec. 5.1")
        _record(sources, "mu_exponent", 0.17, "Reference", "Reference [20] Sec. 5.1")
        _record(sources, "sigma_u", 0.01, "Reference", "Reference [20] Sec. 5.1")
        _record(
            sources,
            "query_noise_std",
            1.0,
            "Reference",
            "Reference [20] Sec. 5.1",
        )
    if method.startswith("zo2_"):
        _record(
            sources,
            "gradient_oracle",
            "symmetric_two_point",
            "Assumed",
            "Loss-only black-box adaptation requested for this comparison; normalized Rademacher central difference",
        )
        _record(
            sources,
            "mu_schedule",
            f"{base.mu0}*(k+1)^(-{base.mu_exponent})",
            "Assumed",
            "Reference [20] smoothing schedule transferred to the 2P black-box adaptation",
        )
    return replace(base, parameter_sources=sources, **values)


def run_method(
    method: str,
    config: ExperimentConfig,
    dataset: DatasetBundle,
    partition: PartitionResult,
    seed: int,
) -> RunResult:
    if method not in METHOD_REGISTRY:
        raise ValueError(f"unknown comparison method {method}")
    if method in {"spade_dfl", "spade_dfl_private", "multi_vr_admm"}:
        result = run_experiment(config, dataset, seed, partition=partition)
    elif method in {
        "lt_admm",
        "lt_admm_vr",
        "zo2_lt_admm",
        "zo2_lt_admm_vr",
    }:
        result = run_first_order_admm(config, dataset, partition, seed)
    elif method in {"dp_sgd", "zo2_dp_sgd"}:
        result = run_dp_sgd(config, dataset, seed)
    elif method in {"dp_fedavg", "zo2_dp_fedavg"}:
        result = run_dp_fedavg(config, dataset, partition, seed)
    elif method == "one_point_dsg":
        result = run_one_point_dsg(config, dataset, partition, seed)
    else:
        result = run_one_point_dsgt(config, dataset, partition, seed)

    if method in {"dp_sgd", "zo2_dp_sgd"}:
        outer_round_semantics = "centralized_optimization_step_no_communication"
    elif method in {"dp_fedavg", "zo2_dp_fedavg"}:
        outer_round_semantics = "client_compute_then_server_aggregate"
    else:
        outer_round_semantics = "decentralized_neighbor_model_exchange"
    if method.startswith("zo2_"):
        gradient_oracle = "symmetric_two_point"
    elif method in {
        "spade_dfl_private",
        "spade_dfl",
        "multi_vr_admm",
        "one_point_dsg",
        "one_point_dsgt",
    }:
        gradient_oracle = "one_point"
    else:
        gradient_oracle = "analytic_gradient"
    metadata = dict(result.metadata)
    metadata.update(
        {
            "outer_round_semantics": outer_round_semantics,
            "outer_round_count": config.n_rounds,
            "gradient_oracle": gradient_oracle,
            "gradient_access": gradient_oracle == "analytic_gradient",
            "transmitted_bytes": result.transmitted_bytes,
            "x_axis_caveat": (
                "optimization-step index; no network communication"
                if method in {"dp_sgd", "zo2_dp_sgd"}
                else "one communication/aggregation event per outer round"
            ),
        }
    )
    return replace(result, metadata=metadata)
