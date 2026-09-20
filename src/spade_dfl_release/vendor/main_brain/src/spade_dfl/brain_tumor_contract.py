from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from .config import ExperimentConfig, ParameterRecord
from .isolation import BRAIN_TUMOR_METHOD_IDS
from .methods import method_config
from .partitioning import PartitionResult, partition_labels


BRAIN_TUMOR_LEGEND_ORDER = (
    "LT-ADMM-VR",
    "LT-ADMM",
    "DP-SGD",
    "DP-FedAvg",
    "1P-DSG",
    "1P-DSGT",
    "SPADE-DFL",
)


def _record(name: str, value, source_type: str, source: str) -> ParameterRecord:
    return ParameterRecord(name, value, source_type, source)


def brain_tumor_base_config(
    *,
    n_rounds: int = 20,
    seeds: Sequence[int] = (0, 1, 2),
    n_clients: int = 31,
    eta0: float = 1.5 / 40.0,
    rho: float = 1.0,
    batch_size: int = 1,
) -> ExperimentConfig:
    sources = {
        "dataset_name": _record(
            "dataset_name",
            "brain_tumor_mri",
            "Assumed",
            "User-supplied Brain Tumor MRI Dataset with official Training/Testing folders",
        ),
        "feature_dimension": _record(
            "feature_dimension",
            10,
            "Reference",
            "Reference [20] uses a 10-dimensional lossy representation",
        ),
        "dimension": _record(
            "dimension",
            40,
            "TheoryDerived",
            "10 PCA features times four softmax classes",
        ),
        "n_clients": _record(
            "n_clients",
            n_clients,
            "Reference" if n_clients == 31 else "Assumed",
            "Reference [20] uses 31 clients; reduced values are test-only",
        ),
        "graph_probability": _record(
            "graph_probability",
            0.3,
            "Reference",
            "Reference [20] connected Erdos-Renyi probability",
        ),
        "eta0": _record(
            "eta0",
            eta0,
            "Validation",
            "Training-holdout grid centered on Reference [20] 1.5/d with d=40",
        ),
        "eta_exponent": _record(
            "eta_exponent", 0.51, "Reference", "Reference [20], Sec. 5.1"
        ),
        "mu0": _record("mu0", 3.5, "Reference", "Reference [20], Sec. 5.1"),
        "mu_exponent": _record(
            "mu_exponent", 0.17, "Reference", "Reference [20], Sec. 5.1"
        ),
        "regularization": _record(
            "regularization", 0.1, "Reference", "Reference [20] experimental objective"
        ),
        "rho": _record(
            "rho",
            rho,
            "Validation",
            "Training-only holdout grid because SPADE-DFL gives only rho>0",
        ),
        "tau": _record("tau", 2, "Reference", "Reference [13] main comparison"),
        "batch_size": _record(
            "batch_size",
            batch_size,
            "Reference" if batch_size == 1 else "Validation",
            "Reference [13] batch=1; larger values require recorded training-only validation",
        ),
        "initialization": _record(
            "initialization",
            "shared_zero",
            "Assumed",
            "Common public zero initialization applied identically to all methods",
        ),
        "n_rounds": _record(
            "n_rounds",
            n_rounds,
            "Assumed",
            "20-round local validation or frozen 50-round server horizon",
        ),
        "seeds": _record(
            "seeds",
            list(seeds),
            "Reference" if len(tuple(seeds)) == 30 else "Assumed",
            "Reference [20] uses 30 instances; three seeds are local validation only",
        ),
    }
    return ExperimentConfig(
        stage="spade_dfl",
        objective_model="multiclass_softmax",
        dataset_name="brain_tumor_mri",
        task_type="multiclass",
        feature_dimension=10,
        n_classes=4,
        dimension=40,
        n_rounds=n_rounds,
        tau=2,
        batch_size=batch_size,
        n_clients=n_clients,
        seeds=tuple(seeds),
        eta0=eta0,
        eta_exponent=0.51,
        mu0=3.5,
        mu_exponent=0.17,
        regularization=0.1,
        sigma_u=0.01,
        query_noise_std=1.0,
        rho=rho,
        beta=0.0,
        clip_norm=4.0,
        target_epsilon=0.0,
        target_delta=0.0,
        initialization="shared_zero",
        graph_topology="erdos_renyi",
        graph_probability=0.3,
        partition_mode="iid",
        dirichlet_alpha=0.3,
        vr_mode="persistent_saga",
        data_path="../1、Brain Tumor MRI Dataset.zip",
        output_dir="results",
        download_url="",
        parameter_sources=sources,
    )


def validate_brain_tumor_method_set(methods: Sequence[str]) -> None:
    if tuple(methods) != BRAIN_TUMOR_METHOD_IDS:
        raise ValueError(
            "Brain Tumor suite must use the exact ordered seven-method contract"
        )


def partition_brain_tumor(
    labels: np.ndarray,
    n_clients: int,
    panel: str,
    *,
    seed: int,
) -> PartitionResult:
    mapping = {
        "dir_0.3": ("dirichlet", 0.3),
        "dir_0.6": ("dirichlet", 0.6),
        "iid": ("iid", 0.3),
        "pathological": ("pathological_shards", 0.3),
    }
    if panel not in mapping:
        raise ValueError(f"unknown Brain Tumor panel: {panel}")
    mode, alpha = mapping[panel]
    return partition_labels(
        np.asarray(labels),
        n_clients,
        mode,
        np.random.default_rng(seed),
        alpha=alpha,
    )


def brain_tumor_method_config(
    base: ExperimentConfig,
    method: str,
    seeds: tuple[int, ...],
    partition_mode: str,
    alpha: float,
) -> ExperimentConfig:
    validate_brain_tumor_method_set(BRAIN_TUMOR_METHOD_IDS)
    if method not in BRAIN_TUMOR_METHOD_IDS:
        raise ValueError(f"method is outside Brain Tumor seven-method contract: {method}")
    configured = method_config(base, method, seeds, partition_mode, alpha)
    if method == "spade_dfl":
        configured = replace(
            configured,
            stage="spade_dfl",
            vr_mode="persistent_saga",
            target_epsilon=0.0,
            target_delta=0.0,
            dp_noise_std=0.0,
        )
    return configured
