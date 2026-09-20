from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from .config import ExperimentConfig
from .data import DatasetBundle
from .graph import build_graph
from .evaluation import new_history, record_metrics
from .initialization import initialize_model_states, initialization_policy
from .objectives import Objective, build_objective
from .partitioning import PartitionResult
from .trainer import RunResult
from .zo import (
    normalized_rademacher,
    spade_one_point_batch_queries,
    spade_one_point_query,
)


def _reference_query(
    objective: Objective,
    theta: np.ndarray,
    x: np.ndarray,
    y: float,
    mu: float,
    rng: np.random.Generator,
    sigma_u: float,
    query_noise_std: float,
) -> np.ndarray:
    u = float(rng.normal(1.0, sigma_u)) if sigma_u > 0 else 1.0

    def oracle(point: np.ndarray) -> float:
        noise = float(rng.normal(0.0, query_noise_std)) if query_noise_std > 0 else 0.0
        return objective.value(point, x[None, :], np.array([y]), u=u) + noise

    return spade_one_point_query(
        oracle,
        theta,
        mu,
        rng,
        baseline=float(getattr(objective, "one_point_baseline", 0.0)),
    ).unscaled_vector


def _batch_query(
    objective: Objective,
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    mu: float,
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> np.ndarray:
    ids = rng.choice(x.shape[0], size=batch_size, replace=True)
    directions = np.empty((batch_size, theta.size), dtype=np.float64)
    multiplicative_noise = np.ones(batch_size, dtype=np.float64)
    additive_noise = np.zeros(batch_size, dtype=np.float64)
    for index in range(batch_size):
        if config.sigma_u > 0:
            multiplicative_noise[index] = float(rng.normal(1.0, config.sigma_u))
        directions[index], _ = normalized_rademacher(rng, theta.size)
        if config.query_noise_std > 0:
            additive_noise[index] = float(rng.normal(0.0, config.query_noise_std))
    scaled = spade_one_point_batch_queries(
        objective,
        theta,
        x[ids],
        y[ids],
        mu,
        directions,
        multiplicative_noise,
        additive_noise,
        baseline=float(getattr(objective, "one_point_baseline", 0.0)),
    )
    return np.mean(scaled, axis=0) * (1.0 / theta.size)


def _record(
    history: Dict[str, List[float]],
    states: np.ndarray,
    trackers: np.ndarray,
    estimates: np.ndarray,
    objective: Objective,
    dataset: DatasetBundle,
) -> None:
    record_metrics(history, states, objective, dataset)
    history["tracking_identity_error"].append(
        float(np.linalg.norm(trackers.mean(axis=0) - estimates.mean(axis=0)))
    )


def run_one_point_dsgt(
    config: ExperimentConfig,
    dataset: DatasetBundle,
    partition: PartitionResult,
    seed: int,
) -> RunResult:
    if config.stage != "one_point_dsgt":
        raise ValueError("one-point DSGT requires stage=one_point_dsgt")
    if len(partition.indices) != config.n_clients:
        raise ValueError("partition client count does not match configuration")
    graph = build_graph(
        config.n_clients,
        topology=config.graph_topology,
        probability=config.graph_probability,
        seed=seed,
    )
    mixing = graph.metropolis_matrix
    seed_sequence = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(child) for child in seed_sequence.spawn(config.n_clients)]
    objective = build_objective(
        config.objective_model,
        config.regularization,
        feature_dimension=config.input_dimension,
        n_classes=config.n_classes,
    )
    local_x = [dataset.train_x[indices] for indices in partition.indices]
    local_y = [dataset.train_y[indices] for indices in partition.indices]
    states = initialize_model_states(config, seed, config.n_clients)
    estimates = np.vstack(
        [
            _batch_query(
                objective,
                states[i],
                local_x[i],
                local_y[i],
                config.batch_size,
                config.mu0,
                rngs[i],
                config,
            )
            for i in range(config.n_clients)
        ]
    )
    trackers = estimates.copy()
    history = new_history("tracking_identity_error")
    _record(history, states, trackers, estimates, objective, dataset)
    start = time.perf_counter()
    for k in range(config.n_rounds):
        # _batch_query returns the unscaled Reference [20] vector z*f:
        # spade_one_point_batch_queries supplies d*z*f and the explicit 1/d
        # above removes that factor.  Any cross-dimension eta conversion is
        # therefore explicit in the frozen experiment configuration.
        eta = config.eta0 * (k + 1.0) ** (-config.eta_exponent)
        states_new = np.clip(
            mixing @ (states - eta * trackers),
            -10.0,
            10.0,
        )
        mu_new = config.mu0 * (k + 2.0) ** (-config.mu_exponent)
        estimates_new = np.vstack(
            [
                _batch_query(
                    objective,
                    states_new[i],
                    local_x[i],
                    local_y[i],
                    config.batch_size,
                    mu_new,
                    rngs[i],
                    config,
                )
                for i in range(config.n_clients)
            ]
        )
        trackers = mixing @ trackers + estimates_new - estimates
        states = states_new
        estimates = estimates_new
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(trackers)):
            raise FloatingPointError(f"non-finite DSGT state at round {k}")
        _record(history, states, trackers, estimates, objective, dataset)
    directed_edges = int(graph.adjacency.sum())
    transmitted_bytes = (
        config.n_rounds * directed_edges * 2 * config.dimension * np.dtype(np.float64).itemsize
    )
    return RunResult(
        states=states,
        history={name: np.asarray(values) for name, values in history.items()},
        query_count=config.n_clients * config.batch_size * (config.n_rounds + 1),
        transmitted_bytes=int(transmitted_bytes),
        wall_time=time.perf_counter() - start,
        metadata={
            "stage": "one_point_dsgt",
            "objective_model": config.objective_model,
            "sigma_u": config.sigma_u,
            "query_noise_std": config.query_noise_std,
            "seed": seed,
            "n_clients": config.n_clients,
            "graph_topology": config.graph_topology,
            "graph_probability": config.graph_probability,
            "partition_mode": partition.mode,
            "dirichlet_alpha": partition.alpha,
            "client_sample_counts": [int(indices.size) for indices in partition.indices],
            "label_histograms": partition.label_histograms.tolist(),
            "vr_mode": "none",
            "beta": 0.0,
            "communication_rounds": config.n_rounds,
            "gradient_evaluations": 0,
            "epsilon": None,
            "delta": 0.0,
            "privacy_scope": "none",
            "privacy_valid": True,
            "one_point_estimator_convention": "unscaled_z_times_f",
            "step_size_dimension_conversion": "external_config_only",
            "initialization": initialization_policy(config),
        },
    )
