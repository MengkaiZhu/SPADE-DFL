from __future__ import annotations

import time

import numpy as np

from .config import ExperimentConfig
from .data import DatasetBundle
from .evaluation import new_history, record_metrics
from .graph import build_graph
from .initialization import initialization_policy, initialize_model_states
from .objectives import Objective, build_objective
from .partitioning import PartitionResult
from .trainer import RunResult
from .zo import (
    normalized_rademacher,
    spade_one_point_batch_queries,
)


def _one_point_estimate(
    objective: Objective,
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    radius: float,
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> np.ndarray:
    ids = rng.choice(x.shape[0], size=batch_size, replace=True)
    directions = np.empty((batch_size, theta.size), dtype=np.float64)
    multiplicative_noise = np.ones(batch_size, dtype=np.float64)
    additive_noise = np.zeros(batch_size, dtype=np.float64)
    # Preserve the scalar implementation's exact per-sample RNG order:
    # multiplicative noise, direction, then additive query noise.
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
        radius,
        directions,
        multiplicative_noise,
        additive_noise,
        baseline=float(getattr(objective, "one_point_baseline", 0.0)),
    )
    return np.mean(scaled, axis=0) * (1.0 / theta.size)


def run_one_point_dsg(
    config: ExperimentConfig,
    dataset: DatasetBundle,
    partition: PartitionResult,
    seed: int,
) -> RunResult:
    if config.stage != "one_point_dsg":
        raise ValueError("one-point DSG requires stage=one_point_dsg")
    if len(partition.indices) != config.n_clients:
        raise ValueError("partition client count does not match configuration")
    graph = build_graph(
        config.n_clients,
        topology=config.graph_topology,
        probability=config.graph_probability,
        seed=seed,
    )
    mixing = graph.metropolis_matrix
    rngs = [
        np.random.default_rng(child)
        for child in np.random.SeedSequence(seed).spawn(config.n_clients)
    ]
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
            _one_point_estimate(
                objective,
                states[client],
                local_x[client],
                local_y[client],
                config.batch_size,
                config.mu0,
                rngs[client],
                config,
            )
            for client in range(config.n_clients)
        ]
    )
    history = new_history()
    record_metrics(history, states, objective, dataset)
    start = time.perf_counter()
    for round_index in range(config.n_rounds):
        alpha = config.eta0 * (round_index + 1.0) ** (-config.eta_exponent)
        states = np.clip(mixing @ (states - alpha * estimates), -10.0, 10.0)
        radius = config.mu0 * (round_index + 2.0) ** (-config.mu_exponent)
        estimates = np.vstack(
            [
                _one_point_estimate(
                    objective,
                    states[client],
                    local_x[client],
                    local_y[client],
                    config.batch_size,
                    radius,
                    rngs[client],
                    config,
                )
                for client in range(config.n_clients)
            ]
        )
        if not np.all(np.isfinite(states)):
            raise FloatingPointError(f"non-finite DSG state at round {round_index}")
        record_metrics(history, states, objective, dataset)
    directed_edges = int(graph.adjacency.sum())
    transmitted_bytes = (
        config.n_rounds
        * directed_edges
        * config.dimension
        * np.dtype(np.float64).itemsize
    )
    return RunResult(
        states=states,
        history={name: np.asarray(values, dtype=np.float64) for name, values in history.items()},
        query_count=config.n_clients * config.batch_size * (config.n_rounds + 1),
        transmitted_bytes=int(transmitted_bytes),
        wall_time=time.perf_counter() - start,
        metadata={
            "stage": "one_point_dsg",
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
