from __future__ import annotations

import time

import numpy as np

from .config import ExperimentConfig
from .data import DatasetBundle
from .evaluation import new_history, record_metrics
from .graph import build_graph
from .initialization import initialize_model_states, initialization_policy
from .objectives import Objective, build_objective
from .partitioning import PartitionResult
from .trainer import RunResult
from .zo import spade_one_point_query


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
    estimates = []
    for sample in ids:
        u = float(rng.normal(1.0, config.sigma_u)) if config.sigma_u > 0 else 1.0

        def oracle(point: np.ndarray) -> float:
            noise = (
                float(rng.normal(0.0, config.query_noise_std))
                if config.query_noise_std > 0
                else 0.0
            )
            return (
                objective.value(
                    point,
                    x[int(sample) : int(sample) + 1],
                    y[int(sample) : int(sample) + 1],
                    u=u,
                )
                + noise
            )

        estimates.append(
            spade_one_point_query(oracle, theta, radius, rng).unscaled_vector
        )
    return np.mean(estimates, axis=0)


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
    objective = build_objective(config.objective_model, config.regularization)
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
            "initialization": initialization_policy(config),
        },
    )
