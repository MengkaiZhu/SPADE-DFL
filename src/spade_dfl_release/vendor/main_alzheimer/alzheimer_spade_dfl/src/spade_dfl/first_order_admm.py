from __future__ import annotations

import time

import numpy as np

from .config import ExperimentConfig
from .data import DatasetBundle
from .evaluation import new_history, record_metrics
from .graph import (
    admm_penalties,
    build_graph,
    derive_midpoint_beta,
    exchange_messages,
    initialize_auxiliary,
)
from .initialization import initialization_policy, initialize_model_states
from .objectives import Objective, build_objective
from .partitioning import PartitionResult
from .trainer import RunResult
from .two_point import (
    symmetric_two_point_component_batch,
    symmetric_two_point_query,
)


def _component_gradients(
    objective: Objective,
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    return np.vstack(
        [
            objective.gradient(theta, x[index : index + 1], y[index : index + 1], u=1.0)
            for index in range(x.shape[0])
        ]
    )


def _two_point_component(
    objective: Objective,
    theta: np.ndarray,
    x: np.ndarray,
    y: float,
    mu: float,
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> np.ndarray:
    u = float(rng.normal(1.0, config.sigma_u)) if config.sigma_u > 0.0 else 1.0

    def oracle(point: np.ndarray) -> float:
        noise = (
            float(rng.normal(0.0, config.query_noise_std))
            if config.query_noise_std > 0.0
            else 0.0
        )
        return objective.value(point, x[None, :], np.array([y]), u=u) + noise

    return symmetric_two_point_query(oracle, theta, mu, rng).vector


def _two_point_component_batch(
    objective: Objective,
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: float,
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> tuple[np.ndarray, int]:
    if config.sigma_u > 0.0:
        def sample_u() -> float:
            return float(rng.normal(1.0, config.sigma_u))

        u = sample_u
    else:
        u = 1.0
    return symmetric_two_point_component_batch(
        objective,
        theta,
        x,
        y,
        mu,
        rng,
        u,
        config.query_noise_std,
    )


def run_first_order_admm(
    config: ExperimentConfig,
    dataset: DatasetBundle,
    partition: PartitionResult,
    seed: int,
) -> RunResult:
    valid_stages = {
        "lt_admm",
        "lt_admm_vr",
        "zo2_lt_admm",
        "zo2_lt_admm_vr",
    }
    if config.stage not in valid_stages:
        raise ValueError(
            "LT-ADMM requires stage=lt_admm, lt_admm_vr, "
            "zo2_lt_admm, or zo2_lt_admm_vr"
        )
    if len(partition.indices) != config.n_clients:
        raise ValueError("partition client count does not match configuration")
    if dataset.train_x.shape[1] != config.input_dimension:
        raise ValueError("dataset feature dimension does not match configuration")
    if dataset.model_dimension not in {0, config.dimension}:
        raise ValueError("dataset model dimension does not match configuration")

    graph = build_graph(
        config.n_clients,
        topology=config.graph_topology,
        probability=config.graph_probability,
        seed=seed,
    )
    beta = 0.0
    if config.n_clients > 1:
        beta = (
            config.beta
            if config.beta > 0.0
            else derive_midpoint_beta(graph, config.tau, config.rho)
        )
    objective = build_objective(
        config.objective_model,
        config.regularization,
        feature_dimension=config.input_dimension,
        n_classes=config.n_classes,
    )
    children = np.random.SeedSequence(seed).spawn(config.n_clients + 1)
    rngs = [np.random.default_rng(child) for child in children[1:]]
    states = initialize_model_states(config, seed, config.n_clients)
    local_x = [dataset.train_x[indices] for indices in partition.indices]
    local_y = [dataset.train_y[indices] for indices in partition.indices]
    auxiliaries = initialize_auxiliary(states, graph)
    history = new_history()
    penalties = admm_penalties(states, auxiliaries, graph, config.rho)
    record_metrics(history, states, objective, dataset, penalties)
    gradient_evaluations = 0
    directed_edges = int(graph.adjacency.sum())
    transmitted_bytes = 0
    use_two_point = config.stage.startswith("zo2_")
    use_vr = config.stage in {"lt_admm_vr", "zo2_lt_admm_vr"}
    query_count = 0
    start = time.perf_counter()

    for round_index in range(config.n_rounds):
        round_start = states.copy()
        penalties = admm_penalties(round_start, auxiliaries, graph, config.rho)
        local_outputs = np.empty_like(states)
        mu = config.mu0 * (round_index + 1.0) ** (-config.mu_exponent)
        for client in range(config.n_clients):
            rng = rngs[client]
            phi = round_start[client].copy()
            memory: np.ndarray | None = None
            if use_vr:
                if use_two_point:
                    memory, batch_queries = _two_point_component_batch(
                        objective,
                        phi,
                        local_x[client],
                        local_y[client],
                        mu,
                        rng,
                        config,
                    )
                    query_count += batch_queries
                else:
                    memory = _component_gradients(
                        objective, phi, local_x[client], local_y[client]
                    )
                    gradient_evaluations += local_x[client].shape[0]
            for _ in range(config.tau):
                ids = rng.choice(
                    local_x[client].shape[0], size=config.batch_size, replace=True
                )
                if use_vr:
                    assert memory is not None
                    if use_two_point:
                        current, batch_queries = _two_point_component_batch(
                            objective,
                            phi,
                            local_x[client][ids],
                            local_y[client][ids],
                            mu,
                            rng,
                            config,
                        )
                        query_count += batch_queries
                    else:
                        current = np.vstack(
                            [
                                objective.gradient(
                                    phi,
                                    local_x[client][sample : sample + 1],
                                    local_y[client][sample : sample + 1],
                                    u=1.0,
                                )
                                for sample in ids
                            ]
                        )
                        gradient_evaluations += config.batch_size
                    direction = (current - memory[ids]).mean(axis=0) + memory.mean(
                        axis=0
                    )
                    for sample, value in zip(ids, current):
                        memory[int(sample)] = value
                else:
                    if use_two_point:
                        direction = np.mean(
                            [
                                _two_point_component(
                                    objective,
                                    phi,
                                    local_x[client][sample],
                                    float(local_y[client][sample]),
                                    mu,
                                    rng,
                                    config,
                                )
                                for sample in ids
                            ],
                            axis=0,
                        )
                        query_count += 2 * config.batch_size
                    else:
                        direction = objective.gradient(
                            phi,
                            local_x[client][ids],
                            local_y[client][ids],
                            u=1.0,
                        )
                        gradient_evaluations += config.batch_size
                phi -= config.eta0 * direction + beta * penalties[client]
                if not np.all(np.isfinite(phi)):
                    raise FloatingPointError(
                        f"non-finite first-order ADMM state at round {round_index}, "
                        f"client {client}"
                    )
            local_outputs[client] = phi
        states = local_outputs
        if config.n_clients > 1:
            _, auxiliaries = exchange_messages(
                states, auxiliaries, graph, config.rho
            )
            transmitted_bytes += (
                directed_edges * config.dimension * np.dtype(np.float64).itemsize
            )
        penalties_after = admm_penalties(states, auxiliaries, graph, config.rho)
        record_metrics(history, states, objective, dataset, penalties_after)

    return RunResult(
        states=states,
        history={name: np.asarray(values, dtype=np.float64) for name, values in history.items()},
        query_count=int(query_count),
        transmitted_bytes=int(transmitted_bytes),
        wall_time=time.perf_counter() - start,
        metadata={
            "stage": config.stage,
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
            "vr_mode": "full_refresh" if use_vr else "none",
            "beta": beta,
            "communication_rounds": config.n_rounds if config.n_clients > 1 else 0,
            "gradient_evaluations": gradient_evaluations,
            "epsilon": None,
            "delta": 0.0,
            "privacy_scope": "none",
            "privacy_valid": True,
            "gradient_access": not use_two_point,
            "oracle": "two_point" if use_two_point else "analytic_gradient",
            "queries_per_estimate": 2 if use_two_point else 0,
            "query_granularity": "component",
            "direction_distribution": "normalized_rademacher",
            "query_noise_pair_coupling": "independent_additive_noise",
            "stochastic_u_pair_coupling": "shared_within_pair",
            "oracle_transcript_visibility": "trusted_local_internal",
            "training_metrics": "trusted_evaluator_only/outside_DP_claim",
            "initialization": initialization_policy(config),
            "metric_gradient_evaluations": (
                (config.n_rounds + 1) * dataset.train_x.shape[0]
            ),
        },
    )
