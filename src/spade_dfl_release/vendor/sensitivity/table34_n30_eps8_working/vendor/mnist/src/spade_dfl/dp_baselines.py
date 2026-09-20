from __future__ import annotations

import time

import numpy as np

from .config import ExperimentConfig
from .data import DatasetBundle
from .evaluation import new_history, record_metrics
from .initialization import initialize_model_states, initialization_policy
from .objectives import build_objective
from .partitioning import PartitionResult
from .privacy import (
    clip_update,
    gaussian_composition_epsilon,
    noise_std_for_gaussian_budget,
)
from .trainer import RunResult
from .two_point import symmetric_two_point_query


def _privacy_parameters(
    config: ExperimentConfig, sensitivity: float
) -> tuple[float, float, float, float]:
    if config.target_epsilon <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    delta = config.target_delta if config.target_delta > 0.0 else 1e-5
    noise_std = noise_std_for_gaussian_budget(
        config.n_rounds,
        sensitivity,
        delta,
        config.target_epsilon,
    )
    epsilon, rho = gaussian_composition_epsilon(
        config.n_rounds, sensitivity, noise_std, delta
    )
    return noise_std, epsilon, delta, rho


def _two_point_components(
    objective,
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: float,
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> np.ndarray:
    estimates = []
    for sample in range(x.shape[0]):
        u = (
            float(rng.normal(1.0, config.sigma_u))
            if config.sigma_u > 0.0
            else 1.0
        )

        def oracle(point: np.ndarray) -> float:
            noise = (
                float(rng.normal(0.0, config.query_noise_std))
                if config.query_noise_std > 0.0
                else 0.0
            )
            return (
                objective.value(
                    point,
                    x[sample : sample + 1],
                    y[sample : sample + 1],
                    u=u,
                )
                + noise
            )

        estimates.append(
            symmetric_two_point_query(oracle, theta, mu, rng).vector
        )
    return np.vstack(estimates)


def run_dp_sgd(
    config: ExperimentConfig,
    dataset: DatasetBundle,
    seed: int,
) -> RunResult:
    if config.stage not in {"dp_sgd", "zo2_dp_sgd"}:
        raise ValueError("DP-SGD requires stage=dp_sgd or zo2_dp_sgd")
    rng = np.random.default_rng(seed)
    objective = build_objective(config.objective_model, config.regularization)
    state = initialize_model_states(config, seed, 1)[0]
    if config.batch_size > dataset.train_x.shape[0]:
        raise ValueError("DP-SGD batch_size cannot exceed the training set")
    use_two_point = config.stage == "zo2_dp_sgd"
    sensitivity = 2.0 * config.clip_norm / config.batch_size
    noise_std, epsilon, delta, rho = _privacy_parameters(config, sensitivity)
    history = new_history()
    record_metrics(history, state[None, :], objective, dataset)
    gradient_evaluations = 0
    query_count = 0
    start = time.perf_counter()
    for round_index in range(config.n_rounds):
        ids = rng.choice(
            dataset.train_x.shape[0], size=config.batch_size, replace=False
        )
        if use_two_point:
            mu = config.mu0 * (round_index + 1.0) ** (-config.mu_exponent)
            components = _two_point_components(
                objective,
                state,
                dataset.train_x[ids],
                dataset.train_y[ids],
                mu,
                rng,
                config,
            )
            query_count += 2 * config.batch_size
        else:
            components = objective.component_gradients(
                state, dataset.train_x[ids], dataset.train_y[ids], u=1.0
            )
            gradient_evaluations += config.batch_size
        norms = np.linalg.norm(components, axis=1)
        scales = np.minimum(1.0, config.clip_norm / np.maximum(norms, 1e-30))
        clipped_mean = np.mean(components * scales[:, None], axis=0)
        noise = (
            rng.normal(0.0, noise_std, size=config.dimension)
            if noise_std > 0.0
            else np.zeros(config.dimension)
        )
        eta = config.eta0 * (round_index + 1.0) ** (-config.eta_exponent)
        state -= eta * (clipped_mean + noise)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(f"non-finite DP-SGD state at round {round_index}")
        record_metrics(history, state[None, :], objective, dataset)
    return RunResult(
        states=state[None, :],
        history={name: np.asarray(values, dtype=np.float64) for name, values in history.items()},
        query_count=int(query_count),
        transmitted_bytes=0,
        wall_time=time.perf_counter() - start,
        metadata={
            "stage": config.stage,
            "objective_model": config.objective_model,
            "sigma_u": config.sigma_u,
            "query_noise_std": config.query_noise_std,
            "seed": seed,
            "n_clients": 1,
            "communication_rounds": 0,
            "gradient_evaluations": gradient_evaluations,
            "epsilon": epsilon if epsilon > 0.0 else None,
            "delta": delta,
            "rho_zcdp": rho,
            "privacy_scope": "example_replace",
            "privacy_valid": bool(
                config.target_epsilon <= 0.0
                or np.isclose(epsilon, config.target_epsilon)
                and np.isclose(delta, config.target_delta)
            ),
            "sensitivity": sensitivity,
            "noise_std": noise_std,
            "noise_multiplier": noise_std / sensitivity if sensitivity > 0 else 0.0,
            "clip_norm": config.clip_norm,
            "vr_mode": "none",
            "beta": 0.0,
            "gradient_access": not use_two_point,
            "oracle": "two_point" if use_two_point else "analytic_gradient",
            "queries_per_estimate": 2 if use_two_point else 0,
            "query_granularity": "component",
            "sampling_scheme": "without_replacement",
            "privacy_adjacency": "training datasets differ in one example",
            "privacy_amplification_used": False,
            "oracle_transcript_visibility": "trusted_local_internal",
            "training_metrics": "trusted_evaluator_only/outside_DP_claim",
            "initialization": initialization_policy(config),
            "metric_gradient_evaluations": (
                (config.n_rounds + 1) * dataset.train_x.shape[0]
            ),
        },
    )


def run_dp_fedavg(
    config: ExperimentConfig,
    dataset: DatasetBundle,
    partition: PartitionResult,
    seed: int,
) -> RunResult:
    if config.stage not in {"dp_fedavg", "zo2_dp_fedavg"}:
        raise ValueError(
            "DP-FedAvg requires stage=dp_fedavg or zo2_dp_fedavg"
        )
    if len(partition.indices) != config.n_clients:
        raise ValueError("partition client count does not match configuration")
    objective = build_objective(config.objective_model, config.regularization)
    children = np.random.SeedSequence(seed).spawn(config.n_clients + 1)
    server_rng = np.random.default_rng(children[0])
    client_rngs = [np.random.default_rng(child) for child in children[1:]]
    state = initialize_model_states(config, seed, 1)[0]
    use_two_point = config.stage == "zo2_dp_fedavg"
    sensitivity = 2.0 * config.clip_norm
    noise_std_sum, epsilon, delta, rho = _privacy_parameters(config, sensitivity)
    history = new_history()
    consensus_states = np.tile(state, (config.n_clients, 1))
    record_metrics(history, consensus_states, objective, dataset)
    gradient_evaluations = 0
    query_count = 0
    start = time.perf_counter()
    for round_index in range(config.n_rounds):
        updates = []
        for client, indices in enumerate(partition.indices):
            local_state = state.copy()
            for _ in range(config.local_epochs):
                order = client_rngs[client].permutation(indices.size)
                for offset in range(0, indices.size, config.batch_size):
                    local_ids = indices[order[offset : offset + config.batch_size]]
                    if use_two_point:
                        mu = config.mu0 * (round_index + 1.0) ** (
                            -config.mu_exponent
                        )
                        direction = np.mean(
                            _two_point_components(
                                objective,
                                local_state,
                                dataset.train_x[local_ids],
                                dataset.train_y[local_ids],
                                mu,
                                client_rngs[client],
                                config,
                            ),
                            axis=0,
                        )
                        query_count += 2 * local_ids.size
                    else:
                        direction = objective.gradient(
                            local_state,
                            dataset.train_x[local_ids],
                            dataset.train_y[local_ids],
                            u=1.0,
                        )
                        gradient_evaluations += local_ids.size
                    local_state -= config.eta0 * direction
            updates.append(clip_update(local_state - state, config.clip_norm))
        summed = np.sum(updates, axis=0)
        if noise_std_sum > 0.0:
            summed += server_rng.normal(
                0.0, noise_std_sum, size=config.dimension
            )
        state = state + summed / config.n_clients
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(
                f"non-finite DP-FedAvg state at round {round_index}"
            )
        consensus_states = np.tile(state, (config.n_clients, 1))
        record_metrics(history, consensus_states, objective, dataset)
    transmitted_bytes = (
        config.n_rounds
        * config.n_clients
        * 2
        * config.dimension
        * np.dtype(np.float64).itemsize
    )
    return RunResult(
        states=consensus_states,
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
            "partition_mode": partition.mode,
            "dirichlet_alpha": partition.alpha,
            "client_sample_counts": [int(indices.size) for indices in partition.indices],
            "label_histograms": partition.label_histograms.tolist(),
            "communication_rounds": config.n_rounds,
            "gradient_evaluations": gradient_evaluations,
            "epsilon": epsilon if epsilon > 0.0 else None,
            "delta": delta,
            "rho_zcdp": rho,
            "privacy_scope": "client_dataset_replace",
            "privacy_valid": bool(
                config.target_epsilon <= 0.0
                or np.isclose(epsilon, config.target_epsilon)
                and np.isclose(delta, config.target_delta)
            ),
            "sensitivity": sensitivity,
            "noise_std_sum": noise_std_sum,
            "noise_multiplier": noise_std_sum / sensitivity if sensitivity > 0 else 0.0,
            "clip_norm": config.clip_norm,
            "local_epochs": config.local_epochs,
            "vr_mode": "none",
            "beta": 0.0,
            "gradient_access": not use_two_point,
            "oracle": "two_point" if use_two_point else "analytic_gradient",
            "queries_per_estimate": 2 if use_two_point else 0,
            "query_granularity": "component",
            "privacy_adjacency": (
                "fixed client slot; replace one complete client dataset"
            ),
            "privacy_model": "central_dp_trusted_server",
            "secure_aggregation_assumed": False,
            "raw_client_updates_visible_to_server": True,
            "oracle_transcript_visibility": "trusted_local_internal",
            "training_metrics": "trusted_evaluator_only/outside_DP_claim",
            "initialization": initialization_policy(config),
            "metric_gradient_evaluations": (
                (config.n_rounds + 1) * dataset.train_x.shape[0]
            ),
        },
    )
