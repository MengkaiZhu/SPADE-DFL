from __future__ import annotations

import time

import numpy as np

from .config import ExperimentConfig
from .data import DatasetBundle
from .evaluation import new_history, record_metrics
from .initialization import initialization_policy, initialize_model_states
from .objectives import build_objective
from .partitioning import PartitionResult
from .privacy import (
    clip_update,
    gaussian_composition_epsilon,
    noise_multiplier_for_poisson_sampled_gaussian_budget,
    noise_std_for_gaussian_budget,
    poisson_sampled_gaussian_epsilon,
)
from .trainer import RunResult
from .two_point import symmetric_two_point_component_batch


def _privacy_parameters(
    config: ExperimentConfig,
    sensitivity: float,
    composition_steps: int,
) -> tuple[float, float, float, float]:
    if config.target_epsilon <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    delta = config.target_delta if config.target_delta > 0.0 else 1e-5
    noise_std = noise_std_for_gaussian_budget(
        composition_steps,
        sensitivity,
        delta,
        config.target_epsilon,
    )
    epsilon, rho = gaussian_composition_epsilon(
        composition_steps, sensitivity, noise_std, delta
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


def run_dp_sgd(
    config: ExperimentConfig,
    dataset: DatasetBundle,
    seed: int,
) -> RunResult:
    if config.stage not in {"dp_sgd", "zo2_dp_sgd"}:
        raise ValueError("DP-SGD requires stage=dp_sgd or zo2_dp_sgd")
    rng = np.random.default_rng(seed)
    objective = build_objective(
        config.objective_model,
        config.regularization,
        feature_dimension=config.input_dimension,
        n_classes=config.n_classes,
    )
    state = initialize_model_states(config, seed, 1)[0]
    if config.batch_size > dataset.train_x.shape[0]:
        raise ValueError("DP-SGD batch_size cannot exceed the training set")
    use_two_point = config.stage == "zo2_dp_sgd"
    # Reference [24], Algorithm 1, clips each example to C, adds noise to
    # the sampled lot sum, and uses add/remove example adjacency.  After
    # division by the lot size the sensitivity is C/L.
    sensitivity = config.clip_norm / config.batch_size
    privacy_composition_steps = config.n_rounds * config.outer_local_steps
    sample_rate = config.batch_size / dataset.train_x.shape[0]
    if config.target_epsilon > 0.0:
        delta = config.target_delta if config.target_delta > 0.0 else 1e-5
        noise_multiplier = noise_multiplier_for_poisson_sampled_gaussian_budget(
            privacy_composition_steps,
            sample_rate,
            delta,
            config.target_epsilon,
        )
        noise_std = noise_multiplier * sensitivity
        epsilon = poisson_sampled_gaussian_epsilon(
            privacy_composition_steps,
            sample_rate,
            noise_multiplier,
            delta,
        )
        rho = None
    else:
        noise_std = 0.0
        noise_multiplier = 0.0
        epsilon = 0.0
        delta = 0.0
        rho = None
    history = new_history()
    record_metrics(history, state[None, :], objective, dataset)
    gradient_evaluations = 0
    query_count = 0
    start = time.perf_counter()
    for round_index in range(config.n_rounds):
        for _ in range(config.outer_local_steps):
            ids = rng.choice(
                dataset.train_x.shape[0], size=config.batch_size, replace=False
            )
            if use_two_point:
                mu = config.mu0 * (round_index + 1.0) ** (-config.mu_exponent)
                components, batch_queries = _two_point_components(
                    objective,
                    state,
                    dataset.train_x[ids],
                    dataset.train_y[ids],
                    mu,
                    rng,
                    config,
                )
                query_count += batch_queries
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
                raise FloatingPointError(
                    f"non-finite DP-SGD state at round {round_index}"
                )
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
            "outer_local_steps": config.outer_local_steps,
            "privacy_composition_steps": privacy_composition_steps,
            "gradient_evaluations": gradient_evaluations,
            "epsilon": epsilon if epsilon > 0.0 else None,
            "delta": delta,
            "rho_zcdp": rho,
            "privacy_scope": "example_add_remove",
            "privacy_valid": bool(
                config.target_epsilon <= 0.0
                or np.isclose(epsilon, config.target_epsilon)
                and np.isclose(delta, config.target_delta)
            ),
            "sensitivity": sensitivity,
            "noise_std": noise_std,
            "noise_multiplier": noise_multiplier,
            "clip_norm": config.clip_norm,
            "vr_mode": "none",
            "beta": 0.0,
            "gradient_access": not use_two_point,
            "oracle": "two_point" if use_two_point else "analytic_gradient",
            "queries_per_estimate": 2 if use_two_point else 0,
            "query_granularity": "component",
            "sampling_scheme": "without_replacement",
            "privacy_adjacency": "add or remove one training example",
            "privacy_amplification_used": True,
            "privacy_accountant": "integer_order_poisson_sampled_gaussian_rdp",
            "sample_rate": sample_rate,
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
    objective = build_objective(
        config.objective_model,
        config.regularization,
        feature_dimension=config.input_dimension,
        n_classes=config.n_classes,
    )
    children = np.random.SeedSequence(seed).spawn(config.n_clients + 1)
    server_rng = np.random.default_rng(children[0])
    client_rngs = [np.random.default_rng(child) for child in children[1:]]
    state = initialize_model_states(config, seed, 1)[0]
    use_two_point = config.stage == "zo2_dp_fedavg"
    # Reference [47], Algorithm 1, selects m clients and calibrates the
    # Gaussian mechanism to the S-clipped sum under client adjacency.
    clients_per_round = config.clients_per_round or config.n_clients
    client_sampling_rate = clients_per_round / config.n_clients
    if config.target_epsilon > 0.0 and clients_per_round < config.n_clients:
        delta = config.target_delta if config.target_delta > 0.0 else 1e-3
        noise_multiplier = noise_multiplier_for_poisson_sampled_gaussian_budget(
            config.n_rounds,
            client_sampling_rate,
            delta,
            config.target_epsilon,
        )
        epsilon = poisson_sampled_gaussian_epsilon(
            config.n_rounds,
            client_sampling_rate,
            noise_multiplier,
            delta,
        )
        rho = None
        fixed_noise_std_sum = noise_multiplier * config.clip_norm
    else:
        fixed_noise_std_sum, epsilon, delta, rho = _privacy_parameters(
            config,
            config.clip_norm,
            config.n_rounds,
        )
        noise_multiplier = (
            fixed_noise_std_sum / config.clip_norm if config.clip_norm > 0.0 else 0.0
        )
    history = new_history()
    consensus_states = np.tile(state, (config.n_clients, 1))
    record_metrics(history, consensus_states, objective, dataset)
    gradient_evaluations = 0
    query_count = 0
    round_learning_rate = []
    mean_unclipped_update_norm = []
    raw_update_norms_by_round = []
    clip_fraction = []
    clip_norm_schedule = []
    selected_client_counts = []
    start = time.perf_counter()
    for round_index in range(config.n_rounds):
        eta = config.eta0 * (round_index + 1.0) ** (-config.eta_exponent)
        round_learning_rate.append(float(eta))
        if clients_per_round == config.n_clients:
            selected_clients = np.arange(config.n_clients, dtype=np.int64)
        else:
            selected_clients = np.sort(
                server_rng.choice(
                    config.n_clients,
                    size=clients_per_round,
                    replace=False,
                )
            )
        selected_client_counts.append(int(selected_clients.size))
        raw_updates = []
        for client in selected_clients:
            indices = partition.indices[int(client)]
            local_state = state.copy()
            for _ in range(config.local_epochs):
                order = client_rngs[client].permutation(indices.size)
                for offset in range(0, indices.size, config.batch_size):
                    local_ids = indices[order[offset : offset + config.batch_size]]
                    if use_two_point:
                        mu = config.mu0 * (round_index + 1.0) ** (
                            -config.mu_exponent
                        )
                        components, batch_queries = _two_point_components(
                            objective,
                            local_state,
                            dataset.train_x[local_ids],
                            dataset.train_y[local_ids],
                            mu,
                            client_rngs[client],
                            config,
                        )
                        direction = np.mean(components, axis=0)
                        query_count += batch_queries
                    else:
                        direction = objective.gradient(
                            local_state,
                            dataset.train_x[local_ids],
                            dataset.train_y[local_ids],
                            u=1.0,
                        )
                        gradient_evaluations += local_ids.size
                    local_state -= eta * direction
            raw_update = local_state - state
            raw_updates.append(raw_update)
        update_norms = np.linalg.norm(np.vstack(raw_updates), axis=1)
        raw_update_norms_by_round.append(update_norms.tolist())
        round_clip_norm = (
            float(np.median(update_norms))
            if config.client_clip_mode == "round_median"
            else float(config.clip_norm)
        )
        round_clip_norm = max(round_clip_norm, np.finfo(np.float64).tiny)
        clip_norm_schedule.append(round_clip_norm)
        updates = [clip_update(update, round_clip_norm) for update in raw_updates]
        mean_unclipped_update_norm.append(float(update_norms.mean()))
        clip_fraction.append(float(np.mean(update_norms > round_clip_norm)))
        summed = np.sum(updates, axis=0)
        noise_std_sum = noise_multiplier * round_clip_norm
        if noise_std_sum > 0.0:
            summed += server_rng.normal(
                0.0, noise_std_sum, size=config.dimension
            )
        state = state + summed / selected_clients.size
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(
                f"non-finite DP-FedAvg state at round {round_index}"
            )
        consensus_states = np.tile(state, (config.n_clients, 1))
        record_metrics(history, consensus_states, objective, dataset)
    transmitted_bytes = (
        config.n_rounds
        * clients_per_round
        * 2
        * config.dimension
        * np.dtype(np.float64).itemsize
    )
    nominal_privacy_budget = {
        "epsilon": epsilon if epsilon > 0.0 else None,
        "delta": delta,
    }
    budget_matches_target = bool(
        config.target_epsilon <= 0.0
        or np.isclose(epsilon, config.target_epsilon)
        and np.isclose(delta, config.target_delta)
    )
    strict_privacy_caveat = (
        "Reference [47] round-median clipping is data-dependent and does not "
        "by itself provide a strict DP guarantee without a separately private "
        "clipping-bound mechanism; fixed-size sampling is accounted for with "
        "the Poisson-subsampled RDP approximation. The reported epsilon and "
        "delta are nominal reproduction values."
        if config.client_clip_mode == "round_median"
        else ""
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
            "privacy_scope": "client_add_remove",
            "privacy_valid": budget_matches_target
            and config.client_clip_mode == "fixed",
            "nominal_privacy_budget": nominal_privacy_budget,
            "strict_privacy_caveat": strict_privacy_caveat,
            "sensitivity": (
                config.clip_norm
                if config.client_clip_mode == "fixed"
                else max(clip_norm_schedule)
            ),
            "noise_std_sum": noise_multiplier * clip_norm_schedule[-1],
            "noise_multiplier": noise_multiplier,
            "clip_norm": config.clip_norm,
            "clip_norm_schedule": clip_norm_schedule,
            "client_clip_mode": config.client_clip_mode,
            "clients_per_round": clients_per_round,
            "selected_client_counts": selected_client_counts,
            "client_sampling_rate": client_sampling_rate,
            "local_epochs": config.local_epochs,
            "round_learning_rate": round_learning_rate,
            "mean_unclipped_update_norm": mean_unclipped_update_norm,
            "raw_update_norms_by_round": raw_update_norms_by_round,
            "clip_fraction": clip_fraction,
            "vr_mode": "none",
            "beta": 0.0,
            "gradient_access": not use_two_point,
            "oracle": "two_point" if use_two_point else "analytic_gradient",
            "queries_per_estimate": 2 if use_two_point else 0,
            "query_granularity": "component",
            "privacy_adjacency": (
                "add or remove one complete client contribution"
            ),
            "privacy_model": "central_dp_trusted_server",
            "client_sampling_scheme": "fixed_size_without_replacement",
            "privacy_amplification_used": clients_per_round < config.n_clients,
            "privacy_accountant": (
                "integer_order_poisson_sampled_gaussian_rdp"
                if clients_per_round < config.n_clients
                else "gaussian_zcdp_composition"
            ),
            "adaptive_clip_provenance": (
                "Reference [47], Algorithm 1, round median of selected client update norms"
                if config.client_clip_mode == "round_median"
                else "fixed configured client update bound"
            ),
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
