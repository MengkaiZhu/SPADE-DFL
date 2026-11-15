from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .baselines import train_exact_gradient
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
from .partitioning import PartitionResult, partition_labels
from .privacy import private_release, spade_private_schedule, zcdp_epsilon
from .vr import ZOMemory
from .zo import (
    spade_one_point_batch_queries,
    spade_one_point_query,
)

VR_STAGES = {"single_vr", "multi_vr_admm", "spade_dfl", "spade_dfl_private"}
MULTI_STAGES = {"multi_zo_admm", "multi_vr_admm", "spade_dfl", "spade_dfl_private"}


@dataclass(frozen=True)
class RunResult:
    states: np.ndarray
    history: dict[str, np.ndarray]
    query_count: int
    transmitted_bytes: int
    wall_time: float
    metadata: dict[str, object]


def _partition_indices(n_samples: int, n_clients: int, rng: np.random.Generator) -> list[np.ndarray]:
    if n_clients > n_samples:
        raise ValueError("n_clients cannot exceed the number of training samples")
    return [part.astype(np.int64) for part in np.array_split(rng.permutation(n_samples), n_clients)]


def _component_query(
    objective: Objective,
    theta: np.ndarray,
    x: np.ndarray,
    y: float,
    mu: float,
    rng: np.random.Generator,
    sigma_u: float,
    query_noise_std: float,
) -> np.ndarray:
    u = float(rng.normal(1.0, sigma_u)) if sigma_u > 0.0 else 1.0

    def oracle(point: np.ndarray) -> float:
        noise = float(rng.normal(0.0, query_noise_std)) if query_noise_std > 0.0 else 0.0
        return objective.value(point, x[None, :], np.array([y]), u=u) + noise

    return spade_one_point_query(
        oracle,
        theta,
        mu,
        rng,
        baseline=float(getattr(objective, "one_point_baseline", 0.0)),
    ).vector


def _batch_component_queries(
    objective: Objective,
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: float,
    rng: np.random.Generator,
    sigma_u: float,
    query_noise_std: float,
) -> np.ndarray:
    count = x.shape[0]
    dimension = theta.size
    multiplicative_noise = (
        rng.normal(1.0, sigma_u, size=count)
        if sigma_u > 0.0
        else np.ones(count, dtype=np.float64)
    )
    signs = rng.choice(np.array([-1.0, 1.0]), size=(count, dimension))
    directions = signs / np.sqrt(float(dimension))
    additive_noise = (
        rng.normal(0.0, query_noise_std, size=count)
        if query_noise_std > 0.0
        else np.zeros(count, dtype=np.float64)
    )
    queries = spade_one_point_batch_queries(
        objective,
        theta,
        x,
        y,
        mu,
        directions,
        multiplicative_noise,
        additive_noise,
    )
    baseline = float(getattr(objective, "one_point_baseline", 0.0))
    if baseline:
        queries -= directions * baseline * float(dimension)
    return queries


def _make_memory(
    objective: Objective,
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: float,
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> ZOMemory:
    entries = _batch_component_queries(
        objective,
        state,
        x,
        y,
        mu,
        rng,
        config.sigma_u,
        config.query_noise_std,
    )
    return ZOMemory(entries, mode=config.vr_mode, initial_queries=x.shape[0])


def _refresh_memory(
    memory: ZOMemory,
    objective: Objective,
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: float,
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> None:
    entries = _batch_component_queries(
        objective,
        state,
        x,
        y,
        mu,
        rng,
        config.sigma_u,
        config.query_noise_std,
    )
    memory.refresh(entries)


def _run_exact_gradient(
    config: ExperimentConfig,
    dataset: DatasetBundle,
    seed: int,
    objective: Objective,
) -> RunResult:
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-1.0, 1.0, size=config.dimension)
    start = time.perf_counter()
    result = train_exact_gradient(
        theta0=theta0,
        gradient=lambda theta, _: objective.gradient(
            theta, dataset.train_x, dataset.train_y, u=1.0
        ),
        evaluate=lambda theta: objective.value(theta, dataset.train_x, dataset.train_y, u=1.0),
        n_rounds=config.n_rounds,
        eta0=config.exact_eta0,
        eta_exponent=config.eta_exponent,
        rng=rng,
    )
    accuracy_history = np.asarray(
        [objective.accuracy(theta, dataset.test_x, dataset.test_y) for theta in result.theta_history],
        dtype=np.float64,
    )
    train_accuracy_history = np.asarray(
        [objective.accuracy(theta, dataset.train_x, dataset.train_y) for theta in result.theta_history],
        dtype=np.float64,
    )
    gradient_norm_history = np.asarray(
        [
            np.linalg.norm(objective.gradient(theta, dataset.train_x, dataset.train_y, u=1.0))
            for theta in result.theta_history
        ],
        dtype=np.float64,
    )
    stationarity_history = gradient_norm_history**2
    return RunResult(
        states=result.theta[None, :],
        history={
            "loss": result.losses,
            "accuracy": accuracy_history,
            "train_accuracy": train_accuracy_history,
            "generalization_error": train_accuracy_history - accuracy_history,
            "gradient_norm": gradient_norm_history,
            "stationarity": stationarity_history,
            "consensus_error": np.zeros_like(result.losses),
            "stationarity_consensus": stationarity_history,
            "penalty_energy": np.zeros_like(result.losses),
        },
        query_count=0,
        transmitted_bytes=0,
        wall_time=time.perf_counter() - start,
        metadata={
            "stage": "exact_gradient",
            "objective_model": config.objective_model,
            "seed": seed,
            "beta": 0.0,
            "vr_mode": "none",
            "communication_rounds": 0,
            "gradient_evaluations": config.n_rounds * dataset.train_x.shape[0],
            "epsilon": None,
            "delta": 0.0,
            "privacy_scope": "none",
            "privacy_valid": True,
        },
    )


def run_experiment(
    config: ExperimentConfig,
    dataset: DatasetBundle,
    seed: int,
    partition: PartitionResult | None = None,
) -> RunResult:
    if dataset.train_x.shape[1] != config.input_dimension:
        raise ValueError("dataset feature dimension does not match configuration")
    if dataset.model_dimension not in {0, config.dimension}:
        raise ValueError("dataset model dimension does not match configuration")
    objective = build_objective(
        config.objective_model,
        config.regularization,
        feature_dimension=config.input_dimension,
        n_classes=config.n_classes,
    )
    if config.stage == "exact_gradient":
        return _run_exact_gradient(config, dataset, seed, objective)

    n_clients = config.n_clients if config.stage in MULTI_STAGES else 1
    graph = build_graph(
        n_clients,
        topology=config.graph_topology,
        probability=config.graph_probability,
        seed=seed,
    )
    beta = (
        config.beta
        if config.beta > 0.0
        else derive_midpoint_beta(graph, config.tau, config.rho)
        if n_clients > 1
        else 0.0
    )
    if n_clients > 1:
        lambda_u = float(graph.laplacian_eigenvalues[-1])
        lower = 1.0 / (config.tau * lambda_u * config.rho)
        upper = 2.0 / (config.tau * lambda_u * config.rho)
        if not lower <= beta < upper:
            raise ValueError(f"beta={beta} violates [{lower}, {upper})")

    seed_sequence = np.random.SeedSequence(seed)
    children = seed_sequence.spawn(n_clients + 1)
    partition_rng = np.random.default_rng(children[0])
    client_rngs = [np.random.default_rng(child) for child in children[1:]]
    if partition is None:
        partition = partition_labels(
            dataset.train_y,
            n_clients,
            config.partition_mode,
            partition_rng,
            alpha=config.dirichlet_alpha,
            pathological_minor_fraction=config.pathological_minor_fraction,
        )
    if len(partition.indices) != n_clients:
        raise ValueError("partition client count does not match configuration")
    combined_indices = np.concatenate(partition.indices)
    if not np.array_equal(np.sort(combined_indices), np.arange(dataset.train_x.shape[0])):
        raise ValueError("partition must cover every training sample exactly once")
    partitions = list(partition.indices)
    local_x = [dataset.train_x[indices] for indices in partitions]
    local_y = [dataset.train_y[indices] for indices in partitions]
    privacy_delta = (
        config.target_delta
        if config.target_delta > 0.0
        else 1.0 / min(len(indices) for indices in partitions)
    )
    clip_norm_schedule = np.full(config.n_rounds, config.clip_norm, dtype=np.float64)
    release_noise_std_schedule = np.full(
        config.n_rounds, config.dp_noise_std, dtype=np.float64
    )
    rho_steps: np.ndarray | None = None
    if config.stage == "spade_dfl_private" and config.target_epsilon > 0.0:
        rounds = np.arange(1.0, config.n_rounds + 1.0, dtype=np.float64)
        effective_step_weights = rounds ** -(
            config.eta_exponent + config.mu_exponent
        )
        clip_norm_schedule, release_noise_std_schedule, rho_steps = spade_private_schedule(
            config.n_rounds,
            config.clip_norm,
            config.clip_exponent,
            privacy_delta,
            config.target_epsilon,
            utility_weights=effective_step_weights,
            allocation=config.privacy_allocation,
        )
    initialization = initialization_policy(config)
    states = initialize_model_states(config, seed, n_clients)
    z_edges = initialize_auxiliary(states, graph)
    start = time.perf_counter()
    use_vr = config.stage in VR_STAGES
    memories: list[ZOMemory | None] = [None] * n_clients
    if use_vr:
        for i in range(n_clients):
            memories[i] = _make_memory(
                objective,
                states[i],
                local_x[i],
                local_y[i],
                config.mu0,
                client_rngs[i],
                config,
            )

    role_history_fields = (
        ("private_local_accuracy", "released_proxy_accuracy")
        if config.stage == "spade_dfl_private"
        else ()
    )
    history = new_history(*role_history_fields)
    penalties = admm_penalties(states, z_edges, graph, config.rho)
    record_metrics(history, states, objective, dataset, penalties)
    plain_queries = 0
    directed_edges = int(graph.adjacency.sum())
    transmitted_bytes = 0
    unclipped_update_norm_mean: list[float] = []
    clip_fraction: list[float] = []
    for k in range(config.n_rounds):
        eta = config.eta0 * (k + 1.0) ** (-config.eta_exponent)
        mu = config.mu0 * (k + 1.0) ** (-config.mu_exponent)
        round_start = states.copy()
        penalties = admm_penalties(round_start, z_edges, graph, config.rho)
        private_local_models = np.empty_like(states)

        for i in range(n_clients):
            rng = client_rngs[i]
            memory = memories[i]
            if use_vr and config.vr_mode == "full_refresh" and k > 0:
                assert memory is not None
                _refresh_memory(
                    memory,
                    objective,
                    round_start[i],
                    local_x[i],
                    local_y[i],
                    mu,
                    rng,
                    config,
                )
            phi = round_start[i].copy()
            for _ in range(config.tau):
                batch_ids = rng.choice(local_x[i].shape[0], size=config.batch_size, replace=True)
                queries = _batch_component_queries(
                    objective,
                    phi,
                    local_x[i][batch_ids],
                    local_y[i][batch_ids],
                    mu,
                    rng,
                    config.sigma_u,
                    config.query_noise_std,
                )
                if use_vr:
                    assert memory is not None
                    direction = memory.control_variate(batch_ids, queries)
                    memory.update(batch_ids, queries)
                else:
                    direction = queries.mean(axis=0)
                    plain_queries += config.batch_size
                phi -= eta * (direction + mu * beta * penalties[i])
                if not np.all(np.isfinite(phi)):
                    raise FloatingPointError(f"non-finite local state at round {k}, client {i}")
            private_local_models[i] = phi

        if config.stage == "spade_dfl_private":
            history["private_local_accuracy"].append(
                objective.accuracy(
                    private_local_models.mean(axis=0),
                    dataset.test_x,
                    dataset.test_y,
                )
            )
            update_norms = np.linalg.norm(
                private_local_models - round_start, axis=1
            )
            unclipped_update_norm_mean.append(float(update_norms.mean()))
            clip_fraction.append(
                float(np.mean(update_norms > float(clip_norm_schedule[k])))
            )
            released_proxy_models = np.empty_like(states)
            for i in range(n_clients):
                released_proxy_models[i] = private_release(
                    round_start[i],
                    private_local_models[i] - round_start[i],
                    float(clip_norm_schedule[k]),
                    float(release_noise_std_schedule[k]),
                    client_rngs[i],
                ).released
            states = released_proxy_models
        else:
            states = private_local_models

        if n_clients > 1:
            _, z_edges = exchange_messages(states, z_edges, graph, config.rho)
            transmitted_bytes += directed_edges * config.dimension * np.dtype(np.float64).itemsize
        penalties_after = admm_penalties(states, z_edges, graph, config.rho)
        record_metrics(history, states, objective, dataset, penalties_after)
        if config.stage == "spade_dfl_private":
            history["released_proxy_accuracy"].append(history["accuracy"][-1])

    query_count = (
        sum(memory.query_count for memory in memories if memory is not None)
        if use_vr
        else plain_queries
    )
    metadata: dict[str, object] = {
        "stage": config.stage,
        "objective_model": config.objective_model,
        "sigma_u": config.sigma_u,
        "query_noise_std": config.query_noise_std,
        "seed": seed,
        "beta": beta,
        "vr_mode": config.vr_mode if use_vr else "none",
        "n_clients": n_clients,
        "graph_topology": config.graph_topology,
        "graph_probability": config.graph_probability,
        "partition_mode": partition.mode,
        "dirichlet_alpha": partition.alpha,
        "pathological_minor_fraction": config.pathological_minor_fraction,
        "client_sample_counts": [len(indices) for indices in partition.indices],
        "label_histograms": partition.label_histograms.tolist(),
        "communication_rounds": config.n_rounds if n_clients > 1 else 0,
        "gradient_oracle": "one_point",
        "gradient_evaluations": 0,
        "epsilon": None,
        "delta": 0.0,
        "privacy_scope": "none",
        "privacy_valid": True,
        "shared_initialization": config.shared_initialization,
        "initialization": initialization,
        "privacy_allocation": config.privacy_allocation,
    }
    if config.stage == "spade_dfl_private":
        metadata.update(
            {
                "unclipped_update_norm_mean": unclipped_update_norm_mean,
                "clip_fraction": clip_fraction,
                "model_roles": {
                    "private_local_model": "y_i,k+1 from SPADE-DFL Eq. (18)",
                    "released_proxy_model": "x_i,k+1 from SPADE-DFL Eq. (20)",
                },
                "communicated_model_role": (
                    "released_proxy_model" if n_clients > 1 else "not_communicated"
                ),
                "private_model_communicated": False,
            }
        )
    if config.stage == "spade_dfl_private" and np.any(release_noise_std_schedule > 0.0):
        if rho_steps is None:
            epsilon, rho_zcdp = zcdp_epsilon(
                config.n_rounds,
                config.clip_norm,
                config.dp_noise_std,
                privacy_delta,
            )
        else:
            rho_zcdp = float(rho_steps.sum())
            epsilon = float(
                rho_zcdp
                + 2.0 * np.sqrt(rho_zcdp * np.log(1.0 / privacy_delta))
            )
        metadata.update(
            {
                "epsilon": epsilon,
                "delta": privacy_delta,
                "rho_zcdp": rho_zcdp,
                "clip_norm_schedule": clip_norm_schedule.tolist(),
                "release_noise_std_schedule": release_noise_std_schedule.tolist(),
                "privacy_rho_schedule": rho_steps.tolist() if rho_steps is not None else None,
                "privacy_scope": "record_within_client_transcript",
                "privacy_adjacency": (
                    "fixed client i; local datasets D_i and D_i' differ in one sample"
                ),
            }
        )
    return RunResult(
        states=states,
        history={name: np.asarray(values, dtype=np.float64) for name, values in history.items()},
        query_count=int(query_count),
        transmitted_bytes=int(transmitted_bytes),
        wall_time=time.perf_counter() - start,
        metadata=metadata,
    )
