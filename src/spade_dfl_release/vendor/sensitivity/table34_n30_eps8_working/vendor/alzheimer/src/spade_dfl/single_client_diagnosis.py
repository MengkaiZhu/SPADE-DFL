"""Reproducible validation-only diagnostics for the single-client smoke path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .alzheimer_data import build_validation_split
from .alzheimer_frozen_config import (
    apply_frozen_method_parameters,
    load_frozen_alzheimer_suite_config,
)
from .config import ParameterRecord
from .graph import admm_penalties, build_graph, initialize_auxiliary
from .initialization import initialize_model_states
from .methods import method_config
from .objectives import build_objective, central_difference
from .partitioning import partition_labels
from .single_client_check import EVALUATION_SCOPE, run_single_client_check
from .trainer import _batch_component_queries, _make_memory, run_experiment

_DIAGNOSTIC_SEEDS = (0, 1, 2)
_DIAGNOSTIC_HORIZONS = (3, 50)
_FIXTURE_ROWS = 32
_ESTIMATE_REPETITIONS = 4096


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def _single_client_config(config_path: Path, *, seed: int, rounds: int):
    suite = load_frozen_alzheimer_suite_config(config_path)
    base = method_config(suite.experiment, "spade_dfl", (seed,), "iid", 0.3)
    frozen = apply_frozen_method_parameters(base, suite=suite, method="spade_dfl")
    sources = dict(frozen.parameter_sources)
    sources["n_clients"] = ParameterRecord(
        "n_clients",
        1,
        "ProtocolAssumption",
        "Single-client diagnostic configuration: one logical client and no communication",
    )
    return replace(
        frozen,
        n_clients=1,
        n_rounds=rounds,
        seeds=(seed,),
        parameter_sources=sources,
    )


def _alignment(split, config_path: Path) -> list[dict[str, object]]:
    records = []
    for seed in _DIAGNOSTIC_SEEDS:
        config = _single_client_config(config_path, seed=seed, rounds=1)
        theta = initialize_model_states(config, seed, 1)[0]
        objective = build_objective(
            config.objective_model,
            config.regularization,
            feature_dimension=config.input_dimension,
            n_classes=config.n_classes,
        )
        fixture_x = split.bundle.train_x[:_FIXTURE_ROWS]
        fixture_y = split.bundle.train_y[:_FIXTURE_ROWS]
        analytic = objective.gradient(theta, fixture_x, fixture_y)
        symmetric = central_difference(
            lambda point, objective=objective, x=fixture_x, y=fixture_y: objective.value(
                point, x, y
            ),
            theta,
        )
        rng = np.random.default_rng(20_260_725 + seed)
        estimates = []
        for _ in range(_ESTIMATE_REPETITIONS):
            sample_ids = rng.choice(
                fixture_x.shape[0], size=config.batch_size, replace=True
            )
            estimates.append(
                _batch_component_queries(
                    objective,
                    theta,
                    fixture_x[sample_ids],
                    fixture_y[sample_ids],
                    config.mu0,
                    rng,
                    config.sigma_u,
                    config.query_noise_std,
                ).mean(axis=0)
            )
        values = np.asarray(estimates)
        mean = values.mean(axis=0)
        coordinate_variance = values.var(axis=0, ddof=1)
        records.append(
            {
                "seed": seed,
                "fixture_rows": int(fixture_x.shape[0]),
                "diagnostic_query_count": _ESTIMATE_REPETITIONS * config.batch_size,
                "analytic_gradient_norm": float(np.linalg.norm(analytic)),
                "symmetric_difference_relative_error": float(
                    np.linalg.norm(symmetric - analytic) / np.linalg.norm(analytic)
                ),
                "mean_over_mu_cosine_to_analytic": _cosine(mean / config.mu0, analytic),
                "mean_over_mu_relative_error": float(
                    np.linalg.norm(mean / config.mu0 - analytic) / np.linalg.norm(analytic)
                ),
                "coordinate_variance_mean": float(coordinate_variance.mean()),
                "mean_estimate_standard_error_norm": float(
                    np.sqrt(coordinate_variance.sum() / _ESTIMATE_REPETITIONS)
                ),
            }
        )
    return records


def _exact_replay(split, config_path: Path) -> list[dict[str, object]]:
    records = []
    for seed in _DIAGNOSTIC_SEEDS:
        config = _single_client_config(config_path, seed=seed, rounds=1)
        partition = partition_labels(
            split.bundle.train_y, 1, "iid", np.random.default_rng(seed), alpha=0.3
        )
        actual = run_experiment(config, split.bundle, seed, partition=partition)
        objective = build_objective(
            config.objective_model,
            config.regularization,
            feature_dimension=config.input_dimension,
            n_classes=config.n_classes,
        )
        states = initialize_model_states(config, seed, 1)
        graph = build_graph(
            1,
            topology=config.graph_topology,
            probability=config.graph_probability,
            seed=seed,
        )
        auxiliary = initialize_auxiliary(states, graph)
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
        local_x = split.bundle.train_x[partition.indices[0]]
        local_y = split.bundle.train_y[partition.indices[0]]
        memory = _make_memory(
            objective, states[0], local_x, local_y, config.mu0, rng, config
        )
        penalties = admm_penalties(states, auxiliary, graph, config.rho)
        phi = states[0].copy()
        for _ in range(config.tau):
            sample_ids = rng.choice(
                local_x.shape[0], size=config.batch_size, replace=True
            )
            queries = _batch_component_queries(
                objective,
                phi,
                local_x[sample_ids],
                local_y[sample_ids],
                config.mu0,
                rng,
                config.sigma_u,
                config.query_noise_std,
            )
            direction = memory.control_variate(sample_ids, queries)
            memory.update(sample_ids, queries)
            phi -= config.eta0 * (direction + config.mu0 * config.beta * penalties[0])
        records.append(
            {
                "seed": seed,
                "state_max_abs_difference": float(
                    np.max(np.abs(actual.states[0] - phi))
                ),
                "communication_rounds": int(actual.metadata["communication_rounds"]),
                "transmitted_bytes": int(actual.transmitted_bytes),
                "admm_penalty_l2": float(np.linalg.norm(penalties)),
                "auxiliary_edge_count": len(auxiliary),
                "query_count": int(actual.query_count),
                "expected_query_count": int(
                    local_x.shape[0] + config.tau * config.batch_size
                ),
            }
        )
    return records


def run_single_client_diagnosis(config_path: Path) -> dict[str, object]:
    """Diagnose frozen one-client behavior without using final-test data."""
    config_path = Path(config_path).resolve()
    suite = load_frozen_alzheimer_suite_config(config_path)
    split = build_validation_split(
        Path(str(suite.experiment.data_path)), split_seed=suite.split_seed
    )
    runs = [
        {
            "rounds": rounds,
            "seed": seed,
            **run_single_client_check(config_path, rounds=rounds, seed=seed),
        }
        for rounds in _DIAGNOSTIC_HORIZONS
        for seed in _DIAGNOSTIC_SEEDS
    ]
    alignment = _alignment(split, config_path)
    exact_replay = _exact_replay(split, config_path)
    mechanics = bool(
        all(record["mean_over_mu_cosine_to_analytic"] > 0.0 for record in alignment)
        and all(
            record["symmetric_difference_relative_error"] < 1e-6
            for record in alignment
        )
        and all(record["state_max_abs_difference"] == 0.0 for record in exact_replay)
        and all(record["communication_rounds"] == 0 for record in exact_replay)
        and all(record["transmitted_bytes"] == 0 for record in exact_replay)
        and all(
            record["query_count"] == record["expected_query_count"]
            for record in exact_replay
        )
    )
    return {
        "evaluation_scope": EVALUATION_SCOPE,
        "outer_final_test_accessed": False,
        "single_client_utility_accepted": all(
            bool(run["loss_decreased"]) for run in runs
        ),
        "mechanical_estimator_accepted": mechanics,
        "runs": runs,
        "alignment": alignment,
        "exact_replay": exact_replay,
        "diagnostic_scope": "inner_training_and_inner_validation_only",
    }
