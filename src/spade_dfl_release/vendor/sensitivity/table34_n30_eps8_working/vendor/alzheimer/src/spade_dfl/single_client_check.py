"""Validation-only single-client SPADE-DFL smoke diagnostic."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .alzheimer_data import AlzheimerSplit, build_validation_split
from .alzheimer_frozen_config import (
    FrozenAlzheimerSuiteConfig,
    apply_frozen_method_parameters,
    load_frozen_alzheimer_suite_config,
)
from .config import ParameterRecord
from .data import DatasetBundle
from .initialization import initialize_model_states
from .methods import method_config
from .objectives import build_objective
from .partitioning import partition_labels
from .trainer import run_experiment

EVALUATION_SCOPE = "inner_validation_smoke_only"
_FORMAL_ROLES = {"scientific_base", "scientific_local", "scientific_server"}


def _validate_inner_validation_split(split: AlzheimerSplit) -> DatasetBundle:
    """Admit only a bundle that `build_validation_split` structurally produced."""
    if (
        split.validation_x is None
        or split.validation_y is None
        or split.validation_indices.size == 0
        or split.bundle.test_x.shape != split.validation_x.shape
        or split.bundle.test_y.shape != split.validation_y.shape
        or not np.array_equal(split.bundle.test_x, split.validation_x)
        or not np.array_equal(split.bundle.test_y, split.validation_y)
    ):
        raise ValueError("single-client check requires an inner validation split")
    return split.bundle


def _load_inner_validation_bundle(
    config_path: Path,
) -> tuple[FrozenAlzheimerSuiteConfig, DatasetBundle]:
    if not isinstance(config_path, (str, Path)):
        raise TypeError("config_path must be a locked frozen configuration path")
    frozen_path = Path(config_path).resolve()
    suite = load_frozen_alzheimer_suite_config(frozen_path)
    if (
        suite.protocol_role not in _FORMAL_ROLES
        or suite.evaluation_scope != "outer_final_test_frozen"
    ):
        raise ValueError("single-client check requires a frozen formal configuration")
    split = build_validation_split(
        Path(str(suite.experiment.data_path)), split_seed=suite.split_seed
    )
    return suite, _validate_inner_validation_split(split)


def _run_single_client_mechanics(
    bundle: DatasetBundle,
    *,
    suite: FrozenAlzheimerSuiteConfig,
    rounds: int,
    seed: int,
) -> dict[str, object]:
    """Compute single-client mechanics only; this helper makes no provenance claim."""
    base = method_config(suite.experiment, "spade_dfl", (int(seed),), "iid", 0.3)
    frozen = apply_frozen_method_parameters(base, suite=suite, method="spade_dfl")
    sources = dict(frozen.parameter_sources)
    sources["n_clients"] = ParameterRecord(
        "n_clients",
        1,
        "ProtocolAssumption",
        "Single-client diagnostic configuration: one logical client and no communication",
    )
    config = replace(
        frozen,
        n_clients=1,
        n_rounds=int(rounds),
        seeds=(int(seed),),
        parameter_sources=sources,
    )
    partition = partition_labels(
        bundle.train_y, 1, "iid", np.random.default_rng(seed), alpha=0.3
    )
    objective = build_objective(
        config.objective_model,
        config.regularization,
        feature_dimension=config.input_dimension,
        n_classes=config.n_classes,
    )
    initial_state = initialize_model_states(config, seed, 1)[0]
    result = run_experiment(config, bundle, seed, partition=partition)
    final_state = result.states[0]
    initial_loss = float(objective.value(initial_state, bundle.test_x, bundle.test_y))
    final_loss = float(objective.value(final_state, bundle.test_x, bundle.test_y))
    initial_accuracy = float(objective.accuracy(initial_state, bundle.test_x, bundle.test_y))
    final_accuracy = float(objective.accuracy(final_state, bundle.test_x, bundle.test_y))
    expected_query_count = int(
        bundle.train_x.shape[0] + config.n_rounds * config.tau * config.batch_size
    )
    finite = bool(
        np.all(np.isfinite(result.states))
        and all(np.all(np.isfinite(values)) for values in result.history.values())
        and all(
            np.isfinite(value)
            for value in (initial_loss, final_loss, initial_accuracy, final_accuracy)
        )
    )
    return {
        "finite": finite,
        "logical_clients": 1,
        "communication_rounds": int(result.metadata["communication_rounds"]),
        "query_count": int(result.query_count),
        "expected_query_count": expected_query_count,
        "query_count_exact": bool(result.query_count == expected_query_count),
        "history_length": int(result.history["loss"].size),
        "history": {
            name: [float(value) for value in values]
            for name, values in result.history.items()
        },
        "initial_training_loss": float(result.history["loss"][0]),
        "final_training_loss": float(result.history["loss"][-1]),
        "initial_validation_loss": initial_loss,
        "final_validation_loss": final_loss,
        "initial_validation_accuracy": initial_accuracy,
        "final_validation_accuracy": final_accuracy,
        "loss_decreased": bool(final_loss < initial_loss),
        "vr_mode": str(result.metadata["vr_mode"]),
        "gradient_oracle": str(result.metadata["gradient_oracle"]),
    }


def run_single_client_check(
    config_path: Path,
    *,
    rounds: int,
    seed: int,
) -> dict[str, object]:
    """Run frozen SPADE-DFL on a validated inner training/validation split only."""
    suite, bundle = _load_inner_validation_bundle(config_path)
    mechanics = _run_single_client_mechanics(
        bundle, suite=suite, rounds=rounds, seed=seed
    )
    return {
        **mechanics,
        "evaluation_scope": EVALUATION_SCOPE,
        "outer_final_test_accessed": False,
        "outer_final_test_firewall": (
            "validated frozen formal config plus build_validation_split inner bundle"
        ),
    }
