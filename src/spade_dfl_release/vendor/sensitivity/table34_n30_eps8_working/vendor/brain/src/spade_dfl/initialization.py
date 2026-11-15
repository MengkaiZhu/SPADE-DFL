from __future__ import annotations

import numpy as np

from .config import ExperimentConfig


def initialization_policy(config: ExperimentConfig) -> str:
    """Resolve the retained legacy shared-initialization flag."""
    if config.shared_initialization:
        return "shared_uniform"
    return config.initialization


def initialize_model_states(
    config: ExperimentConfig,
    seed: int,
    n_clients: int,
) -> np.ndarray:
    """Create a reproducible initial model for every participating client."""
    if n_clients <= 0:
        raise ValueError("n_clients must be positive")
    policy = initialization_policy(config)
    shape = (n_clients, config.dimension)
    if policy == "shared_zero":
        return np.zeros(shape, dtype=np.float64)
    if policy == "shared_uniform":
        state = np.random.default_rng(seed).uniform(
            -1.0, 1.0, size=config.dimension
        )
        return np.repeat(state[None, :], n_clients, axis=0)
    if policy == "independent_uniform":
        child_seeds = np.random.SeedSequence(seed).spawn(n_clients)
        return np.vstack(
            [
                np.random.default_rng(child).uniform(
                    -1.0, 1.0, size=config.dimension
                )
                for child in child_seeds
            ]
        )
    raise ValueError(f"unsupported initialization policy: {policy}")
