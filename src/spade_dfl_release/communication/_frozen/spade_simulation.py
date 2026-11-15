"""Simulation primitives for the SPADE-DFL communication-cost figure.

This module follows Algorithm 1 and Eq. (25) of the 2026-09-07 manuscript.
The experiment is deliberately non-private: clipping error, release noise, and
query noise are all zero so that the communication/local-update trade-off is
isolated.
"""

from dataclasses import asdict, dataclass
import math

import mpmath
import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    n_clients: int = 20
    samples_per_client: int = 100
    dimension: int = 20
    batch_size: int = 8
    rho: float = 1.0
    eta0: float = 1.0
    mu0: float = 1.0


def _validate_config(config):
    if config.n_clients < 3:
        raise ValueError("a simple cycle requires at least three clients")
    if config.samples_per_client < 1 or config.dimension < 1:
        raise ValueError("sample count and dimension must be positive")
    if not 1 <= config.batch_size <= config.samples_per_client:
        raise ValueError("batch_size must lie in [1, samples_per_client]")
    if min(config.rho, config.eta0, config.mu0) <= 0:
        raise ValueError("rho, eta0, and mu0 must be positive")


def build_budget_schedule():
    roots = (10, 15, 22, 31)
    return [
        {
            "T": root**3,
            "tau_local": root,
            "K_local": root**2,
            "tau_baseline": 1,
            "K_baseline": root**3,
        }
        for root in roots
    ]


def cumulative_transmitted_scalars(n_clients, dimension, rounds):
    if n_clients < 3 or dimension < 1 or rounds < 0:
        raise ValueError("invalid cycle communication arguments")
    n_edges = n_clients
    return int(2 * n_edges * dimension * rounds)


def _student_t_cdf_positive(value, degrees_of_freedom):
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail_twice = mpmath.betainc(
        degrees_of_freedom / 2.0,
        0.5,
        0,
        x,
        regularized=True,
    )
    return 1.0 - 0.5 * float(tail_twice)


def _student_t_critical_975(degrees_of_freedom):
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    low, high = 0.0, 20.0
    for _ in range(90):
        middle = (low + high) / 2.0
        if _student_t_cdf_positive(middle, degrees_of_freedom) < 0.975:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def mean_ci95(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a finite one-dimensional sample with n >= 2")
    mean = float(array.mean())
    sem = float(array.std(ddof=1) / math.sqrt(array.size))
    half_width = _student_t_critical_975(array.size - 1) * sem
    return mean, mean - half_width, mean + half_width


def _sigmoid(values):
    # The generated features and manuscript schedule keep logits moderate, but
    # clipping here prevents overflow without changing any representable result.
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def generate_synthetic_data(seed, config=SimulationConfig()):
    """Generate the bounded nonconvex diagnostic specified in Sec. V-C1."""
    _validate_config(config)
    data_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 104729]))
    shape = (config.n_clients, config.samples_per_client, config.dimension)
    features = data_rng.normal(size=shape)
    norms = np.linalg.norm(features, axis=2, keepdims=True)
    features = features / np.maximum(norms, 1.0)

    teacher = data_rng.normal(size=config.dimension)
    teacher /= max(float(np.linalg.norm(teacher)), 1.0)
    biases = np.linspace(-1.0, 1.0, config.n_clients, dtype=np.float64)
    logits = np.einsum("nmd,d->nm", features, teacher) + biases[:, None]
    probabilities = _sigmoid(logits)
    labels = (data_rng.random(size=probabilities.shape) < probabilities).astype(np.float64)
    return features.astype(np.float64), labels


def _full_gradient_squared(features_flat, labels_flat, point):
    logits = features_flat @ point
    probabilities = _sigmoid(logits)
    weights = (probabilities - labels_flat) * probabilities * (1.0 - probabilities)
    gradient = features_flat.T @ weights / labels_flat.size
    return float(gradient @ gradient)


def _one_point_queries(features, labels, points, mu, rng):
    """Vectorized normalized-Rademacher implementation of Eq. (3)."""
    dimension = features.shape[-1]
    signs = rng.integers(0, 2, size=features.shape, dtype=np.int8)
    directions = (2.0 * signs.astype(np.float64) - 1.0) / math.sqrt(dimension)
    perturbed = points + mu * directions
    logits = np.sum(features * perturbed, axis=-1)
    loss = 0.5 * (_sigmoid(logits) - labels) ** 2
    # For normalized Rademacher directions E[uu^T] = I/d, hence 1/c_u = d.
    return dimension * directions * loss[..., None]


def _cycle_laplacian_max_eigenvalue(n_clients):
    # Eigenvalues of C_N are 2 - 2 cos(2 pi j/N).
    indices = np.arange(n_clients, dtype=np.float64)
    return float(np.max(2.0 - 2.0 * np.cos(2.0 * np.pi * indices / n_clients)))


def run_spade(seed, total_local_steps, tau, config=SimulationConfig()):
    """Run one seeded non-private SPADE-DFL simulation.

    The returned criterion is the finite-sample analogue of S_K: the exact
    empirical full-objective gradient norm is evaluated at every averaged local
    iterate, and released-state disagreement is evaluated at every round start.
    These offline gradients never enter the optimizer.
    """
    _validate_config(config)
    total_local_steps = int(total_local_steps)
    tau = int(tau)
    if total_local_steps < 1 or tau < 1 or total_local_steps % tau:
        raise ValueError("total_local_steps must be positive and divisible by tau")

    rounds = total_local_steps // tau
    features, labels = generate_synthetic_data(seed, config)
    features_flat = features.reshape(-1, config.dimension)
    labels_flat = labels.reshape(-1)
    query_rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(tau), 130363]))

    eta = config.eta0 * total_local_steps ** (-0.5)
    mu = config.mu0 * total_local_steps ** (-1.0 / 6.0)
    lambda_max = _cycle_laplacian_max_eigenvalue(config.n_clients)
    a0 = 1.0 / lambda_max
    beta = a0 / (tau * eta * mu)

    x = np.zeros((config.n_clients, config.dimension), dtype=np.float64)
    z_left = np.zeros_like(x)
    z_right = np.zeros_like(x)
    stationarity_sum = 0.0
    consensus_sum = 0.0

    client_index = np.arange(config.n_clients)[:, None]
    for _round in range(rounds):
        x_bar = x.mean(axis=0)
        consensus_sum += float(np.mean(np.sum((x - x_bar) ** 2, axis=1)))

        penalty = config.rho * 2.0 * x - z_left - z_right
        phi = x.copy()

        reference_points = np.broadcast_to(
            phi[:, None, :],
            (config.n_clients, config.samples_per_client, config.dimension),
        )
        memory = _one_point_queries(features, labels, reference_points, mu, query_rng)
        memory_average = memory.mean(axis=1)

        for _local_step in range(tau):
            stationarity_sum += _full_gradient_squared(
                features_flat,
                labels_flat,
                phi.mean(axis=0),
            )
            batches = np.empty((config.n_clients, config.batch_size), dtype=np.int64)
            for client in range(config.n_clients):
                batches[client] = query_rng.choice(
                    config.samples_per_client,
                    size=config.batch_size,
                    replace=False,
                )
            selected_features = features[client_index, batches]
            selected_labels = labels[client_index, batches]
            selected_points = np.broadcast_to(
                phi[:, None, :],
                (config.n_clients, config.batch_size, config.dimension),
            )
            new_queries = _one_point_queries(
                selected_features,
                selected_labels,
                selected_points,
                mu,
                query_rng,
            )
            old_queries = memory[client_index, batches]
            direction = (new_queries - old_queries).mean(axis=1) + memory_average
            phi = phi - eta * (direction + mu * beta * penalty)

            memory[client_index, batches] = new_queries
            memory_average += (new_queries - old_queries).sum(axis=1) / config.samples_per_client

        # Non-private specialization of Eq. (8): clipping error and release noise
        # are zero, so the released model equals the local-stage endpoint.
        x_new = phi

        message_left = z_left - 2.0 * config.rho * x_new
        message_right = z_right - 2.0 * config.rho * x_new
        z_left = 0.5 * (z_left - np.roll(message_right, 1, axis=0))
        z_right = 0.5 * (z_right - np.roll(message_left, -1, axis=0))
        x = x_new

    stationarity = stationarity_sum / total_local_steps
    consensus = consensus_sum / rounds
    query_count = config.n_clients * (
        rounds * config.samples_per_client + total_local_steps * config.batch_size
    )
    transmitted_scalars = cumulative_transmitted_scalars(
        config.n_clients,
        config.dimension,
        rounds,
    )
    result = {
        "seed": int(seed),
        "T": total_local_steps,
        "tau": tau,
        "rounds": rounds,
        "local_steps": total_local_steps,
        "eta": float(eta),
        "mu": float(mu),
        "beta": float(beta),
        "a0": float(a0),
        "stationarity": float(stationarity),
        "consensus": float(consensus),
        "criterion": float(stationarity + consensus),
        "query_count": int(query_count),
        "transmitted_scalars": int(transmitted_scalars),
        "config": asdict(config),
    }
    if not all(
        np.isfinite(result[key])
        for key in ("stationarity", "consensus", "criterion", "eta", "mu", "beta")
    ):
        raise FloatingPointError("non-finite simulation result")
    return result
