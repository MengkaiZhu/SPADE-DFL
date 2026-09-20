from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReleaseResult:
    released: np.ndarray
    clipped_update: np.ndarray
    noise: np.ndarray


def clip_update(update: np.ndarray, clip_norm: float) -> np.ndarray:
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    values = np.asarray(update, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    scale = min(1.0, clip_norm / norm) if norm > 0.0 else 1.0
    return values * scale


def private_release(
    state: np.ndarray,
    update: np.ndarray,
    clip_norm: float,
    noise_std: float,
    rng: np.random.Generator,
) -> ReleaseResult:
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative")
    state = np.asarray(state, dtype=np.float64)
    clipped = clip_update(update, clip_norm)
    noise = (
        rng.normal(0.0, noise_std, size=state.shape)
        if noise_std > 0.0
        else np.zeros_like(state)
    )
    released = state + clipped + noise
    if not np.all(np.isfinite(released)):
        raise FloatingPointError("private release is non-finite")
    return ReleaseResult(released=released, clipped_update=clipped, noise=noise)


def gaussian_composition_epsilon(
    n_steps: int,
    sensitivity: float,
    noise_std: float,
    delta: float,
) -> tuple[float, float]:
    if n_steps <= 0 or sensitivity <= 0 or noise_std <= 0:
        raise ValueError("n_steps, sensitivity, and noise_std must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    rho_zcdp = n_steps * sensitivity**2 / (2.0 * noise_std**2)
    epsilon = rho_zcdp + 2.0 * np.sqrt(rho_zcdp * np.log(1.0 / delta))
    return float(epsilon), float(rho_zcdp)


def noise_std_for_gaussian_budget(
    n_steps: int,
    sensitivity: float,
    delta: float,
    target_epsilon: float,
) -> float:
    if target_epsilon <= 0.0:
        raise ValueError("target_epsilon must be positive")
    if n_steps <= 0 or sensitivity <= 0 or not 0.0 < delta < 1.0:
        raise ValueError("invalid privacy calibration inputs")
    log_term = np.log(1.0 / delta)
    sqrt_rho = -np.sqrt(log_term) + np.sqrt(log_term + target_epsilon)
    rho_zcdp = sqrt_rho**2
    return float(sensitivity * np.sqrt(n_steps / (2.0 * rho_zcdp)))


def zcdp_epsilon(
    n_rounds: int,
    clip_norm: float,
    noise_std: float,
    delta: float,
) -> tuple[float, float]:
    return gaussian_composition_epsilon(
        n_rounds, 2.0 * clip_norm, noise_std, delta
    )


def noise_std_for_target_epsilon(
    n_rounds: int,
    clip_norm: float,
    delta: float,
    target_epsilon: float,
) -> float:
    return noise_std_for_gaussian_budget(
        n_rounds, 2.0 * clip_norm, delta, target_epsilon
    )


def spade_private_schedule(
    n_rounds: int,
    clip_norm0: float,
    clip_exponent: float,
    delta: float,
    target_epsilon: float,
    utility_weights: np.ndarray | None = None,
    allocation: str = "theorem_weighted",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build SPADE-DFL ``{C_k}`` and ``{sigma_dp,k}`` sequences.

    With utility weights ``w_k``, the total zCDP budget is split with
    ``rho_k`` proportional to ``2 C_k sqrt(w_k)``.  This minimizes the
    weighted noise term ``sum_k w_k sigma_k^2`` for the fixed clipping
    sequence and total zCDP budget.  Unit weights recover minimum cumulative
    released-state noise variance.
    """
    if n_rounds <= 0 or clip_norm0 <= 0.0 or clip_exponent < 0.0:
        raise ValueError("invalid SPADE private schedule inputs")
    if target_epsilon <= 0.0 or not 0.0 < delta < 1.0:
        raise ValueError("invalid privacy target")
    if allocation not in {"theorem_weighted", "uniform_zcdp"}:
        raise ValueError("allocation must be theorem_weighted or uniform_zcdp")
    rounds = np.arange(1.0, n_rounds + 1.0, dtype=np.float64)
    clip_norms = clip_norm0 * rounds ** (-clip_exponent)
    sensitivities = 2.0 * clip_norms
    if utility_weights is None:
        weights = np.ones(n_rounds, dtype=np.float64)
    else:
        weights = np.asarray(utility_weights, dtype=np.float64)
        if weights.shape != (n_rounds,) or not np.all(np.isfinite(weights)):
            raise ValueError("utility_weights must be a finite n_rounds vector")
        if np.any(weights <= 0.0):
            raise ValueError("utility_weights must be positive")
    log_term = np.log(1.0 / delta)
    sqrt_rho = -np.sqrt(log_term) + np.sqrt(log_term + target_epsilon)
    rho_total = float(sqrt_rho**2)
    allocation_basis = (
        np.ones(n_rounds, dtype=np.float64)
        if allocation == "uniform_zcdp"
        else sensitivities * np.sqrt(weights)
    )
    rho_steps = rho_total * allocation_basis / allocation_basis.sum()
    noise_stds = sensitivities / np.sqrt(2.0 * rho_steps)
    return clip_norms, noise_stds, rho_steps
