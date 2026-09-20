from __future__ import annotations

import numpy as np


def consensus_error(states: np.ndarray) -> float:
    values = np.asarray(states, dtype=np.float64)
    mean = values.mean(axis=0)
    return float(np.mean(np.sum((values - mean) ** 2, axis=1)))


def penalty_energy(penalties: np.ndarray) -> float:
    values = np.asarray(penalties, dtype=np.float64)
    return float(np.mean(np.sum(values**2, axis=1)))


def paper_stationarity_consensus(
    stationarity: np.ndarray,
    consensus: np.ndarray,
) -> float:
    """Compute SPADE-DFL Eq. (32) over communication rounds 0,...,K-1."""
    stationary_values = np.asarray(stationarity, dtype=np.float64)
    consensus_values = np.asarray(consensus, dtype=np.float64)
    if stationary_values.ndim != 1 or consensus_values.ndim != 1:
        raise ValueError("stationarity and consensus histories must be vectors")
    if stationary_values.shape != consensus_values.shape:
        raise ValueError("stationarity and consensus histories must have equal length")
    if stationary_values.size < 2:
        raise ValueError("history must include round zero and at least one update")
    values = stationary_values[:-1] + consensus_values[:-1]
    if not np.all(np.isfinite(values)):
        raise ValueError("stationarity-consensus history must be finite")
    return float(values.mean())
