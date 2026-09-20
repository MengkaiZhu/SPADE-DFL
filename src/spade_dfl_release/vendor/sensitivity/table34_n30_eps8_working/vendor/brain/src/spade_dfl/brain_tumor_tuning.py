from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TrainingHoldout:
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    train_indices: list[int]
    validation_indices: list[int]
    fraction: float
    seed: int


def make_training_holdout(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    fraction: float = 0.1,
    seed: int = 20260724,
) -> TrainingHoldout:
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("training features and labels have incompatible shapes")
    if not 0.0 < fraction < 0.5:
        raise ValueError("holdout fraction must lie in (0, 0.5)")
    rng = np.random.default_rng(seed)
    validation: list[int] = []
    training: list[int] = []
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        count = max(1, int(round(len(indices) * fraction)))
        validation.extend(int(index) for index in indices[:count])
        training.extend(int(index) for index in indices[count:])
    train_indices = sorted(training)
    validation_indices = sorted(validation)
    return TrainingHoldout(
        train_x=x[train_indices],
        train_y=y[train_indices],
        validation_x=x[validation_indices],
        validation_y=y[validation_indices],
        train_indices=train_indices,
        validation_indices=validation_indices,
        fraction=float(fraction),
        seed=int(seed),
    )


def brain_tumor_candidate_grid() -> list[dict[str, object]]:
    eta_center = 1.5 / 40.0
    rows: list[dict[str, object]] = []
    for multiplier in (0.25, 0.5, 1.0, 2.0):
        for rho in (0.25, 0.5, 1.0, 2.0):
            eta0 = float(multiplier * eta_center)
            rows.append(
                {
                    "candidate_id": f"eta_{eta0:.6f}_rho_{rho:g}_batch_1",
                    "eta0": eta0,
                    "rho": float(rho),
                    "batch_size": 1,
                    "n_rounds": 20,
                    "provenance": (
                        "eta0 multiplier around Reference [20] 1.5/d with d=40; "
                        "rho from the documented positive training-holdout grid; "
                        "batch=1 from Reference [13]"
                    ),
                }
            )
    return rows


def brain_tumor_batch_extension_grid() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"eta_0.009375_rho_1_batch_{batch_size}",
            "eta0": 0.009375,
            "rho": 1.0,
            "batch_size": batch_size,
            "n_rounds": 20,
            "provenance": (
                "Approved batch variance extension {4,8,16}; eta0 is the only "
                "finite reference-centered candidate and rho=1 is retained "
                "because all four rho grid values produced identical diagnostics"
            ),
        }
        for batch_size in (4, 8, 16)
    ]


def brain_tumor_horizon_extension_grid() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"eta_0.009375_rho_1_batch_{batch_size}_rounds_50",
            "eta0": 0.009375,
            "rho": 1.0,
            "batch_size": batch_size,
            "n_rounds": 50,
            "provenance": (
                "Approved 50-round horizon extension after the 20-round "
                "variance grid improved accuracy and loss but had not yet "
                "reduced the joint stationarity-consensus metric"
            ),
        }
        for batch_size in (4, 8, 16)
    ]


def brain_tumor_stability_extension_grid() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": "eta_0.0009375_rho_1_batch_16_rounds_50",
            "eta0": 0.0009375,
            "rho": 1.0,
            "batch_size": 16,
            "n_rounds": 50,
            "provenance": (
                "Single approved tenfold stability extension after the "
                "reference-centered, batch-variance, and horizon grids all "
                "failed the joint optimization acceptance criteria"
            ),
        }
    ]


def brain_tumor_fairness_extension_grid() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": (
                f"fair_eta_{eta0:.8f}_rho_1_batch_{batch_size}_rounds_50"
            ),
            "eta0": float(eta0),
            "rho": 1.0,
            "batch_size": int(batch_size),
            "n_rounds": 50,
            "provenance": (
                "Post-review symmetric stability grid around the previously "
                "accepted eta0=0.0009375 using multipliers {0.5,1,2} and "
                "geometric variance-reduction batches {16,32,64}; selected "
                "only on raw-training inner-PCA holdout with the original "
                "joint optimization acceptance criteria"
            ),
        }
        for eta0 in (0.00046875, 0.0009375, 0.001875)
        for batch_size in (16, 32, 64)
    ]


def brain_tumor_robust_extension_grid() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": (
                f"robust_eta_{eta0:.8f}_rho_1_batch_{batch_size}_rounds_50"
            ),
            "eta0": float(eta0),
            "rho": 1.0,
            "batch_size": int(batch_size),
            "n_rounds": 50,
            "provenance": (
                "Pre-registered geometric boundary extension from the prior "
                "Training-holdout optimum eta0=0.001875,batch=64; official "
                "Testing excluded from generation and selection"
            ),
        }
        for eta0 in (0.001875, 0.00375, 0.0075)
        for batch_size in (64, 128, 256)
    ]


def _eligible(row: Mapping[str, object]) -> bool:
    return bool(
        row["finite"]
        and float(row["accuracy_delta"]) > 0.0
        and float(row["loss_delta"]) < 0.0
        and float(row["joint_delta"]) <= 0.0
    )


def select_candidate(
    rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    eligible = [row for row in rows if _eligible(row)]
    if not eligible:
        raise ValueError("no candidate satisfies the frozen optimization criteria")
    return min(
        eligible,
        key=lambda row: (
            -float(row["final_validation_accuracy"]),
            float(row["query_count"]),
            float(row["eta0"]),
            float(row["rho"]),
            str(row["candidate_id"]),
        ),
    )


def select_robust_candidate(
    rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    eligible = [row for row in rows if bool(row["eligible"])]
    if not eligible:
        raise ValueError("no eligible robust candidate")
    return min(
        eligible,
        key=lambda row: (
            -float(row["minimum_panel_margin"]),
            -float(row["mean_panel_margin"]),
            -float(row["final_validation_accuracy"]),
            float(row["query_count"]),
            float(row["eta0"]),
            int(row["batch_size"]),
            str(row["candidate_id"]),
        ),
    )


def evaluate_candidate_safely(
    candidate: Mapping[str, object],
    evaluator: Callable[[Mapping[str, object]], Mapping[str, object]],
) -> dict[str, object]:
    try:
        row = dict(evaluator(candidate))
        row.setdefault("error_type", "")
        row.setdefault("error", "")
        return row
    except (FloatingPointError, OverflowError, ValueError) as error:
        row = dict(candidate)
        row.update(
            {
                "finite": False,
                "initial_validation_accuracy": np.nan,
                "final_validation_accuracy": np.nan,
                "initial_loss": np.nan,
                "final_loss": np.nan,
                "initial_joint": np.nan,
                "final_joint": np.nan,
                "query_count": 0,
                "per_panel_json": "[]",
                "accuracy_delta": np.nan,
                "loss_delta": np.nan,
                "joint_delta": np.nan,
                "eligible": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        return row
