"""Read-only paired evidence and statistics for the MNIST 6/7 paper suite."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy import integrate, stats

from .mnist67_paper_protocol import METHOD_IDS, PARTITION_IDS
from .mnist67_paper_runner import validate_result_root


@dataclass(frozen=True)
class PaperEvidence:
    result_root: Path
    seeds: tuple[int, ...]
    rounds: np.ndarray
    accuracy: Mapping[tuple[str, str], np.ndarray]
    label_counts: Mapping[str, np.ndarray]
    inference_status: str


@dataclass(frozen=True)
class CurveInterval:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    p_value: float


@dataclass(frozen=True)
class AulcComparison:
    partition: str
    baseline_method: str
    paired_difference_pp: tuple[float, ...]
    mean_difference_pp: float
    wilcoxon_statistic: float
    raw_p_value: float
    adjusted_q_value: float
    minus_log10_q: float
    classification: str


def load_paper_evidence(result_root: Path, *, formal: bool) -> PaperEvidence:
    """Load raw histories and SPADE partition counts after artifact validation."""

    root = Path(result_root).resolve()
    report = validate_result_root(root, formal=formal)
    manifest = _read_json(root / "suite_manifest.json")
    suite = manifest["suite_config"]
    runs = manifest["runs"]
    seeds = tuple(int(seed) for seed in suite["seeds"])
    by_identity = {
        (str(row["partition"]), str(row["method"]), int(row["seed"])): row
        for row in runs
    }
    accuracy: dict[tuple[str, str], np.ndarray] = {}
    canonical_rounds: np.ndarray | None = None
    for partition in PARTITION_IDS:
        for method in METHOD_IDS:
            curves = []
            for seed in seeds:
                run_dir = _resolve_run_dir(
                    root, by_identity[(partition, method, seed)]["run_dir"]
                )
                rounds, values = _read_accuracy(run_dir / "history.csv")
                if canonical_rounds is None:
                    canonical_rounds = rounds
                elif not np.array_equal(canonical_rounds, rounds):
                    raise ValueError("saved round axes are inconsistent")
                curves.append(values)
            accuracy[(partition, method)] = _freeze(np.vstack(curves))
    label_counts: dict[str, np.ndarray] = {}
    for partition in PARTITION_IDS:
        seed_counts = []
        for seed in seeds:
            run_dir = _resolve_run_dir(
                root,
                by_identity[(partition, "spade_dfl_private", seed)]["run_dir"],
            )
            environment = _read_json(run_dir / "environment.json")
            dataset = environment.get("dataset")
            if not isinstance(dataset, dict):
                raise ValueError("run environment is missing dataset metadata")
            counts = np.asarray(dataset.get("label_histograms"), dtype=np.int64)
            if counts.shape != (1000, 2) or np.any(counts < 0):
                raise ValueError(
                    f"SPADE label histograms for {partition}/seed_{seed} "
                    "must have shape (1000, 2)"
                )
            normalized_binary_entropy(counts)
            seed_counts.append(counts)
        label_counts[partition] = _freeze(np.stack(seed_counts, axis=0))
    if canonical_rounds is None:
        raise ValueError("no accuracy histories were loaded")
    return PaperEvidence(
        result_root=root,
        seeds=seeds,
        rounds=_freeze(canonical_rounds),
        accuracy=MappingProxyType(accuracy),
        label_counts=MappingProxyType(label_counts),
        inference_status=report.inference_status,
    )


def mean_and_95_t_ci(curves: np.ndarray) -> CurveInterval:
    """Return the seed mean and two-sided 95% Student-t interval."""

    values = np.asarray(curves, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("curves must be a finite seed-by-round matrix with at least two seeds")
    mean = values.mean(axis=0)
    standard_error = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    half_width = float(stats.t.ppf(0.975, values.shape[0] - 1)) * standard_error
    return CurveInterval(
        mean=_freeze(mean),
        lower=_freeze(mean - half_width),
        upper=_freeze(mean + half_width),
    )


def normalized_aulc(rounds: np.ndarray, curves: np.ndarray) -> np.ndarray:
    """Return trapezoidal area divided by the observed round horizon."""

    x = np.asarray(rounds, dtype=np.float64)
    y = np.asarray(curves, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("rounds must be a finite vector with at least two values")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("rounds must be strictly increasing")
    if y.ndim != 2 or y.shape[1] != x.size or not np.all(np.isfinite(y)):
        raise ValueError("curves must be a finite seed-by-round matrix")
    horizon = float(x[-1] - x[0])
    if horizon <= 0.0:
        raise ValueError("round horizon must be positive")
    return np.asarray(integrate.trapezoid(y, x=x, axis=1) / horizon, dtype=np.float64)


def paired_wilcoxon(differences: np.ndarray) -> WilcoxonResult:
    """Run the registered two-sided paired signed-rank test."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("paired differences must be a nonempty finite vector")
    if np.all(values == 0.0):
        return WilcoxonResult(statistic=0.0, p_value=1.0)
    result = stats.wilcoxon(
        values,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )
    return WilcoxonResult(
        statistic=float(result.statistic), p_value=float(result.pvalue)
    )


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Adjust one complete family of p-values with the BH step-up rule."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a nonempty vector")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must lie in [0,1]")
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    factors = values.size / np.arange(1, values.size + 1, dtype=np.float64)
    adjusted_ranked = np.minimum.accumulate((ranked * factors)[::-1])[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def build_aulc_comparisons(
    evidence: PaperEvidence,
) -> tuple[AulcComparison, ...]:
    """Build exactly six baselines by six partitions using paired seeds."""

    if evidence.seeds != tuple(range(30)):
        raise ValueError("formal AULC comparisons require seeds 0 through 29")
    provisional: list[AulcComparison] = []
    raw_p_values = []
    for partition in PARTITION_IDS:
        spade = np.asarray(
            evidence.accuracy[(partition, "spade_dfl_private")], dtype=np.float64
        )
        spade_aulc = normalized_aulc(evidence.rounds, spade * 100.0)
        for baseline in METHOD_IDS[:-1]:
            baseline_aulc = normalized_aulc(
                evidence.rounds,
                np.asarray(evidence.accuracy[(partition, baseline)]) * 100.0,
            )
            differences = spade_aulc - baseline_aulc
            test = paired_wilcoxon(differences)
            raw_p_values.append(test.p_value)
            provisional.append(
                AulcComparison(
                    partition=partition,
                    baseline_method=baseline,
                    paired_difference_pp=tuple(float(value) for value in differences),
                    mean_difference_pp=float(np.mean(differences)),
                    wilcoxon_statistic=test.statistic,
                    raw_p_value=test.p_value,
                    adjusted_q_value=float("nan"),
                    minus_log10_q=float("nan"),
                    classification="pending",
                )
            )
    if len(provisional) != 36:
        raise AssertionError("AULC comparison family must contain exactly 36 records")
    adjusted = benjamini_hochberg(np.asarray(raw_p_values, dtype=np.float64))
    completed = []
    for record, q_value in zip(provisional, adjusted):
        if q_value >= 0.05:
            classification = "unresolved"
        elif record.mean_difference_pp > 0.0:
            classification = "better"
        elif record.mean_difference_pp < 0.0:
            classification = "worse"
        else:
            classification = "unresolved"
        completed.append(
            replace(
                record,
                adjusted_q_value=float(q_value),
                minus_log10_q=float(
                    -np.log10(max(float(q_value), np.finfo(np.float64).tiny))
                ),
                classification=classification,
            )
        )
    return tuple(completed)


def normalized_binary_entropy(label_counts: np.ndarray) -> np.ndarray:
    """Return normalized Shannon entropy for two-class client counts."""

    counts = np.asarray(label_counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != 2:
        raise ValueError("label_counts must have shape (clients, 2)")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("label counts must be finite and nonnegative")
    totals = counts.sum(axis=1)
    if np.any(totals <= 0.0):
        raise ValueError("label counts contain empty clients")
    probabilities = counts / totals[:, None]
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    terms[positive] = probabilities[positive] * np.log2(probabilities[positive])
    return -terms.sum(axis=1)


def _read_accuracy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        rounds = np.asarray([float(row["iteration"]) for row in rows], dtype=np.float64)
        accuracy = np.asarray([float(row["accuracy"]) for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"malformed accuracy history: {path}") from error
    if not np.all(np.isfinite(accuracy)) or np.any((accuracy < 0.0) | (accuracy > 1.0)):
        raise ValueError(f"invalid accuracy history: {path}")
    return rounds, accuracy


def _resolve_run_dir(root: Path, raw: object) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("run_dir must be a nonempty path")
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _freeze(values: np.ndarray) -> np.ndarray:
    frozen = np.asarray(values).copy()
    frozen.setflags(write=False)
    return frozen
