"""Method-comparison volcano for the 30-seed MNIST 6/7 black-box suite."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from .mnist67_paper_evidence import benjamini_hochberg, paired_wilcoxon
from .mnist67_spade_distribution import _plot_context, _save, _sidecar


METHOD_ORDER = (
    "zo2_lt_admm_vr",
    "zo2_lt_admm",
    "zo2_dp_sgd",
    "zo2_dp_fedavg",
    "one_point_dsg",
    "one_point_dsgt",
    "spade_dfl",
)
PARTITION_ORDER = ("iid", "dir_0.6", "dir_0.3", "pathological")
METHOD_LABELS = {
    "zo2_lt_admm_vr": "LT-ADMM-VR",
    "zo2_lt_admm": "LT-ADMM",
    "zo2_dp_sgd": "DP-SGD",
    "zo2_dp_fedavg": "DP-FedAvg",
    "one_point_dsg": "1P-DSG",
    "one_point_dsgt": "1P-DSGT",
    "spade_dfl": "SPADE-DFL",
}
PARTITION_LABELS = {
    "iid": "IID",
    "dir_0.6": "Dir. 0.6",
    "dir_0.3": "Dir. 0.3",
    "pathological": "Pathological",
}


@dataclass(frozen=True)
class MethodVolcanoRecord:
    method: str
    partition: str
    round: int
    mean_paired_accuracy_difference_pp: float
    paired_differences_pp: tuple[float, ...]
    wilcoxon_statistic: float
    raw_p_value: float
    adjusted_q_value: float
    minus_log10_q: float
    classification: str


def load_server_accuracy(
    result_root: Path,
) -> tuple[Mapping[tuple[str, str], np.ndarray], tuple[int, ...]]:
    """Load the exact 840 saved histories after checking the formal suite identity."""

    root = Path(result_root).resolve()
    manifest = _read_json(root / "suite_manifest.json")
    suite = manifest.get("suite_config")
    runs = manifest.get("runs")
    if not isinstance(suite, dict) or not isinstance(runs, list):
        raise ValueError("server manifest is missing suite_config or runs")
    seeds = _validate_suite_identity(suite, run_count=len(runs))
    identities = {
        (str(row["partition"]), str(row["method"]), int(row["seed"])): row
        for row in runs
        if isinstance(row, dict)
    }
    if len(identities) != 840:
        raise ValueError("server manifest has missing or duplicate run identities")
    accuracy: dict[tuple[str, str], np.ndarray] = {}
    for partition in PARTITION_ORDER:
        for method in METHOD_ORDER:
            curves = []
            for seed in seeds:
                record = identities[(partition, method, seed)]
                run_dir = _resolve_run_dir(root, str(record["run_dir"]))
                curves.append(_read_accuracy(run_dir / "history.csv"))
            accuracy[(partition, method)] = _freeze(np.vstack(curves))
    return MappingProxyType(accuracy), seeds


def _validate_suite_identity(
    suite: Mapping[str, object], *, run_count: int
) -> tuple[int, ...]:
    """Validate identities while allowing semantically irrelevant storage order."""

    if tuple(suite.get("methods", ())) != METHOD_ORDER:
        raise ValueError("server methods do not match the seven-method black-box protocol")
    declared_partitions = tuple(str(value) for value in suite.get("partitions", ()))
    if len(declared_partitions) != len(PARTITION_ORDER) or set(declared_partitions) != set(
        PARTITION_ORDER
    ):
        raise ValueError("server partitions do not match the four-distribution protocol")
    seeds = tuple(int(seed) for seed in suite.get("seeds", ()))
    if seeds != tuple(range(30)):
        raise ValueError("method volcano requires paired seeds 0 through 29")
    if run_count != 840:
        raise ValueError("server manifest must contain 4 × 7 × 30 = 840 runs")
    return seeds


def build_method_volcano_records(
    accuracy: Mapping[tuple[str, str], np.ndarray], seeds: tuple[int, ...]
) -> tuple[MethodVolcanoRecord, ...]:
    """Build one globally adjusted family of 1200 paired roundwise comparisons."""

    if seeds != tuple(range(30)):
        raise ValueError("formal method volcano requires paired seeds 0 through 29")
    _validate_accuracy_mapping(accuracy, len(seeds))
    provisional: list[MethodVolcanoRecord] = []
    p_values: list[float] = []
    for partition in PARTITION_ORDER:
        reference = np.asarray(accuracy[(partition, "zo2_lt_admm_vr")], dtype=float)
        for method in METHOD_ORDER[1:]:
            curves = np.asarray(accuracy[(partition, method)], dtype=float)
            paired = curves - reference
            for round_ in range(1, 51):
                differences = np.asarray(paired[:, round_], dtype=float)
                test = paired_wilcoxon(differences)
                p_values.append(test.p_value)
                provisional.append(
                    MethodVolcanoRecord(
                        method=method,
                        partition=partition,
                        round=round_,
                        mean_paired_accuracy_difference_pp=float(np.mean(differences)),
                        paired_differences_pp=tuple(float(value) for value in differences),
                        wilcoxon_statistic=test.statistic,
                        raw_p_value=test.p_value,
                        adjusted_q_value=float("nan"),
                        minus_log10_q=float("nan"),
                        classification="pending",
                    )
                )
    if len(provisional) != 1200:
        raise AssertionError("method volcano must contain 6 × 4 × 50 comparisons")
    adjusted = benjamini_hochberg(np.asarray(p_values, dtype=float))
    completed = []
    for record, q_value in zip(provisional, adjusted):
        if q_value >= 0.05 or record.mean_paired_accuracy_difference_pp == 0.0:
            classification = "unresolved"
        elif record.mean_paired_accuracy_difference_pp > 0.0:
            classification = "better"
        else:
            classification = "worse"
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


def plot_method_volcano(
    accuracy: Mapping[tuple[str, str], np.ndarray],
    seeds: tuple[int, ...],
    output_base: Path,
) -> tuple[Path, Path, Path, Path]:
    """Render a screenshot-like, statistically defined method volcano."""

    records = build_method_volcano_records(accuracy, seeds)
    categories = (
        ("worse", "Worse", "#3561cc"),
        ("unresolved", "Not resolved", "#bdbdbd"),
        ("better", "Better", "#ef3b35"),
    )
    with _plot_context():
        figure, axis = plt.subplots(figsize=(3.5, 2.65))
        axis.axvline(0.0, color="#666666", linestyle="--", linewidth=0.7, zorder=1)
        axis.axhline(
            -np.log10(0.05), color="#666666", linestyle="--", linewidth=0.7, zorder=1
        )
        for classification, label, color in categories:
            selected = [record for record in records if record.classification == classification]
            axis.scatter(
                [record.mean_paired_accuracy_difference_pp for record in selected],
                [record.minus_log10_q for record in selected],
                s=8.5,
                color=color,
                alpha=0.55 if classification != "unresolved" else 0.42,
                linewidths=0,
                label=label,
                zorder=2,
            )
        point_y_max = max(record.minus_log10_q for record in records)
        axis.set_ylim(bottom=-0.35, top=point_y_max + 2.15)
        _annotate(axis, records, point_y_max)
        axis.set_xlabel("Accuracy change vs. LT-ADMM-VR (percentage points)")
        axis.set_ylabel("−log10(BH-adjusted p-value)")
        axis.grid(color="#dddddd", linewidth=0.4, alpha=0.48)
        axis.legend(
            loc="upper center",
            ncol=3,
            fontsize=5.6,
            handletextpad=0.35,
            bbox_to_anchor=(0.5, 1.005),
        )
        axis.text(
            0.98,
            0.035,
            "n=30 paired seeds; 1200 comparisons",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.1,
            color="#555555",
        )
        figure.subplots_adjust(left=0.19, right=0.99, bottom=0.21, top=0.98)
        images = _save(figure, Path(output_base))
    counts = {
        classification: sum(record.classification == classification for record in records)
        for classification in ("worse", "unresolved", "better")
    }
    return (
        *images,
        _sidecar(
            Path(output_base),
            {
                "figure": "MNIST 6/7 method-comparison volcano",
                "dataset": "MNIST 6/7",
                "seed_count": len(seeds),
                "comparison_count": len(records),
                "reference_method": "LT-ADMM-VR",
                "test": "two-sided paired Wilcoxon signed-rank test",
                "correction": "Benjamini-Hochberg across all 1200 method-partition-round comparisons",
                "significance_threshold": 0.05,
                "classification_counts": counts,
                "source_data": [asdict(record) for record in records],
            },
        ),
    )


def plot_method_volcano_from_result_root(
    result_root: Path, output_base: Path
) -> tuple[Path, Path, Path, Path]:
    accuracy, seeds = load_server_accuracy(result_root)
    return plot_method_volcano(accuracy, seeds, output_base)


def _annotate(
    axis: plt.Axes,
    records: tuple[MethodVolcanoRecord, ...],
    point_y_max: float,
) -> None:
    requests = (
        ("spade_dfl", "iid"),
        ("spade_dfl", "dir_0.6"),
        ("spade_dfl", "dir_0.3"),
        ("spade_dfl", "pathological"),
        ("one_point_dsg", "dir_0.3"),
        ("zo2_dp_sgd", "dir_0.3"),
    )
    chosen = []
    for method, partition in requests:
        subset = [
            record
            for record in records
            if record.method == method
            and record.partition == partition
            and record.classification != "unresolved"
        ]
        if subset:
            chosen.append(max(subset, key=lambda record: record.minus_log10_q))

    label_specs = {
        ("spade_dfl", "pathological"): (0.68, 0.82, "left", 0.10),
        ("spade_dfl", "dir_0.3"): (0.78, 0.77, "left", 0.06),
        ("spade_dfl", "dir_0.6"): (0.88, 0.72, "left", 0.03),
        ("spade_dfl", "iid"): (0.98, 0.67, "right", -0.05),
        ("one_point_dsg", "dir_0.3"): (0.15, 0.76, "left", -0.08),
        ("zo2_dp_sgd", "dir_0.3"): (0.15, 0.70, "left", -0.05),
    }

    for record in chosen:
        key = (record.method, record.partition)
        if key not in label_specs:
            continue
        text_x, text_y, align, rad = label_specs[key]
        is_left = record.mean_paired_accuracy_difference_pp < 0.0
        color = "#3561cc" if is_left else "#ef3b35"
        axis.annotate(
            f"{METHOD_LABELS[record.method]}, {PARTITION_LABELS[record.partition]}",
            xy=(record.mean_paired_accuracy_difference_pp, record.minus_log10_q),
            xytext=(text_x, text_y),
            xycoords="data",
            textcoords="axes fraction",
            ha=align,
            va="center",
            fontsize=4.5,
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.45,
                "alpha": 0.82,
                "shrinkA": 2.0,
                "shrinkB": 2.0,
                "connectionstyle": f"arc3,rad={rad}",
            },
        )

def _validate_accuracy_mapping(
    accuracy: Mapping[tuple[str, str], np.ndarray], seed_count: int
) -> None:
    for partition in PARTITION_ORDER:
        for method in METHOD_ORDER:
            key = (partition, method)
            if key not in accuracy:
                raise ValueError(f"missing accuracy matrix for {key}")
            values = np.asarray(accuracy[key], dtype=float)
            if values.shape != (seed_count, 51) or not np.all(np.isfinite(values)):
                raise ValueError(f"accuracy matrix for {key} must have shape ({seed_count}, 51)")


def _resolve_run_dir(root: Path, declared: str) -> Path:
    path = Path(declared)
    parts = path.parts
    try:
        index = parts.index("runs")
    except ValueError as error:
        raise ValueError(f"run_dir lacks a runs anchor: {declared}") from error
    candidate = (root / Path(*parts[index:])).resolve()
    try:
        candidate.relative_to((root / "runs").resolve())
    except ValueError as error:
        raise ValueError(f"run_dir escapes extracted result root: {declared}") from error
    return candidate


def _read_accuracy(path: Path) -> np.ndarray:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    try:
        round_key = "round" if rows and "round" in rows[0] else "iteration"
        rounds = np.asarray([int(row[round_key]) for row in rows], dtype=int)
        values = np.asarray([float(row["accuracy"]) * 100.0 for row in rows], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"malformed history: {path}") from error
    if not np.array_equal(rounds, np.arange(51)) or not np.all(np.isfinite(values)):
        raise ValueError(f"history must contain rounds 0 through 50: {path}")
    return values


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _freeze(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result.setflags(write=False)
    return result
