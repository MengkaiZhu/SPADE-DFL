"""Three honest preview figures for the frozen strong-DP MNIST 6/7 SPADE study."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .distribution_study import DistributionStudy
from .mnist67_paper_evidence import benjamini_hochberg, paired_wilcoxon

plt.switch_backend("Agg")

PARTITION_ORDER = (
    "iid",
    "dir_0.6",
    "dir_0.3",
    "pathological",
    "dir_0.1",
)
HETEROGENEITY_ORDER = (
    "iid",
    "pathological",
    "dir_0.6",
    "dir_0.3",
    "dir_0.1",
)
LABELS = {
    "iid": "IID",
    "dir_0.6": "Dir. 0.6",
    "dir_0.3": "Dir. 0.3",
    "pathological": "Mixed Path.",
    "dir_0.1": "Dir. 0.1",
}
COLORS = {
    "iid": "#3182bd",
    "dir_0.6": "#6baed6",
    "dir_0.3": "#ffb74d",
    "pathological": "#ef5350",
    "dir_0.1": "#756bb1",
}
MARKERS = {
    "iid": "o",
    "dir_0.6": "s",
    "dir_0.3": "^",
    "pathological": "D",
    "dir_0.1": "P",
}
LINESTYLES = {
    "iid": "-",
    "dir_0.6": "--",
    "dir_0.3": "-.",
    "pathological": ":",
    "dir_0.1": (0, (5, 1.4)),
}


@dataclass(frozen=True)
class DistributionRoundVolcanoRecord:
    """One paired partition-versus-IID comparison at one communication round."""

    partition: str
    round: int
    mean_paired_accuracy_difference_pp: float
    paired_differences_pp: tuple[float, ...]
    wilcoxon_statistic: float
    raw_p_value: float
    adjusted_q_value: float
    minus_log10_q: float
    classification: str


def plot_accuracy_with_iid_inset(
    study: DistributionStudy, output_base: Path
) -> tuple[Path, Path, Path, Path]:
    """Plot five strong-DP SPADE curves and paired differences from IID."""

    _validate(study)
    rounds = np.arange(_round_count(study), dtype=float)
    iid = np.asarray(study.accuracy_by_seed["iid"], dtype=float) * 100.0
    source: dict[str, object] = {}
    visible_lower: list[float] = []
    visible_upper: list[float] = []
    with _plot_context():
        figure, axis = plt.subplots(figsize=(3.5, 2.65))
        # Figure-relative placement keeps the inset registered as a real second
        # axes in exported/vector files and matches the approved IEEE-column layout.
        inset = figure.add_axes([0.28, 0.55, 0.38, 0.29])
        for index, partition in enumerate(PARTITION_ORDER):
            curves = np.asarray(study.accuracy_by_seed[partition], dtype=float) * 100.0
            mean, lower, upper = _mean_t_ci(curves)
            visible_lower.append(float(np.min(lower[1:])))
            visible_upper.append(float(np.max(upper[1:])))
            axis.fill_between(
                rounds, lower, upper, color=COLORS[partition], alpha=0.05, linewidth=0
            )
            axis.plot(
                rounds,
                mean,
                color=COLORS[partition],
                linestyle=LINESTYLES[partition],
                marker=MARKERS[partition],
                markevery=(index, 10),
                markersize=2.8,
                linewidth=1.10,
                label=LABELS[partition],
            )
            paired = curves - iid
            if partition != "iid":
                inset.plot(
                    rounds,
                    paired.mean(axis=0),
                    color=COLORS[partition],
                    linestyle=LINESTYLES[partition],
                    linewidth=0.80,
                )
            source[partition] = {
                "accuracy_by_seed_percent": curves.tolist(),
                "mean_percent": mean.tolist(),
                "confidence_interval_95_lower_percent": lower.tolist(),
                "confidence_interval_95_upper_percent": upper.tolist(),
                "paired_difference_vs_iid_pp": paired.tolist(),
            }
        y_min = max(0.0, float(np.floor(min(visible_lower) - 0.5)))
        y_max = min(100.0, float(np.ceil(max(visible_upper) + 0.5)))
        axis.set_xlim(0, int(rounds[-1]))
        axis.set_ylim(y_min, max(y_min + 2.0, y_max))
        axis.set_xlabel("Rounds")
        axis.set_ylabel("Accuracy (%)")
        axis.grid(color="#dddddd", linewidth=0.4, alpha=0.55)
        axis.legend(
            loc="lower right",
            ncol=2,
            fontsize=5.0,
            title="mean ± 95% CI",
            title_fontsize=5.3,
        )
        inset.axhline(0.0, color="#666666", linestyle="--", linewidth=0.6)
        inset.set_xlim(0, int(rounds[-1]))
        inset.set_title("Difference vs. IID (pp)", fontsize=5.5)
        inset.tick_params(labelsize=4.9)
        inset.grid(color="#e5e5e5", linewidth=0.3, alpha=0.45)
        figure.subplots_adjust(left=0.16, right=0.99, bottom=0.18, top=0.98)
        images = _save(figure, Path(output_base))
    return (
        *images,
        _sidecar(
            Path(output_base),
            {
                "figure": "SPADE-DFL accuracy across five MNIST 6/7 client distributions",
                "dataset": "MNIST 6/7",
                "method": "SPADE-DFL",
                "privacy": {"epsilon": 8.0, "delta": 1e-5},
                "seed_count": len(study.seeds),
                "seeds": list(study.seeds),
                "partition_order": list(PARTITION_ORDER),
                "uncertainty": "two-sided 95% Student-t confidence interval",
                "curve_processing": "none",
                "inset": "paired accuracy difference versus IID in percentage points",
                "source_data": source,
            },
        ),
    )


def plot_label_heterogeneity(
    study: DistributionStudy, output_base: Path
) -> tuple[Path, Path, Path, Path]:
    """Plot the empirical client-level positive-label fractions."""

    _validate(study)
    empirical_order = empirical_heterogeneity_order(study)
    mean_label_skew = {
        partition: float(
            1.0 - np.mean(np.asarray(study.label_entropy[partition], dtype=float))
        )
        for partition in PARTITION_ORDER
    }
    rng = np.random.default_rng(20260827)
    with _plot_context():
        figure, axis = plt.subplots(figsize=(3.5, 2.65))
        positions = np.arange(1, len(empirical_order) + 1)
        values = [np.asarray(study.label_proportions[p], dtype=float) for p in empirical_order]
        boxes = axis.boxplot(
            values,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#333333", "linewidth": 0.9},
            whiskerprops={"color": "#777777", "linewidth": 0.65},
            capprops={"color": "#777777", "linewidth": 0.65},
        )
        for patch, partition in zip(boxes["boxes"], empirical_order):
            patch.set_facecolor(COLORS[partition])
            patch.set_alpha(0.22)
            patch.set_edgecolor(COLORS[partition])
        for position, partition, observed in zip(positions, empirical_order, values):
            select = observed if observed.size <= 1400 else rng.choice(observed, 1400, replace=False)
            axis.scatter(
                position + rng.uniform(-0.18, 0.18, select.size),
                select,
                s=3.2,
                color=COLORS[partition],
                alpha=0.13,
                linewidths=0,
            )
        axis.set_xticks(positions, [LABELS[p] for p in empirical_order], rotation=28, ha="right")
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel("Client p(y=7)")
        axis.set_xlabel("Client-data distribution")
        axis.grid(axis="y", color="#dddddd", linewidth=0.4, alpha=0.55)
        figure.subplots_adjust(left=0.17, right=0.99, bottom=0.33, top=0.98)
        images = _save(figure, Path(output_base))
    payload = {
        partition: {
            "p_y7": np.asarray(study.label_proportions[partition], dtype=float).tolist(),
            "mean_binary_entropy": float(np.mean(study.label_entropy[partition])),
        }
        for partition in PARTITION_ORDER
    }
    return (
        *images,
        _sidecar(
            Path(output_base),
            {
                "figure": "Client label heterogeneity",
                "dataset": "MNIST 6/7",
                "partition_order": list(empirical_order),
                "ordering_rule": (
                    "author-approved low-to-high order verified by ascending empirical "
                    "mean label skew (1 - binary entropy)"
                ),
                "mean_label_skew": mean_label_skew,
                "descriptive_only": True,
                "source_data": payload,
            },
        ),
    )


def empirical_heterogeneity_order(study: DistributionStudy) -> tuple[str, ...]:
    """Order partitions by observed mean binary-label skew, low to high."""

    _validate(study)
    original_position = {partition: index for index, partition in enumerate(PARTITION_ORDER)}
    scores = {
        partition: float(
            1.0 - np.mean(np.asarray(study.label_entropy[partition], dtype=float))
        )
        for partition in PARTITION_ORDER
    }
    measured_order = tuple(
        sorted(PARTITION_ORDER, key=lambda p: (scores[p], original_position[p]))
    )
    if measured_order != HETEROGENEITY_ORDER:
        raise ValueError(
            "measured client-label skew contradicts the approved low-to-high "
            f"heterogeneity order: {measured_order}"
        )
    return HETEROGENEITY_ORDER


def build_roundwise_volcano_records(
    study: DistributionStudy,
) -> tuple[DistributionRoundVolcanoRecord, ...]:
    """Compare every non-IID round with IID using paired seeds and one BH family."""

    _validate(study)
    if len(study.seeds) < 10:
        raise ValueError("the exploratory volcano requires at least 10 paired seeds")
    iid = np.asarray(study.accuracy_by_seed["iid"], dtype=float) * 100.0
    provisional: list[DistributionRoundVolcanoRecord] = []
    raw_p_values: list[float] = []
    for partition in PARTITION_ORDER[1:]:
        curves = np.asarray(study.accuracy_by_seed[partition], dtype=float) * 100.0
        paired = curves - iid
        for round_ in range(1, paired.shape[1]):
            differences = np.asarray(paired[:, round_], dtype=float)
            test = paired_wilcoxon(differences)
            raw_p_values.append(test.p_value)
            provisional.append(
                DistributionRoundVolcanoRecord(
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
    expected_comparisons = (len(PARTITION_ORDER) - 1) * 50
    if len(provisional) != expected_comparisons:
        raise AssertionError(
            f"volcano family must contain {len(PARTITION_ORDER) - 1} partitions × 50 rounds"
        )
    adjusted = benjamini_hochberg(np.asarray(raw_p_values, dtype=float))
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


def plot_distribution_volcano(
    study: DistributionStudy, output_base: Path
) -> tuple[Path, Path, Path, Path]:
    """Plot effect size against BH-adjusted paired-test evidence."""

    records = build_roundwise_volcano_records(study)
    categories = (
        ("worse", "Worse", "#3561cc"),
        ("unresolved", "Not resolved", "#bdbdbd"),
        ("better", "Better", "#ef3b35"),
    )
    with _plot_context():
        figure, axis = plt.subplots(figsize=(3.5, 2.65))
        axis.axvline(0.0, color="#666666", linestyle="--", linewidth=0.75, zorder=1)
        axis.axhline(
            -np.log10(0.05), color="#666666", linestyle="--", linewidth=0.75, zorder=1
        )
        for classification, label, color in categories:
            selected = [record for record in records if record.classification == classification]
            axis.scatter(
                [record.mean_paired_accuracy_difference_pp for record in selected],
                [record.minus_log10_q for record in selected],
                s=12,
                color=color,
                alpha=0.62 if classification != "unresolved" else 0.45,
                linewidths=0,
                label=label,
                zorder=2,
            )
        _annotate_volcano(axis, records)
        axis.set_xlabel("Accuracy change vs. IID (percentage points)")
        axis.set_ylabel("−log10(BH-adjusted p-value)")
        axis.grid(color="#dddddd", linewidth=0.4, alpha=0.48)
        axis.legend(loc="upper center", ncol=3, fontsize=5.6, handletextpad=0.35)
        axis.text(
            0.98,
            0.04,
            f"n={len(study.seeds)} paired seeds; {len(records)} comparisons",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.2,
            color="#555555",
        )
        figure.subplots_adjust(left=0.19, right=0.99, bottom=0.20, top=0.98)
        images = _save(figure, Path(output_base))
    return (
        *images,
        _sidecar(
            Path(output_base),
            {
                "figure": "SPADE-DFL distribution-effect volcano",
                "dataset": "MNIST 6/7",
                "reference_distribution": "IID",
                "formal_inference": False,
                "seed_count": len(study.seeds),
                "comparison_count": len(records),
                "test": "two-sided paired Wilcoxon signed-rank test",
                "correction": (
                    f"Benjamini-Hochberg across all {len(records)} "
                    "partition-round comparisons"
                ),
                "significance_threshold": 0.05,
                "warning": "Exploratory small-scale result; use 30 paired seeds for formal claims.",
                "source_data": [asdict(record) for record in records],
            },
        ),
    )


def _annotate_volcano(axis: plt.Axes, records: tuple[DistributionRoundVolcanoRecord, ...]) -> None:
    """Label a sparse set of extremes with offsets that avoid covering points."""

    candidates: list[DistributionRoundVolcanoRecord] = []
    for classification in ("worse", "better"):
        subset = [record for record in records if record.classification == classification]
        subset.sort(
            key=lambda record: (record.minus_log10_q, abs(record.mean_paired_accuracy_difference_pp)),
            reverse=True,
        )
        used_partitions: set[str] = set()
        for record in subset:
            if record.partition not in used_partitions:
                candidates.append(record)
                used_partitions.add(record.partition)
            if len(used_partitions) == 3:
                break
    for index, record in enumerate(candidates):
        is_left = record.mean_paired_accuracy_difference_pp < 0.0
        x_offset = 10 if is_left else -10
        y_offset = 8 if index % 2 == 0 else -11
        color = "#3561cc" if is_left else "#ef3b35"
        axis.annotate(
            f"{LABELS[record.partition]}, r={record.round}",
            xy=(record.mean_paired_accuracy_difference_pp, record.minus_log10_q),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="left" if is_left else "right",
            va="bottom" if y_offset > 0 else "top",
            fontsize=4.8,
            color="#333333",
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.55},
        )


def build_three_figure_preview(
    study: DistributionStudy, output_directory: Path
) -> Mapping[str, tuple[Path, Path, Path, Path]]:
    """Create the exact three-figure local-review bundle."""

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "accuracy": plot_accuracy_with_iid_inset(study, root / "mnist67_distribution_accuracy"),
        "heterogeneity": plot_label_heterogeneity(study, root / "mnist67_client_label_heterogeneity"),
        "volcano": plot_distribution_volcano(study, root / "mnist67_distribution_volcano"),
    }


def _mean_t_ci(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = curves.mean(axis=0)
    standard_error = curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])
    half_width = float(stats.t.ppf(0.975, curves.shape[0] - 1)) * standard_error
    return mean, mean - half_width, mean + half_width


def _round_count(study: DistributionStudy) -> int:
    return int(np.asarray(study.accuracy_by_seed["iid"]).shape[1])


def _validate(study: DistributionStudy) -> None:
    if tuple(study.partitions) != PARTITION_ORDER:
        raise ValueError(f"study partitions must be ordered as {PARTITION_ORDER}")
    if len(study.seeds) < 2:
        raise ValueError("at least two paired seeds are required")
    shape: tuple[int, int] | None = None
    for partition in PARTITION_ORDER:
        curves = np.asarray(study.accuracy_by_seed[partition], dtype=float)
        if curves.ndim != 2 or curves.shape[0] != len(study.seeds) or not np.all(np.isfinite(curves)):
            raise ValueError(f"invalid seed-by-round accuracy for {partition}")
        if shape is None:
            shape = curves.shape
        elif curves.shape != shape:
            raise ValueError("all partitions must share aligned seed-by-round curves")
    if shape is None or shape[1] != 51:
        raise ValueError("the preview requires round 0 through round 50")


def _plot_context():
    return plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": 7.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save(figure: plt.Figure, output_base: Path) -> tuple[Path, Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = tuple(output_base.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg"))
    figure.savefig(outputs[0], dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    figure.savefig(outputs[1], bbox_inches="tight", pad_inches=0.02, facecolor="white")
    figure.savefig(outputs[2], bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(figure)
    return outputs


def _sidecar(output_base: Path, payload: Mapping[str, object]) -> Path:
    path = output_base.with_suffix(".json")
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
