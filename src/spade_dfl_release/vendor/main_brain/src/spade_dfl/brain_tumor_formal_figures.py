from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from .brain_tumor_suite import FIGURE_STYLES, VISIBLE_METHOD_LABELS
from .isolation import BRAIN_TUMOR_METHOD_IDS


FORMAL_PANEL_ORDER = ("iid", "dir_0.6", "dir_0.3", "pathological")
FORMAL_PANEL_LABELS = {
    "iid": "IID",
    "dir_0.6": "Non-IID: Dir 0.6",
    "dir_0.3": "Non-IID: Dir 0.3",
    "pathological": "Pathological",
}
FORMAL_FILE_STEMS = {
    "iid": "Brain Tumor-iid",
    "dir_0.6": "Brain Tumor-dir 0.6",
    "dir_0.3": "Brain Tumor-dir 0.3",
    "pathological": "Brain Tumor-pathological",
}


def formal_accuracy_ylim_percent(
    aggregates: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> tuple[float, float]:
    """Keep the original 15--60% scale, extending only in five-point steps."""
    upper_observed = max(
        float(
            np.max(
                100.0
                * (
                    np.asarray(aggregates[panel][method]["accuracy_mean"])
                    + np.asarray(aggregates[panel][method]["accuracy_std"])
                )
            )
        )
        for panel in FORMAL_PANEL_ORDER
        for method in BRAIN_TUMOR_METHOD_IDS
    )
    upper = max(60.0, 5.0 * math.ceil(upper_observed / 5.0))
    return 15.0, upper


def _write_source_csv(
    path: Path,
    panel: str,
    aggregates: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "panel",
                "method_id",
                "visible_method",
                "round",
                "accuracy_mean_percent",
                "accuracy_std_percent",
            ),
        )
        writer.writeheader()
        for method in BRAIN_TUMOR_METHOD_IDS:
            aggregate = aggregates[panel][method]
            rounds = np.asarray(aggregate["round"], dtype=np.float64)
            means = 100.0 * np.asarray(aggregate["accuracy_mean"], dtype=np.float64)
            stds = 100.0 * np.asarray(aggregate["accuracy_std"], dtype=np.float64)
            if not (rounds.shape == means.shape == stds.shape):
                raise ValueError(f"inconsistent aggregate shapes for {panel}/{method}")
            for round_value, mean, std in zip(rounds, means, stds):
                writer.writerow(
                    {
                        "panel": panel,
                        "method_id": method,
                        "visible_method": VISIBLE_METHOD_LABELS[method],
                        "round": f"{round_value:g}",
                        "accuracy_mean_percent": f"{mean:.12g}",
                        "accuracy_std_percent": f"{std:.12g}",
                    }
                )


def write_formal_accuracy_figures(
    aggregates: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    output_dir: Path,
    *,
    n_rounds: int,
) -> dict[str, tuple[Path, ...]]:
    """Write the four original-format Fig. 4 panels from unsmoothed aggregates."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    y_limits = formal_accuracy_ylim_percent(aggregates)
    marker_every = max(1, int(n_rounds) // 10)
    x_ticks = np.arange(0, int(n_rounds) + 1, 10)
    if int(n_rounds) < 10:
        x_ticks = np.arange(0, int(n_rounds) + 1, 1)

    settings = {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Liberation Serif",
            "DejaVu Serif",
        ],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }

    outputs: dict[str, tuple[Path, ...]] = {}
    with plt.rc_context(settings):
        for panel in FORMAL_PANEL_ORDER:
            figure, axis = plt.subplots(figsize=(2.4, 1.8))
            for method in BRAIN_TUMOR_METHOD_IDS:
                aggregate = aggregates[panel][method]
                rounds = np.asarray(aggregate["round"], dtype=np.float64)
                mean = 100.0 * np.asarray(
                    aggregate["accuracy_mean"], dtype=np.float64
                )
                std = 100.0 * np.asarray(
                    aggregate["accuracy_std"], dtype=np.float64
                )
                if not (rounds.shape == mean.shape == std.shape):
                    raise ValueError(
                        f"inconsistent aggregate shapes for {panel}/{method}"
                    )
                color, linestyle, marker = FIGURE_STYLES[method]
                axis.plot(
                    rounds,
                    mean,
                    label=VISIBLE_METHOD_LABELS[method],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markevery=marker_every,
                    linewidth=1.8 if method == "spade_dfl" else 1.1,
                    markersize=2.8,
                )
                axis.fill_between(
                    rounds,
                    np.clip(mean - std, 0.0, 100.0),
                    np.clip(mean + std, 0.0, 100.0),
                    color=color,
                    alpha=0.14,
                    linewidth=0.0,
                )
            axis.set_xlim(0, int(n_rounds))
            axis.set_xticks(x_ticks)
            axis.set_ylim(*y_limits)
            axis.yaxis.set_major_locator(MultipleLocator(5.0))
            axis.set_xlabel("Rounds")
            axis.set_ylabel("Accuracy (%)")
            axis.grid(
                axis="y",
                color="#d9d9d9",
                linewidth=0.5,
                alpha=0.65,
            )
            axis.text(
                0.03,
                0.04,
                FORMAL_PANEL_LABELS[panel],
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
            )
            axis.legend(
                loc="lower right",
                ncol=1,
                fontsize=5,
                frameon=True,
                framealpha=0.94,
                edgecolor="#cccccc",
            )
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            figure.tight_layout(pad=0.02)

            stem = FORMAL_FILE_STEMS[panel]
            paths = tuple(destination / f"{stem}.{suffix}" for suffix in ("pdf", "svg", "png", "tiff"))
            for path in paths:
                save_kwargs: dict[str, object] = {
                    "bbox_inches": "tight",
                    "pad_inches": 0,
                }
                if path.suffix in {".png", ".tiff"}:
                    save_kwargs["dpi"] = 600
                if path.suffix == ".tiff":
                    save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
                figure.savefig(path, **save_kwargs)
            plt.close(figure)
            _write_source_csv(destination / f"{stem}.source.csv", panel, aggregates)
            outputs[panel] = paths
    return outputs
