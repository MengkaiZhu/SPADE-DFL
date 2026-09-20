from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DISPLAY_PANEL_ORDER = ("iid", "dir_0.6", "dir_0.3", "pathological")
DISPLAY_LABELS = {
    "iid": "IID",
    "dir_0.6": "Dirichlet\nα = 0.6",
    "dir_0.3": "Dirichlet\nα = 0.3",
    "pathological": "Pathological\nnon-IID",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_source_csv(
    path: Path,
    report: dict[str, object],
) -> None:
    rows: list[dict[str, object]] = []
    for panel in DISPLAY_PANEL_ORDER:
        values = report["panels"][panel]
        rows.append(
            {
                "panel": panel,
                "display_label": DISPLAY_LABELS[panel].replace("\n", " "),
                "series": "old_representation_mean",
                "seed": "mean_of_20260724_20260725_20260726",
                "accuracy": values["old_mean_accuracy"],
                "standard_deviation": "",
                "source": report["old_accuracy_source"],
            }
        )
        for seed, accuracy in zip(
            (20260724, 20260725, 20260726),
            values["new_seed_accuracies"],
        ):
            rows.append(
                {
                    "panel": panel,
                    "display_label": DISPLAY_LABELS[panel].replace("\n", " "),
                    "series": "resnet18_v1_pca10_seed",
                    "seed": seed,
                    "accuracy": accuracy,
                    "standard_deviation": "",
                    "source": report["raw_runs_sha256"],
                }
            )
        rows.append(
            {
                "panel": panel,
                "display_label": DISPLAY_LABELS[panel].replace("\n", " "),
                "series": "resnet18_v1_pca10_mean",
                "seed": "mean_of_20260724_20260725_20260726",
                "accuracy": values["new_mean_accuracy"],
                "standard_deviation": float(
                    np.std(values["new_seed_accuracies"], ddof=0)
                ),
                "source": report["raw_runs_sha256"],
            }
        )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_gate2_comparison(
    report_path: Path,
    output_stem: Path,
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    report_file = Path(report_path)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if set(report.get("panels", {})) != set(DISPLAY_PANEL_ORDER):
        raise ValueError("Gate 2 report does not contain the exact four panels")
    output = Path(output_stem)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_csv = output.with_suffix(".source.csv")
    _write_source_csv(source_csv, report)

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "Liberation Sans",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axis = plt.subplots(figsize=(7.0, 3.2), constrained_layout=True)
    x_positions = np.arange(len(DISPLAY_PANEL_ORDER), dtype=np.float64)
    old_means = np.asarray(
        [report["panels"][panel]["old_mean_accuracy"] for panel in DISPLAY_PANEL_ORDER]
    )
    new_means = np.asarray(
        [report["panels"][panel]["new_mean_accuracy"] for panel in DISPLAY_PANEL_ORDER]
    )
    new_stds = np.asarray(
        [
            np.std(report["panels"][panel]["new_seed_accuracies"], ddof=0)
            for panel in DISPLAY_PANEL_ORDER
        ]
    )
    lower = max(0.0, float(min(old_means.min(), new_means.min()) - 0.07))
    upper = min(1.0, float(max(old_means.max(), new_means.max()) + 0.09))
    annotation_y = upper - 0.025
    old_x = x_positions - 0.12
    new_x = x_positions + 0.12
    for index in range(len(DISPLAY_PANEL_ORDER)):
        axis.plot(
            [old_x[index], new_x[index]],
            [old_means[index], new_means[index]],
            color="#b8b8b8",
            linewidth=1.0,
            zorder=1,
        )
    axis.scatter(
        old_x,
        old_means,
        marker="s",
        s=30,
        color="#666666",
        edgecolor="white",
        linewidth=0.6,
        label="Original: 3-holdout mean",
        zorder=3,
    )
    jitter = np.asarray((-0.045, 0.0, 0.045))
    for index, panel in enumerate(DISPLAY_PANEL_ORDER):
        seed_values = np.asarray(report["panels"][panel]["new_seed_accuracies"])
        axis.scatter(
            new_x[index] + jitter,
            seed_values,
            marker="o",
            s=20,
            facecolor="white",
            edgecolor="#0072B2",
            linewidth=0.8,
            label="ResNet18-V1–PCA10: holdouts" if index == 0 else None,
            zorder=3,
        )
        delta_points = 100.0 * (new_means[index] - old_means[index])
        axis.text(
            new_x[index],
            annotation_y,
            f"+{delta_points:.1f} pp",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#005b8c",
        )
    axis.errorbar(
        new_x,
        new_means,
        yerr=new_stds,
        fmt="D",
        markersize=5.8,
        color="#0072B2",
        markeredgecolor="white",
        markeredgewidth=0.6,
        elinewidth=0.9,
        capsize=2.5,
        label="ResNet18-V1–PCA10: mean ± SD",
        zorder=4,
    )
    axis.set_ylim(lower, upper)
    axis.set_xlim(-0.55, len(DISPLAY_PANEL_ORDER) - 0.45)
    axis.set_xticks(x_positions, [DISPLAY_LABELS[p] for p in DISPLAY_PANEL_ORDER])
    axis.set_ylabel("Final Training-holdout accuracy")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        ncol=3,
        handletextpad=0.9,
        columnspacing=1.4,
    )

    outputs = tuple(output.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png", ".tiff"))
    figure.savefig(outputs[0], bbox_inches="tight", facecolor="white", metadata={"CreationDate": None})
    figure.savefig(outputs[1], bbox_inches="tight", facecolor="white", metadata={"Date": None})
    figure.savefig(outputs[2], dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(
        outputs[3],
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    qa_payload = {
        "purpose": "Gate 2 diagnostic only; not manuscript Fig. 4",
        "display_panel_order": list(DISPLAY_PANEL_ORDER),
        "source_report": str(report_file.resolve()),
        "source_report_sha256": _sha256(report_file),
        "source_csv": str(source_csv.resolve()),
        "source_csv_sha256": _sha256(source_csv),
        "outputs": {
            path.suffix.lstrip("."): {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for path in outputs
        },
        "raster_dpi": 600,
        "minimum_font_target_pt": 5.0,
        "smoothing_or_interpolation": False,
    }
    output.with_suffix(".qa.json").write_text(
        json.dumps(qa_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return outputs
