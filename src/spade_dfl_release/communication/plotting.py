"""Portable selected Figure 10 rendering; original geometry and data retained."""
import csv
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch

OUTPUT_STEM = "fig_communication_saving_band"


BASELINE_COLOR = "#3182bd"

LOCAL_COLOR = "#ef5350"

MUTED_COLOR = "#66737c"

GRID_COLOR = "#dddddd"

LIGHT_RED = "#fde3e1"

def read_summary(path):
    numeric_int = {"T", "tau", "rounds", "n_seeds", "query_count", "transmitted_scalars"}
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if key in numeric_int:
                    row[key] = int(value)
                elif key == "schedule":
                    row[key] = value
                else:
                    row[key] = float(value)
            rows.append(row)
    if len(rows) != 8:
        raise ValueError("expected four budgets for each of two schedules")
    return rows

def _series(rows, schedule):
    selected = sorted(
        (row for row in rows if row["schedule"] == schedule),
        key=lambda row: row["T"],
    )
    if len(selected) != 4:
        raise ValueError("each schedule must contain exactly four observed budgets")
    return selected

def matched_budget_records(rows):
    baseline = _series(rows, "tau=1")
    local = _series(rows, "tau=T^(1/3)")
    records = []
    for baseline_row, local_row in zip(baseline, local):
        if baseline_row["T"] != local_row["T"]:
            raise ValueError("schedule rows are not paired by total local-update budget")
        baseline_scalars = baseline_row["transmitted_scalars"]
        local_scalars = local_row["transmitted_scalars"]
        if baseline_scalars <= 0 or local_scalars <= 0:
            raise ValueError("communication values must be positive")
        reduction = baseline_scalars / local_scalars
        records.append(
            {
                "T": baseline_row["T"],
                "tau": local_row["tau"],
                "baseline_scalars": baseline_scalars,
                "local_scalars": local_scalars,
                "reduction": reduction,
                "saving_percent": 100.0 * (1.0 - 1.0 / reduction),
                "baseline": baseline_row,
                "local": local_row,
            }
        )
    return records

def saving_label(rows):
    values=[r["saving_percent"] for r in matched_budget_records(rows)]
    return "{:.1f}\u2013{:.1f}% saved".format(min(values),max(values))


def configure_matplotlib():
    try:
        font_manager.findfont("Times New Roman", fallback_to_default=False)
        family="Times New Roman"
    except ValueError:
        family="DejaVu Serif"
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [family, "DejaVu Serif"],
            "font.size": 7.0,
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 6.3,
            "legend.frameon": False,
            "lines.linewidth": 1.10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )

def build_figure(rows):
    configure_matplotlib()
    records = matched_budget_records(rows)
    x_values = np.array([record["T"] for record in records], dtype=float)
    baseline = np.array(
        [record["baseline_scalars"] for record in records], dtype=float
    )
    local = np.array([record["local_scalars"] for record in records], dtype=float)
    if np.any(x_values <= 0) or np.any(baseline <= 0) or np.any(local <= 0):
        raise ValueError("log axes require strictly positive observed values")

    fig = plt.figure(figsize=(3.5, 2.65), facecolor="white")
    axis = fig.add_axes([0.20, 0.21, 0.77, 0.72])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_axisbelow(True)
    axis.tick_params(direction="out")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(750, 42000)
    axis.set_ylim(5.0e4, 4.2e7)
    axis.set_xticks(x_values)
    axis.set_xticklabels(["1k", "3.4k", "10.6k", "29.8k"])
    axis.set_yticks([1e5, 1e6, 1e7])
    axis.set_yticklabels(["10⁵", "10⁶", "10⁷"])
    axis.set_xlabel("Total local-update budget, T")
    axis.set_ylabel("Transmitted model scalars")
    axis.grid(True, which="major", color=GRID_COLOR, linewidth=0.4, alpha=0.6)
    axis.grid(False, which="minor")

    saving_region = axis.fill_between(
        x_values,
        local,
        baseline,
        color=LIGHT_RED,
        alpha=0.82,
        zorder=1,
    )
    saving_region.set_gid("communication-saving-region")
    axis.plot(
        x_values,
        baseline,
        "o--",
        color=BASELINE_COLOR,
        markerfacecolor="white",
        markeredgewidth=0.8,
        markersize=3.3,
        label="Every update (τ = 1)",
        zorder=4,
    )
    axis.plot(
        x_values,
        local,
        "D-",
        color=LOCAL_COLOR,
        markerfacecolor=LOCAL_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.4,
        markersize=3.5,
        label="Local schedule (τ = T⅓)",
        zorder=5,
    )

    for x_value, low, high, record in zip(x_values, local, baseline, records):
        arrow = FancyArrowPatch(
            (x_value, low * 1.22),
            (x_value, high / 1.22),
            arrowstyle="<->",
            mutation_scale=5.0,
            color=MUTED_COLOR,
            linewidth=0.55,
            zorder=3,
        )
        axis.add_patch(arrow)
        axis.text(
            x_value * 1.06,
            np.sqrt(low * high),
            "{:g}×".format(record["reduction"]),
            ha="left",
            va="center",
            fontsize=5.2,
            fontweight="bold",
            color=MUTED_COLOR,
        )

    axis.text(
        0.01,
        1.012,
        "SPADE-DFL local",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.4,
        color=LOCAL_COLOR,
        clip_on=False,
    )
    axis.text(
        0.50,
        1.012,
        "Shaded area: communication saved",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.1,
        color=MUTED_COLOR,
        clip_on=False,
    )
    axis.text(
        0.99,
        1.012,
        "Every update",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
        color=BASELINE_COLOR,
        clip_on=False,
    )
    axis.text(
        6200,
        1.35e6,
        saving_label(rows),
        ha="center",
        va="center",
        fontsize=6.0,
        fontweight="bold",
        color="#c94843",
        zorder=6,
    )
    return fig

def verify_figure(fig, rows):
    records=matched_budget_records(rows)
    if len(fig.axes)!=1:raise ValueError('Figure 10 requires one panel')
    axis=fig.axes[0];x=np.asarray([r['T'] for r in records],dtype=float)
    baseline=np.asarray([r['baseline_scalars'] for r in records],dtype=float)
    local=np.asarray([r['local_scalars'] for r in records],dtype=float)
    if np.any(x<=0) or np.any(baseline<=0) or np.any(local<=0):raise ValueError('positive log-axis data required')
    if axis.get_xscale()!='log' or axis.get_yscale()!='log' or axis.get_xlabel()!='Total local-update budget, T' or axis.get_ylabel()!='Transmitted model scalars':raise ValueError('Figure 10 axes/units changed')
    if len(axis.lines)!=2 or len(axis.collections)!=1:raise ValueError('Figure 10 requires two series and one saving region')
    for line,values,color in zip(axis.lines,(baseline,local),(BASELINE_COLOR,LOCAL_COLOR)):
        if not np.array_equal(line.get_xdata(),x) or not np.array_equal(line.get_ydata(),values) or line.get_color()!=color:raise ValueError('Figure 10 series changed')
    region=axis.collections[0]
    expected=np.concatenate(([[x[0],baseline[0]]],np.column_stack((x,local)),[[x[-1],baseline[-1]]],np.column_stack((x[::-1],baseline[::-1])),[[x[0],baseline[0]]]))
    if region.get_gid()!='communication-saving-region' or len(region.get_paths())!=1 or not np.array_equal(region.get_paths()[0].vertices,expected):raise ValueError('deterministic saving region changed')
    expected_text=['{:g}\u00d7'.format(r['reduction']) for r in records]+['SPADE-DFL local','Shaded area: communication saved','Every update',saving_label(rows)]
    if [t.get_text() for t in axis.texts]!=expected_text:raise ValueError('Figure 10 visible annotations changed')
    fig.canvas.draw()
    if any(t.get_fontsize()<5 for t in axis.texts+[axis.xaxis.label,axis.yaxis.label]+axis.get_xticklabels()+axis.get_yticklabels()):raise ValueError('Figure 10 visible glyph parent size below 5pt')
    return dict(panels=1,series=2,alignment='not_applicable_single_panel',saving_region='deterministic_communication_saved_not_CI',saving_label=saving_label(rows))


def export_figure(fig, output_dir, rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_figure(fig, rows)
    stem = output_dir / OUTPUT_STEM
    save_options = {
        "bbox_inches": "tight",
        "pad_inches": 0.02,
        "facecolor": "white",
    }
    pdf = stem.with_suffix(".pdf")
    svg = stem.with_suffix(".svg")
    png = stem.with_suffix(".png")
    tiff = stem.with_suffix(".tiff")
    fig.savefig(pdf, **save_options)
    fig.savefig(svg, **save_options)
    fig.savefig(png, dpi=600, **save_options)
    fig.savefig(
        tiff,
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
        **save_options,
    )
    return [pdf, svg, png, tiff]
