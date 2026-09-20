"""Result-preserving paper presentation. No experiment or training entry points."""
from __future__ import annotations

import csv
import ast
import hashlib
import itertools
import json
import os
import re
import shutil
import tempfile
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy import stats

from .statistics import benjamini_hochberg
from .publication import verify_publication_file

DATASETS = ("mnist", "fashion", "alzheimer", "brain")
PARTITIONS = ("iid", "dir_0.6", "dir_0.3", "pathological")
METHODS = ("zo2_lt_admm_vr", "zo2_lt_admm", "zo2_dp_sgd", "zo2_dp_fedavg", "one_point_dsg", "one_point_dsgt", "spade_dfl")
DISTRIBUTIONS = PARTITIONS + ("dir_0.1",)
CLIENT_COUNTS = (30, 40, 50, 75, 100)
EPSILONS = (4, 8, 16, 24, 32)
ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "configs/paper_outputs/manifest.json"
MANIFEST_SHA256 = "5a184c73b1c8dd4509a450e08c6ba4f8b48b21e5f51bffdc4b56b3370f8ed009"
# Stable internal SVG IDs from the approved original delivery, not paper identity.
SVG_HASHSALT = "a8046ac75e26ea21eacba27cc77d75301d81d8be9af765c9de2952f1ad523829"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("paper presentation manifest semantic hash mismatch")
    if manifest["export"]["svg_hashsalt"] != SVG_HASHSALT:
        raise ValueError("frozen SVG export identity mismatch")
    return manifest


def display(value):
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _grid(rows, fields, expected):
    keys = [tuple(str(r[f]) for f in fields) for r in rows]
    expected = {tuple(str(v) for v in key) for key in expected}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"missing, extra or duplicate keys: {fields}")
    return dict(zip(keys, rows))


def validate_data(data):
    manifest = load_manifest()
    if set(data) != set(manifest["inputs"]):
        raise ValueError("input family mismatch")
    for name, spec in manifest["inputs"].items():
        rows = data[name]
        if len(rows) != spec["rows"] or any(set(r) != set(spec["columns"]) for r in rows):
            raise ValueError(f"schema/count mismatch: {name}")
    for name, rounds in (("main_final_accuracy.csv", (50,)), ("main_roundwise.csv", range(51))):
        _grid(data[name], ("dataset", "partition", "method", "round"), itertools.product(DATASETS, PARTITIONS, METHODS, rounds))
        for r in data[name]:
            if int(r["ddof"]) != 0 or int(r["n_trajectories"]) != (150 if r["dataset"] == "alzheimer" else 30):
                raise ValueError("main population SD/trajectory count mismatch")
            if not 0 <= float(r["mean_percent"]) <= 100 or not 0 <= float(r["sd_percent"]) <= 100:
                raise ValueError("invalid main accuracy/spread")
    main = {(r["dataset"], r["partition"], r["method"]): r for r in data["main_roundwise.csv"] if r["round"] == "50"}
    for r in data["main_final_accuracy.csv"]:
        if r != main[(r["dataset"], r["partition"], r["method"])]:
            raise ValueError("final accuracy does not equal round 50")
    cells = data["sensitivity_cells.csv"]
    conditions = set((n, 32) for n in CLIENT_COUNTS) | set((100, e) for e in EPSILONS)
    _grid(cells, ("dataset", "n", "epsilon"), ((d, n, e) for d in DATASETS for n, e in conditions))
    for r in cells:
        if (r["ddof"], r["n_seed"], r["n_fold_seed"]) != ("1", "5", "25" if r["dataset"] == "alzheimer" else "5"):
            raise ValueError("sensitivity sample SD/within-seed fold pooling count mismatch")
        if (r["validation_only"], r["formal_authorized"], r["previous_test_exposure"]) != ("True", "False", "True"):
            raise ValueError("sensitivity validation identity mismatch")
    dist = [r for r in data["s15_roundwise.csv"] if r["suite"] == "distribution"]
    methods = [r for r in data["s15_roundwise.csv"] if r["suite"] == "method"]
    _grid(dist, ("partition", "method", "seed", "round"), itertools.product(DISTRIBUTIONS, ("spade_dfl_private",), range(30), range(51)))
    _grid(methods, ("partition", "method", "seed", "round"), itertools.product(PARTITIONS, METHODS, range(30), range(51)))
    if len(dist) + len(methods) != len(data["s15_roundwise.csv"]):
        raise ValueError("unknown S15 suite")
    for r in dist + methods:
        central = r["method"] == "zo2_dp_sgd"
        if int(r["n_clients"]) != (1000 if r["suite"] == "distribution" else 1 if central else 31):
            raise ValueError("S15 client count")
        if r["axis_semantics"] != ("centralized_optimizer_step" if central else "communication_round") or int(r["communication_round"]) != (0 if central else int(r["round"])):
            raise ValueError("DP-SGD axis semantics/communication mismatch")
        if not 0 <= float(r["accuracy"]) <= 1:
            raise ValueError("invalid S15 accuracy")
    _grid(data["s15_client_proportions.csv"], ("partition", "seed", "client"), itertools.product(DISTRIBUTIONS, range(30), range(1000)))
    if any(not 0 <= float(r["p_y7"]) <= 1 for r in data["s15_client_proportions.csv"]):
        raise ValueError("invalid client proportion")
    _grid(data["s15_wilcoxon_bh.csv"], ("partition", "method", "round"), itertools.product(PARTITIONS, METHODS[1:], range(1, 51)))
    if any(r["n_pairs"] != "30" for r in data["s15_wilcoxon_bh.csv"]):
        raise ValueError("paired count mismatch")
    protocol_path = ROOT / "configs/communication/manifest.json"
    if sha256(protocol_path) != manifest["renderer_lineage"]["communication"]["manifest_sha256"]:
        raise ValueError("Communication protocol hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = [(b["T"], b[field]) for b in protocol["budgets"] for field in ("tau_baseline", "tau_local")]
    _grid(data["communication_summary.csv"], ("T", "tau"), expected)
    cfg = protocol["config"]
    for r in data["communication_summary.csv"]:
        budget, tau, rounds = int(r["T"]), int(r["tau"]), int(r["rounds"])
        if rounds * tau != budget or int(r["local_steps"]) != budget or r["schedule"] != ("tau=1" if tau == 1 else "tau=T^(1/3)"):
            raise ValueError("communication budget/schedule mismatch")
        if int(r["query_count"]) != cfg["n_clients"] * (rounds * cfg["samples_per_client"] + budget * cfg["batch_size"]) or int(r["transmitted_scalars"]) != 2 * cfg["n_clients"] * cfg["dimension"] * rounds:
            raise ValueError("communication independent accounting formula mismatch")
    _communication_rows(data)
    return data


def load_inputs(reference_root):
    reference_root = Path(reference_root)
    manifest = load_manifest()
    verify_renderer_lineage()
    verification = reference_root / "paper_values/verification.json"
    if sha256(verification) != manifest["verification_sha256"]:
        raise ValueError("Reference-verification hash mismatch")
    frozen = json.loads(verification.read_text(encoding="utf-8"))
    data = {}
    for name, spec in manifest["inputs"].items():
        path = reference_root / "aggregates" / name
        if spec["sha256"] != frozen["normalized_sha256"][name] or sha256(path) != spec["sha256"]:
            raise ValueError(f"normalized input hash mismatch: {name}")
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != spec["columns"]:
                raise ValueError(f"input column order mismatch: {name}")
            data[name] = list(reader)
    return validate_data(data)


def verify_renderer_lineage():
    """Check source identities, current publication bytes and presentation AST."""
    manifest = load_manifest()
    for row in manifest["renderer_lineage"].values():
        if row["source_path"].startswith("src/"):
            try:
                verify_publication_file(ROOT / row["source_path"], row["sha256"], root=ROOT)
            except ValueError as error:
                raise ValueError("packaged renderer source hash mismatch") from error
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    spec = manifest["adapted_renderer_ast"]
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    payload = "\n".join(ast.dump(functions[name], include_attributes=False) for name in spec["names"])
    if hashlib.sha256(payload.encode()).hexdigest() != spec["sha256"]:
        raise ValueError("adapted renderer AST hash mismatch")
    return len(manifest["renderer_lineage"])


def method_cells(rows):
    indexed = _grid(rows, ("method",), ((m,) for m in METHODS))
    means = [Decimal(str(indexed[(m,)]["mean_percent"])) for m in METHODS]
    best_other = max(range(len(METHODS) - 1), key=lambda i: means[i])
    for values in (means, [Decimal(display(v)) for v in means]):
        if values[-1] <= max(values[:-1]) or sum(v == max(values[:-1]) for v in values[:-1]) != 1:
            raise ValueError("raw/display tie or SPADE is not uniquely best")
    result = []
    for i, method in enumerate(METHODS):
        row = indexed[(method,)]
        pair = display(row["mean_percent"]) + r"{\pm}" + display(row["sd_percent"])
        if i == len(METHODS) - 1:
            pair = r"\mathbf{" + pair + "}"
        elif i == best_other:
            pair = r"\underline{" + pair + "}"
        result.append("$" + pair + "$")
    result.append("$+" + display(means[-1] - means[best_other]) + "$")
    return result


def render_tables(data):
    validate_data(data)
    cells = []
    for dataset, partition in itertools.product(DATASETS, PARTITIONS):
        cells.extend(method_cells([r for r in data["main_final_accuracy.csv"] if (r["dataset"], r["partition"]) == (dataset, partition)]))
    sensitivity = {(r["dataset"], int(r["n"]), int(r["epsilon"])): r for r in data["sensitivity_cells.csv"]}
    all_cells = {"table_II.tex": cells}
    for name, conditions in (("table_III.tex", [(n, 32) for n in CLIENT_COUNTS]), ("table_IV.tex", [(100, e) for e in EPSILONS])):
        all_cells[name] = ["$" + display(sensitivity[d, n, e]["mean_percent"]) + r"\pm" + display(sensitivity[d, n, e]["sd_percent"]) + "$" for d in DATASETS for n, e in conditions]
    result = {}
    for name, spec in load_manifest()["tables"].items():
        if len(all_cells[name]) != spec["cells"]:
            raise ValueError("table cell count")
        result[name] = re.sub(r"@@CELL_(\d+)@@", lambda m: all_cells[name][int(m[1])], spec["template"])
    return result


def validate_tables(tables):
    expected = load_manifest()["tables"]
    if set(tables) != set(expected):
        raise ValueError("table output set mismatch")
    for name, content in tables.items():
        if any(token.lower() in content.lower() for token in ("notes:", "note:", "footnote", "minipage", "threeparttable", "tablenotes")):
            raise ValueError("forbidden table note")
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != expected[name]["rendered_sha256"]:
            raise ValueError(f"manuscript cell/structure difference: {name}")
    return {"table_cell_differences": 0}


def _main_arrays(data, dataset):
    index = {(r["partition"], r["method"], int(r["round"])): r for r in data["main_roundwise.csv"] if r["dataset"] == dataset}
    return {p: {m: {"round": np.arange(51), "accuracy_mean": np.array([float(index[p, m, t]["mean_percent"]) for t in range(51)]), "accuracy_std": np.array([float(index[p, m, t]["sd_percent"]) for t in range(51)])} for m in METHODS} for p in PARTITIONS}


def _raw_arrays(data, suite):
    rows = [r for r in data["s15_roundwise.csv"] if r["suite"] == suite]
    index = {(r["partition"], r["method"], int(r["seed"]), int(r["round"])): float(r["accuracy"]) for r in rows}
    partitions = DISTRIBUTIONS if suite == "distribution" else PARTITIONS
    methods = ("spade_dfl_private",) if suite == "distribution" else METHODS
    return {(p, m): np.array([[index[p, m, s, t] for t in range(51)] for s in range(30)]) for p, m in itertools.product(partitions, methods)}


def _distribution_study(data):
    curves = _raw_arrays(data, "distribution")
    index = {(r["partition"], int(r["seed"]), int(r["client"])): float(r["p_y7"]) for r in data["s15_client_proportions.csv"]}
    proportions, entropy = {}, {}
    for p in DISTRIBUTIONS:
        values = np.array([index[p, s, c] for s in range(30) for c in range(1000)])
        proportions[p] = values
        e = np.zeros_like(values)
        interior = (values > 0) & (values < 1)
        q = values[interior]
        e[interior] = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
        entropy[p] = e
    return SimpleNamespace(partitions=DISTRIBUTIONS, seeds=tuple(range(30)), accuracy_by_seed={p: curves[p, "spade_dfl_private"] for p in DISTRIBUTIONS}, label_proportions=proportions, label_entropy=entropy)


def _volcano_records(data):
    arrays = {key: values * 100.0 for key, values in _raw_arrays(data, "method").items()}
    indexed = {(r["partition"], r["method"], int(r["round"])): r for r in data["s15_wilcoxon_bh.csv"]}
    records, p_values = [], []
    for partition, method, t in itertools.product(PARTITIONS, METHODS[1:], range(1, 51)):
        row = indexed[partition, method, t]
        differences = arrays[partition, method][:, t] - arrays[partition, METHODS[0]][:, t]
        statistic, probability = (0.0, 1.0) if np.all(differences == 0) else stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto")
        if not np.array_equal(differences, np.array(json.loads(row["paired_differences_pp"]))) or float(row["mean_paired_accuracy_difference_pp"]) != float(differences.mean()) or float(row["wilcoxon_statistic"]) != statistic or float(row["raw_p_value"]) != probability:
            raise ValueError(f"paired Wilcoxon/raw differences mismatch: {partition}/{method}/{t}")
        record = SimpleNamespace(**{k: int(v) if k in ("round", "n_pairs") else str(v) if k in ("partition", "method", "classification") else tuple(json.loads(v)) if k == "paired_differences_pp" else float(v) for k, v in row.items()})
        records.append(record)
        p_values.append(probability)
    for r, q in zip(records, benjamini_hochberg(p_values)):
        classification = "unresolved" if q >= 0.05 or r.mean_paired_accuracy_difference_pp == 0 else "better" if r.mean_paired_accuracy_difference_pp > 0 else "worse"
        if r.adjusted_q_value != q or r.minus_log10_q != float(-np.log10(max(q, np.finfo(float).tiny))) or r.classification != classification:
            raise ValueError("independent BH/classification mismatch")
    counts = {c: sum(r.classification == c for r in records) for c in ("better", "worse", "unresolved")}
    if counts != {"better": 424, "worse": 410, "unresolved": 366}:
        raise ValueError("Figure 9 classification counts mismatch")
    return tuple(records)


def _communication_rows(data):
    text = {"schedule", "diagnostic", "paired_by"}
    integers = {"T", "tau", "n_seeds", "rounds", "local_steps", "query_count", "transmitted_scalars"}
    rows = [{k: v if k in text else int(v) if k in integers else float(v) for k, v in r.items()} for r in data["communication_summary.csv"]]
    if any(r["diagnostic"] != "non_private" or r["paired_by"] != "seed" or r["n_seeds"] != 30 for r in rows):
        raise ValueError("communication diagnostic/paired identity")
    return sorted(rows, key=lambda r: (r["T"], r["tau"]))


def figure_evidence(data):
    validate_data(data)
    records = _volcano_records(data)
    from .communication.plotting import saving_label
    return {
        "main": {"rows": 5712, "panels": 16, "curves": 112, "points_per_curve": 51, "ddof": 0},
        "figure7": {"rows": 7650, "curves": 5, "ci_bands": 5, "paired_inset_curves": 4, "confidence": "two-sided Student-t 95%; 30 paired seeds"},
        "figure8": {"rows": 150000, "partitions": 5, "display_points_per_partition": 1400, "display_seed": 20260827},
        "figure9": {"raw_rows": 42840, "comparisons": len(records), "reference": "zo2_lt_admm_vr", "classifications": {c: sum(r.classification == c for r in records) for c in ("better", "worse", "unresolved")}},
        "figure10": {"rows": len(_communication_rows(data)), "saving_label": saving_label(_communication_rows(data))},
    }


def build_figures(data):
    # Each frozen launcher starts from Matplotlib defaults. The communication renderer changes
    # global rcParams; isolate that side effect without inventing any style value.
    with plt.rc_context(matplotlib.rcParamsDefault):
        return _build_figures(data)


def _build_figures(data):
    figure_evidence(data)
    figures = {}
    prefixes = ("MNIST", "Fashion MNIST", "Alzheimer", "Brain Tumor")
    suffixes = ("iid", "dir 0.6", "dir 0.3", "pathological")
    for number, (dataset, prefix) in enumerate(zip(DATASETS, prefixes), 3):
        arrays = _main_arrays(data, dataset)
        if dataset in ("mnist", "fashion"):
            renderer = _mnist_renderer() if dataset == "mnist" else _fashion_renderer()
            labels = ("2P-LT-ADMM-VR", "2P-LT-ADMM", "2P-DP-SGD", "2P-DP-FedAvg", "1P-DSG", "1P-DSGT", "SPADE-DFL")
            panels = {p: renderer({label: arrays[p][m] for m, label in zip(METHODS, labels)}, None, seed_count=30, partition=p) for p in PARTITIONS}
        elif dataset == "alzheimer":
            panels = _render_alzheimer(arrays)
        else:
            panels = _brain_renderer()(arrays, None, n_rounds=50)
        for p, suffix in zip(PARTITIONS, suffixes):
            figures[f"figure_{number}/{prefix}-{suffix}"] = panels[p]
    study = _distribution_study(data)
    accuracy, labels = _distribution_renderers()
    figures["figure_7/fig-distribution-accuracy"] = accuracy(study, None)
    figures["figure_8/fig-client-label-heterogeneity"] = labels(study, None)
    figures["figure_9/fig-method-volcano"] = _volcano_renderer()(_volcano_records(data))
    from .communication.plotting import build_figure
    figures["figure_10/fig_communication_saving_band"] = build_figure(_communication_rows(data))
    return figures


def _same(actual, expected, label):
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        raise ValueError(f"figure series difference: {label}")


def _band(collection, x, low, high, label):
    expected = np.column_stack((np.r_[x[0], x, x[-1], x[::-1], x[0]], np.r_[high[0], low, high[-1], high[::-1], high[0]]))
    if len(collection.get_paths()) != 1:
        raise ValueError("band path count mismatch")
    _same(collection.get_paths()[0].vertices, expected, label)


def validate_figures(figures, data):
    expected_stems = {f"figure_{number}/{prefix}-{suffix}" for number, prefix in zip(range(3, 7), ("MNIST", "Fashion MNIST", "Alzheimer", "Brain Tumor")) for suffix in ("iid", "dir 0.6", "dir 0.3", "pathological")}
    expected_stems |= {"figure_7/fig-distribution-accuracy", "figure_8/fig-client-label-heterogeneity", "figure_9/fig-method-volcano", "figure_10/fig_communication_saving_band"}
    if set(figures) != expected_stems:
        raise ValueError("figure output stems mismatch")
    for number, (dataset, prefix) in enumerate(zip(DATASETS, ("MNIST", "Fashion MNIST", "Alzheimer", "Brain Tumor")), 3):
        arrays = _main_arrays(data, dataset)
        for p, suffix in zip(PARTITIONS, ("iid", "dir 0.6", "dir 0.3", "pathological")):
            axis = figures[f"figure_{number}/{prefix}-{suffix}"].axes[0]
            if axis.get_xlabel() != "Rounds" or axis.get_ylabel() != "Accuracy (%)":
                raise ValueError("main visible axis text changed")
            if [t.get_text() for t in axis.get_legend().get_texts()] != ["LT-ADMM-VR", "LT-ADMM", "DP-SGD", "DP-FedAvg", "1P-DSG", "1P-DSGT", "SPADE-DFL"]:
                raise ValueError("main legend identity/order changed")
            if len(axis.lines) != 7 or len(axis.collections) != 7:
                raise ValueError("main curve/band count")
            for method, line, band in zip(METHODS, axis.lines, axis.collections):
                a = arrays[p][method]
                _same(line.get_xdata(), a["round"], method)
                _same(line.get_ydata(), a["accuracy_mean"], method)
                lo, hi = a["accuracy_mean"] - a["accuracy_std"], a["accuracy_mean"] + a["accuracy_std"]
                if dataset == "brain":
                    lo, hi = np.clip(lo, 0, 100), np.clip(hi, 0, 100)
                _band(band, a["round"], lo, hi, method)
            if dataset == "mnist" and p == "iid":
                if len(axis.child_axes) != 1:
                    raise ValueError("MNIST IID inset missing")
                inset = axis.child_axes[0]
                if len(inset.lines) != 7 or len(inset.collections) != 7:
                    raise ValueError("MNIST IID inset curve count")
                for method, line, band in zip(METHODS, inset.lines, inset.collections):
                    a = arrays[p][method]
                    _same(line.get_xdata(), a["round"], "MNIST inset rounds")
                    _same(line.get_ydata(), a["accuracy_mean"], "MNIST inset mean")
                    _band(band, a["round"], a["accuracy_mean"] - a["accuracy_std"], a["accuracy_mean"] + a["accuracy_std"], "MNIST inset SD")
    study = _distribution_study(data)
    axis, inset = figures["figure_7/fig-distribution-accuracy"].axes
    if len(axis.lines) != 5 or len(axis.collections) != 5 or len(inset.lines) != 5:
        raise ValueError("Figure 7 curve/band/inset count")
    for i, p in enumerate(DISTRIBUTIONS):
        curves = study.accuracy_by_seed[p] * 100.0
        mean = curves.mean(axis=0)
        half = stats.t.ppf(0.975, 29) * (curves.std(axis=0, ddof=1) / np.sqrt(30))
        _same(axis.lines[i].get_ydata(), mean, "Figure 7 mean")
        _band(axis.collections[i], np.arange(51), mean - half, mean + half, "Figure 7 t interval")
        if i:
            _same(inset.lines[i - 1].get_ydata(), (curves - study.accuracy_by_seed["iid"] * 100.0).mean(axis=0), "paired inset")
    if inset.get_title() != "Difference vs. IID (percentage points)":
        raise ValueError("Figure 7 approved title adapter mismatch")
    axis = figures["figure_8/fig-client-label-heterogeneity"].axes[0]
    order = ("iid", "pathological", "dir_0.6", "dir_0.3", "dir_0.1")
    if len(axis.collections) != 5 or len(axis.patches) != 5 or len(axis.lines) != 25:
        raise ValueError("Figure 8 display partition count")
    rng = np.random.default_rng(20260827)
    for i, p in enumerate(order, 1):
        selected = rng.choice(study.label_proportions[p], 1400, replace=False)
        expected = np.column_stack((i + rng.uniform(-0.18, 0.18, selected.size), selected))
        _same(axis.collections[i - 1].get_offsets(), expected, "deterministic client sample")
        # Independent full-data quartiles/Tukey whiskers, not the display sample.
        values = np.asarray(study.label_proportions[p])
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        iqr = q3 - q1
        lower = values[values >= q1 - 1.5 * iqr].min()
        upper = values[values <= q3 + 1.5 * iqr].max()
        lower, upper = min(lower, q1), max(upper, q3)
        _same(axis.patches[i - 1].get_path().vertices[:, 1], [q1, q1, q3, q3, q1, q1], "Figure 8 quartile box")
        expected_lines = ([q1, lower], [q3, upper], [lower, lower], [upper, upper], [median, median])
        for line, expected_y in zip(axis.lines[(i - 1) * 5:i * 5], expected_lines):
            _same(line.get_ydata(), expected_y, "Figure 8 whisker/cap/median")
    axis = figures["figure_9/fig-method-volcano"].axes[0]
    records = _volcano_records(data)
    if len(axis.collections) != 3:
        raise ValueError("Figure 9 category count")
    if len(axis.lines) != 2:
        raise ValueError("Figure 9 reference threshold count")
    _same(axis.lines[0].get_xdata(), [0, 0], "Figure 9 zero-effect threshold")
    _same(axis.lines[1].get_ydata(), [-np.log10(0.05)] * 2, "Figure 9 q=0.05 threshold")
    if axis.get_xlabel() != "Accuracy change vs. LT-ADMM-VR (percentage points)" or axis.get_ylabel() != "\u2212log10(BH-adjusted p-value)":
        raise ValueError("Figure 9 reference/statistical axis text")
    from matplotlib.text import Annotation
    annotations = [t for t in axis.texts if isinstance(t, Annotation)]
    requests = (("spade_dfl", "iid", "SPADE-DFL, IID"), ("spade_dfl", "dir_0.6", "SPADE-DFL, Dir. 0.6"), ("spade_dfl", "dir_0.3", "SPADE-DFL, Dir. 0.3"), ("spade_dfl", "pathological", "SPADE-DFL, Pathological"), ("one_point_dsg", "dir_0.3", "1P-DSG, Dir. 0.3"), ("zo2_dp_sgd", "dir_0.3", "DP-SGD, Dir. 0.3"))
    if len(annotations) != len(requests):
        raise ValueError("Figure 9 annotation count")
    for annotation, (method, partition, label) in zip(annotations, requests):
        selected = max((r for r in records if r.method == method and r.partition == partition and r.classification != "unresolved"), key=lambda r: r.minus_log10_q)
        if annotation.get_text() != label:
            raise ValueError("Figure 9 annotation identity")
        _same(annotation.xy, [selected.mean_paired_accuracy_difference_pp, selected.minus_log10_q], "Figure 9 annotated point")
    for collection, category in zip(axis.collections, ("worse", "unresolved", "better")):
        expected = [(r.mean_paired_accuracy_difference_pp, r.minus_log10_q) for r in records if r.classification == category]
        _same(collection.get_offsets(), expected, "Figure 9 points")
    from .communication.plotting import verify_figure
    verify_figure(figures["figure_10/fig_communication_saving_band"], _communication_rows(data))
    return {"figure_series_differences": 0}


def _save_options(stem):
    if stem.startswith("figure_6/"):
        return {"bbox_inches": None}
    return {"bbox_inches": "tight", "pad_inches": 0.02 if stem.startswith(("figure_7/", "figure_8/", "figure_9/", "figure_10/")) else 0, "facecolor": "white"}


def export_qa_rasters(fig, base):
    """Temporary author-QA only; raster files never enter the published tree."""
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(base) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(str(base) + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight", pad_inches=0.02, facecolor="white")


def _export_vectors(figures, directory):
    # Freeze the already-reviewed export context, including lazy AutoLocator
    # tick sizing. The communication helper's source hash is part of the frozen lineage.
    from .communication.plotting import configure_matplotlib
    with plt.rc_context():
        configure_matplotlib()
        _export_vectors_in_context(figures, directory)


def _export_vectors_in_context(figures, directory):
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": SVG_HASHSALT}):
        for stem, fig in figures.items():
            fig.canvas.draw()
            base = directory / stem
            base.parent.mkdir(parents=True, exist_ok=True)
            kwargs = _save_options(stem)
            fig.savefig(str(base) + ".pdf", metadata={"CreationDate": None, "ModDate": None}, **kwargs)
            fig.savefig(str(base) + ".svg", metadata={"Date": None}, **kwargs)


def _verify_rendered_figures(data, output):
    """Rebuild from frozen inputs; never trust the colocated output hash claims."""
    figures = {}
    try:
        with tempfile.TemporaryDirectory(prefix="paper-independent-verify-") as temporary:
            root = Path(temporary)
            with plt.rc_context(matplotlib.rcParamsDefault):
                figures = build_figures(data)
                validate_figures(figures, data)
                _export_vectors(figures, root)
            rendered = sorted(p for p in root.rglob("*") if p.is_file())
            if len(rendered) != 40:
                raise ValueError("independent rendered figure inventory mismatch")
            differences = [p.relative_to(root).as_posix() for p in rendered if sha256(p) != sha256(output / p.relative_to(root))]
            if differences:
                raise ValueError(f"independent rendered figure differences ({len(differences)}): {differences}")
    finally:
        for fig in figures.values():
            plt.close(fig)


def write_all_paper_outputs(reference_root, output):
    data = load_inputs(reference_root)
    tables = render_tables(data)
    validate_tables(tables)
    evidence = figure_evidence(data)
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be a new or empty directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".paper-output-", dir=output.parent)).resolve()
    figures = {}
    try:
        for name, content in tables.items():
            path = temporary / "tables" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(content.encode("utf-8"))
        for relative, (name, rows) in _source_csvs(data).items():
            path = temporary / relative
            path.parent.mkdir(exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=load_manifest()["inputs"][name]["columns"], lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        figures = build_figures(data)
        validate_figures(figures, data)
        _export_vectors(figures, temporary)
        for fig in figures.values():
            plt.close(fig)
        figures = {}
        manifest = load_manifest()
        files = {}
        for path in sorted(temporary.rglob("*")):
            if path.is_file():
                relative = path.relative_to(temporary).as_posix()
                files[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size, **_output_metadata(relative, data)}
        report = {"schema_version": 1, "status": "completed", "presentation_manifest_sha256": MANIFEST_SHA256, "reference_verification_sha256": manifest["verification_sha256"], "input_sha256": {n: r["sha256"] for n, r in manifest["inputs"].items()}, "renderer_lineage": manifest["renderer_lineage"], "adaptations": manifest["adaptations"], "evidence_boundaries": manifest["evidence_boundaries"], "semantics": manifest["semantics"], "counts": {"table_cells": 168, "table_numeric_scalars": 320, "figure_files": 40, "source_csvs": 9}, "figure_evidence": evidence, "verification": {"table_cell_differences": 0, "figure_series_differences": 0, "missing_outputs": 0}, "files": files}
        report["export"] = manifest["export"]
        (temporary / "PAPER_OUTPUT_MANIFEST.json").write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
        verify_paper_outputs(reference_root, temporary)
        # Re-read all frozen hashes immediately before publication, including reference verification.
        load_inputs(reference_root)
        if output.exists():
            output.rmdir()  # Only the already checked empty output directory.
        os.replace(temporary, output)
        return report
    finally:
        for fig in figures.values():
            plt.close(fig)
        if temporary.exists():
            if temporary.parent != output.parent or not temporary.name.startswith(".paper-output-"):
                raise RuntimeError("unsafe temporary cleanup target")
            shutil.rmtree(temporary)


def _canonical_rows(rows):
    def key(row):
        ranks = {"dataset": DATASETS, "partition": DISTRIBUTIONS, "method": METHODS + ("spade_dfl_private",), "suite": ("distribution", "method")}
        result = []
        for field in ("dataset", "suite", "partition", "method", "n", "epsilon", "seed", "client", "round", "T", "tau"):
            if field in row:
                result.append(ranks[field].index(row[field]) if field in ranks else int(row[field]))
        return tuple(result)
    return sorted(rows, key=key)


def _source_csvs(data):
    sources = {f"figure_{n}/main_roundwise.csv": ("main_roundwise.csv", [r for r in data["main_roundwise.csv"] if r["dataset"] == dataset]) for n, dataset in enumerate(DATASETS, 3)}
    sources.update({
        "figure_7/s15_roundwise.csv": ("s15_roundwise.csv", [r for r in data["s15_roundwise.csv"] if r["suite"] == "distribution"]),
        "figure_8/s15_client_proportions.csv": ("s15_client_proportions.csv", data["s15_client_proportions.csv"]),
        "figure_9/s15_roundwise.csv": ("s15_roundwise.csv", [r for r in data["s15_roundwise.csv"] if r["suite"] == "method"]),
        "figure_9/s15_wilcoxon_bh.csv": ("s15_wilcoxon_bh.csv", data["s15_wilcoxon_bh.csv"]),
        "figure_10/communication_summary.csv": ("communication_summary.csv", data["communication_summary.csv"]),
    })
    return {relative: (name, _canonical_rows(rows)) for relative, (name, rows) in sources.items()}


def _output_metadata(relative, data):
    manifest = load_manifest()
    directory = relative.split("/")[0]
    if directory == "tables":
        name = relative.split("/")[1]
        source = "main_final_accuracy.csv" if name == "table_II.tex" else "sensitivity_cells.csv"
        return {"inputs": {source: manifest["inputs"][source]["sha256"]}, "manuscript_sha256": manifest["manuscript_sha256"], "result_cells": manifest["tables"][name]["cells"]}
    number = int(directory.split("_")[1])
    source_names = ["main_roundwise.csv"] if number in (3, 4, 5, 6) else {7: ["s15_roundwise.csv"], 8: ["s15_client_proportions.csv"], 9: ["s15_roundwise.csv", "s15_wilcoxon_bh.csv"], 10: ["communication_summary.csv"]}[number]
    is_csv = relative.endswith(".csv")
    if is_csv:
        source_names = [relative.split("/")[1]]
    relevant = {n: manifest["inputs"][n]["sha256"] for n in source_names}
    lineage = {3: ["mnist", "mnist_styles"], 4: ["fashion", "fashion_styles"], 5: ["alzheimer", "alzheimer_styles", "alzheimer_order"], 6: ["brain", "brain_styles", "brain_order"], 7: ["distribution"], 8: ["distribution"], 9: ["volcano", "distribution"], 10: ["communication"]}[number]
    input_rows = {}
    for name in source_names:
        rows = data[name]
        if name == "main_roundwise.csv":
            count = sum(r["dataset"] == DATASETS[number - 3] for r in rows)
        elif name == "s15_roundwise.csv":
            suite = "distribution" if number == 7 else "method"
            count = sum(r["suite"] == suite for r in rows)
        else:
            count = len(rows)
        input_rows[name] = count
    return {"inputs": relevant, "renderer_sources": {k: manifest["renderer_lineage"][k]["sha256"] for k in lineage},
            "source_rows": sum(input_rows.values()), "input_rows_by_file": input_rows,
            "source_rows_semantics": "csv_data_rows" if is_csv else "combined_figure_input_rows"}


def verify_paper_outputs(reference_root, output):
    data = load_inputs(reference_root)
    output = Path(output)
    report = json.loads((output / "PAPER_OUTPUT_MANIFEST.json").read_text(encoding="utf-8"))
    frozen = load_manifest()
    for key in ("renderer_lineage", "adaptations", "evidence_boundaries", "semantics", "export"):
        if report[key] != frozen[key]:
            raise ValueError(f"output manifest {key} mismatch")
    if report["status"] != "completed" or report["presentation_manifest_sha256"] != MANIFEST_SHA256 or report["reference_verification_sha256"] != frozen["verification_sha256"] or report["input_sha256"] != {n: r["sha256"] for n, r in frozen["inputs"].items()}:
        raise ValueError("output provenance mismatch")
    if report["verification"] != {"table_cell_differences": 0, "figure_series_differences": 0, "missing_outputs": 0} or report["figure_evidence"] != figure_evidence(data):
        raise ValueError("output verification evidence mismatch")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} - {"PAPER_OUTPUT_MANIFEST.json"}
    expected = set(_source_csvs(data)) | {"tables/" + n for n in frozen["tables"]}
    expected |= {f"figure_{n}/{prefix}-{suffix}.{extension}" for n, prefix in zip(range(3, 7), ("MNIST", "Fashion MNIST", "Alzheimer", "Brain Tumor")) for suffix in ("iid", "dir 0.6", "dir 0.3", "pathological") for extension in ("pdf", "svg")}
    expected |= {f"figure_{n}/{stem}.{extension}" for n, stem in ((7, "fig-distribution-accuracy"), (8, "fig-client-label-heterogeneity"), (9, "fig-method-volcano"), (10, "fig_communication_saving_band")) for extension in ("pdf", "svg")}
    if actual != expected or set(report["files"]) != expected:
        raise ValueError("missing/extra published outputs")
    for relative, record in report["files"].items():
        path = output / relative
        if sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"output hash mismatch: {relative}")
        if {k: v for k, v in record.items() if k not in ("sha256", "bytes")} != _output_metadata(relative, data):
            raise ValueError("output mapping mismatch")
    validate_tables({name: (output / "tables" / name).read_text(encoding="utf-8") for name in frozen["tables"]})
    for relative, (name, rows) in _source_csvs(data).items():
        with (output / relative).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != frozen["inputs"][name]["columns"] or list(reader) != rows:
                raise ValueError("published source CSV mismatch")
    # Zero figure differences is returned only after this independent 40-file check.
    _verify_rendered_figures(data, output)
    return report


# Source-extracted presentation functions; lineage and adaptations are locked in the manifest.
def _mnist_renderer():
    @dataclass(frozen=True)
    class MethodSpec:
        label: str
        color: str
        linestyle: str = '-'
        linewidth: float = 1.5
        marker: str = ''

    METHOD_REGISTRY: Dict[str, MethodSpec] = {'spade_dfl': MethodSpec('SPADE-DFL', '#ef5350', '-', 2.5, 'v'), 'one_point_dsg': MethodSpec('1P-DSG', '#ffb74d'), 'one_point_dsgt': MethodSpec('1P-DSGT', '#ff8a65'), 'zo2_lt_admm_vr': MethodSpec('2P-LT-ADMM-VR', '#3182bd', '-', 1.7, 'o'), 'zo2_lt_admm': MethodSpec('2P-LT-ADMM', '#6baed6', '--', 1.7, 's'), 'zo2_dp_sgd': MethodSpec('2P-DP-SGD', '#31a354', '-.', 1.7, '^'), 'zo2_dp_fedavg': MethodSpec('2P-DP-FedAvg', '#74c476', '--', 1.7, 'D')}

    BLACK_BOX_LEGEND_ORDER = ('LT-ADMM-VR', 'LT-ADMM', 'DP-SGD', 'DP-FedAvg', '1P-DSG', '1P-DSGT', 'SPADE-DFL')

    BLACK_BOX_LEGEND_LABELS = {'2P-LT-ADMM-VR': 'LT-ADMM-VR', '2P-LT-ADMM': 'LT-ADMM', '2P-DP-SGD': 'DP-SGD', '2P-DP-FedAvg': 'DP-FedAvg', '1P-DSG': '1P-DSG', '1P-DSGT': '1P-DSGT', 'SPADE-DFL': 'SPADE-DFL'}

    PANEL_TITLES = {'dir_0.3': 'Non-IID: Dir 0.3', 'dir_0.6': 'Non-IID: Dir 0.6', 'iid': 'IID', 'pathological': 'Pathological'}

    def plot_black_box_accuracy(aggregates: Mapping[str, Mapping[str, np.ndarray]], output_base: Path, *, seed_count: int, partition: str):
        """Export the primary all-black-box accuracy comparison and its caption data.

        The outer-round axis is a communication round for decentralized methods.  For
        centralized DP-SGD it is an optimization step, which is recorded explicitly in
        the caption sidecar instead of being presented as a communication claim.
        """
        if not aggregates:
            raise ValueError('accuracy plot requires at least one method')
        if seed_count < 1:
            raise ValueError('seed_count must be positive')
        style_by_label = {spec.label: spec for spec in METHOD_REGISTRY.values()}
        missing_styles = sorted(set(aggregates) - set(style_by_label))
        if missing_styles:
            raise ValueError('missing registered plot style: ' + ', '.join(missing_styles))
        lengths = set()
        prepared = {}
        for (label, aggregate) in aggregates.items():
            mean = np.asarray(aggregate['accuracy_mean'], dtype=np.float64)
            std = np.asarray(aggregate['accuracy_std'], dtype=np.float64)
            if mean.ndim != 1 or std.shape != mean.shape or mean.size == 0:
                raise ValueError(f'invalid accuracy aggregate for {label}')
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
                raise ValueError(f'non-finite accuracy aggregate for {label}')
            lengths.add(mean.size)
            prepared[label] = (mean, std)
        if len(lengths) != 1:
            raise ValueError('all accuracy curves must have the same number of rounds')
        rc = {'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}
        with plt.rc_context(rc):
            (fig, axis) = plt.subplots(figsize=(2.4, 1.8))
            lower_envelope = []
            upper_envelope = []
            plot_order = sorted(prepared, key=lambda label: (BLACK_BOX_LEGEND_ORDER.index(BLACK_BOX_LEGEND_LABELS.get(label, label)) if BLACK_BOX_LEGEND_LABELS.get(label, label) in BLACK_BOX_LEGEND_ORDER else len(BLACK_BOX_LEGEND_ORDER), label))
            for label in plot_order:
                (mean, std) = prepared[label]
                spec = style_by_label[label]
                x = np.arange(mean.size)
                marker_every = max(1, mean.size // 10)
                display_label = BLACK_BOX_LEGEND_LABELS.get(label, label)
                axis.plot(x, mean, label=display_label, color=spec.color, linestyle=spec.linestyle, linewidth=1.8 if display_label == 'SPADE-DFL' else 1.1, marker=spec.marker, markevery=marker_every, markersize=2.8, markeredgewidth=0.5, zorder=4 if display_label == 'SPADE-DFL' else 3)
                axis.fill_between(x, mean - std, mean + std, color=spec.color, alpha=0.14, linewidth=0, zorder=1)
                lower_envelope.append(float(np.min(mean - std)))
                upper_envelope.append(float(np.max(mean + std)))
            panel_title = PANEL_TITLES.get(partition, partition)
            final_round = next(iter(lengths)) - 1
            ymin = 30.0
            ymax = 5.0 * np.ceil(max(upper_envelope) / 5.0)
            axis.text(0.03, 0.04, panel_title, transform=axis.transAxes, ha='left', va='bottom', fontsize=8, zorder=10)
            axis.set_xlabel('Rounds')
            axis.set_ylabel('Accuracy (%)')
            axis.set_xlim((0, final_round) if final_round else (-0.5, 0.5))
            axis.set_xticks(np.arange(0, final_round + 1, 10))
            axis.set_ylim(ymin, ymax)
            axis.yaxis.set_major_locator(mticker.MultipleLocator(5))
            axis.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.grid(axis='y', color='#d9d9d9', linewidth=0.5, alpha=0.65)
            if partition == 'iid':
                inset = axis.inset_axes([0.15, 0.2, 0.4, 0.35])
                (x0, x1) = (5, 10)
                (y0, y1) = (75, 85)
                for label in plot_order:
                    (mean, std) = prepared[label]
                    spec = style_by_label[label]
                    x = np.arange(mean.size)
                    display_label = BLACK_BOX_LEGEND_LABELS.get(label, label)
                    inset.plot(x, mean, color=spec.color, linestyle=spec.linestyle, linewidth=1.4 if display_label == 'SPADE-DFL' else 0.9, marker=spec.marker, markevery=max(1, mean.size // 10), markersize=1.8, markeredgewidth=0.4, zorder=4 if display_label == 'SPADE-DFL' else 3)
                    inset.fill_between(x, mean - std, mean + std, color=spec.color, alpha=0.12, linewidth=0, zorder=1)
                inset.set_xlim(x0, x1)
                inset.set_ylim(y0, y1)
                inset.set_xticks([5, 7.5, 10])
                inset.set_yticks([75, 80, 85])
                inset.tick_params(labelsize=5, length=2, pad=1)
                inset.grid(axis='y', color='#d9d9d9', linewidth=0.35, alpha=0.65)
                inset.spines['top'].set_visible(False)
                inset.spines['right'].set_visible(False)
                axis.indicate_inset_zoom(inset, edgecolor='#555555', linewidth=0.6)
            axis.legend(loc='lower right', ncol=1, fontsize=5, frameon=True, framealpha=0.94, edgecolor='#cccccc')
            fig.tight_layout(pad=0.02)
            return fig
    return plot_black_box_accuracy


def _fashion_renderer():
    @dataclass(frozen=True)
    class MethodSpec:
        label: str
        color: str
        linestyle: str = '-'
        linewidth: float = 1.5
        marker: str = ''

    METHOD_REGISTRY: Dict[str, MethodSpec] = {'spade_dfl': MethodSpec('SPADE-DFL', '#ef5350', '-', 2.5, 'v'), 'one_point_dsg': MethodSpec('1P-DSG', '#ffb74d'), 'one_point_dsgt': MethodSpec('1P-DSGT', '#ff8a65'), 'zo2_lt_admm_vr': MethodSpec('2P-LT-ADMM-VR', '#3182bd', '-', 1.7, 'o'), 'zo2_lt_admm': MethodSpec('2P-LT-ADMM', '#6baed6', '--', 1.7, 's'), 'zo2_dp_sgd': MethodSpec('2P-DP-SGD', '#31a354', '-.', 1.7, '^'), 'zo2_dp_fedavg': MethodSpec('2P-DP-FedAvg', '#74c476', '--', 1.7, 'D')}

    BLACK_BOX_LEGEND_ORDER = ('LT-ADMM-VR', 'LT-ADMM', 'DP-SGD', 'DP-FedAvg', '1P-DSG', '1P-DSGT', 'SPADE-DFL')

    BLACK_BOX_LEGEND_LABELS = {'2P-LT-ADMM-VR': 'LT-ADMM-VR', '2P-LT-ADMM': 'LT-ADMM', '2P-DP-SGD': 'DP-SGD', '2P-DP-FedAvg': 'DP-FedAvg', '1P-DSG': '1P-DSG', '1P-DSGT': '1P-DSGT', 'SPADE-DFL': 'SPADE-DFL'}

    PANEL_TITLES = {'dir_0.3': 'Non-IID: Dir 0.3', 'dir_0.6': 'Non-IID: Dir 0.6', 'iid': 'IID', 'pathological': 'Pathological'}

    def plot_black_box_accuracy(aggregates: Mapping[str, Mapping[str, np.ndarray]], output_base: Path, *, seed_count: int, partition: str, dataset_metadata: Optional[Mapping[str, object]]=None, dataset_name: Optional[str]=None, negative_class: Optional[int]=None, negative_class_name: Optional[str]=None, positive_class: Optional[int]=None, positive_class_name: Optional[str]=None):
        """Export the primary all-black-box accuracy comparison and its caption data.

        The outer-round axis is a communication round for decentralized methods.  For
        centralized DP-SGD it is an optimization step, which is recorded explicitly in
        the caption sidecar instead of being presented as a communication claim.
        """
        if not aggregates:
            raise ValueError('accuracy plot requires at least one method')
        if seed_count < 1:
            raise ValueError('seed_count must be positive')
        style_by_label = {spec.label: spec for spec in METHOD_REGISTRY.values()}
        missing_styles = sorted(set(aggregates) - set(style_by_label))
        if missing_styles:
            raise ValueError('missing registered plot style: ' + ', '.join(missing_styles))
        lengths = set()
        prepared = {}
        for (label, aggregate) in aggregates.items():
            mean = np.asarray(aggregate['accuracy_mean'], dtype=np.float64)
            std = np.asarray(aggregate['accuracy_std'], dtype=np.float64)
            if mean.ndim != 1 or std.shape != mean.shape or mean.size == 0:
                raise ValueError(f'invalid accuracy aggregate for {label}')
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
                raise ValueError(f'non-finite accuracy aggregate for {label}')
            lengths.add(mean.size)
            prepared[label] = (mean, std)
        if len(lengths) != 1:
            raise ValueError('all accuracy curves must have the same number of rounds')
        rc = {'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}
        with plt.rc_context(rc):
            (fig, axis) = plt.subplots(figsize=(2.4, 1.8))
            lower_envelope = []
            upper_envelope = []
            plot_order = sorted(prepared, key=lambda label: (BLACK_BOX_LEGEND_ORDER.index(BLACK_BOX_LEGEND_LABELS.get(label, label)) if BLACK_BOX_LEGEND_LABELS.get(label, label) in BLACK_BOX_LEGEND_ORDER else len(BLACK_BOX_LEGEND_ORDER), label))
            for label in plot_order:
                (mean, std) = prepared[label]
                spec = style_by_label[label]
                x = np.arange(mean.size)
                marker_every = max(1, mean.size // 10)
                display_label = BLACK_BOX_LEGEND_LABELS.get(label, label)
                linewidth = 1.8 if display_label == 'SPADE-DFL' else 1.1
                markersize = 2.8
                axis.plot(x, mean, label=display_label, color=spec.color, linestyle=spec.linestyle, linewidth=linewidth, marker=spec.marker, markevery=marker_every, markersize=markersize, markeredgewidth=0.5, zorder=4 if display_label == 'SPADE-DFL' else 3)
                axis.fill_between(x, mean - std, mean + std, color=spec.color, alpha=0.14, linewidth=0, zorder=1)
                lower_envelope.append(float(np.min(mean - std)))
                upper_envelope.append(float(np.max(mean + std)))
            panel_title = PANEL_TITLES.get(partition, partition)
            final_round = next(iter(lengths)) - 1
            ymin = 5.0 * np.floor(min(lower_envelope) / 5.0)
            ymax = 5.0 * np.ceil(max(upper_envelope) / 5.0)
            axis.text(0.03, 0.04, panel_title, transform=axis.transAxes, ha='left', va='bottom', fontsize=8, zorder=10)
            axis.set_xlabel('Rounds')
            axis.set_ylabel('Accuracy (%)')
            axis.set_xlim((0, final_round) if final_round else (-0.5, 0.5))
            axis.set_xticks(np.arange(0, final_round + 1, 10))
            axis.set_ylim(30, 95)
            axis.yaxis.set_major_locator(mticker.MultipleLocator(5))
            axis.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.grid(axis='y', color='#d9d9d9', linewidth=0.5, alpha=0.65)
            axis.legend(loc='lower right', ncol=1, fontsize=5, frameon=True, framealpha=0.94, edgecolor='#cccccc')
            fig.tight_layout(pad=0.02)
            return fig
    return plot_black_box_accuracy


def _render_alzheimer(aggregates):
    ALZHEIMER_PARTITIONS = ('dir_0.3', 'dir_0.6', 'iid', 'pathological')

    FIGURE_STYLES = {'zo2_lt_admm_vr': ('#2166ac', '-', 'o'), 'zo2_lt_admm': ('#67a9cf', '--', 's'), 'zo2_dp_sgd': ('#1b9e77', '-.', '^'), 'zo2_dp_fedavg': ('#66c2a5', '--', 'D'), 'one_point_dsg': ('#e6ab02', '-', 'P'), 'one_point_dsgt': ('#f46d43', '-.', 'X'), 'spade_dfl': ('#d73027', '-', 'v')}

    DISPLAY_LABELS = {'zo2_lt_admm_vr': 'LT-ADMM-VR', 'zo2_lt_admm': 'LT-ADMM', 'zo2_dp_sgd': 'DP-SGD', 'zo2_dp_fedavg': 'DP-FedAvg', 'one_point_dsg': '1P-DSG', 'one_point_dsgt': '1P-DSGT', 'spade_dfl': 'SPADE-DFL'}

    ALZHEIMER_METHOD_IDS = ('zo2_lt_admm_vr', 'zo2_lt_admm', 'zo2_dp_sgd', 'zo2_dp_fedavg', 'one_point_dsg', 'one_point_dsgt', 'spade_dfl')

    titles = {'dir_0.3': 'Non-IID: Dir 0.3', 'dir_0.6': 'Non-IID: Dir 0.6', 'iid': 'IID', 'pathological': 'Pathological'}

    lower_values = [np.min(a['accuracy_mean'] - a['accuracy_std']) for panel in aggregates.values() for a in panel.values()]

    ymin = 5.0 * np.floor(min(lower_values) / 5.0)

    ymax = 80

    rc = {'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}

    figures = {}

    with plt.rc_context(rc):
        for partition in ALZHEIMER_PARTITIONS:
            (fig, axis) = plt.subplots(figsize=(2.4, 1.8))
            for method in ALZHEIMER_METHOD_IDS:
                data = aggregates[partition][method]
                rounds = data['round']
                mean = data['accuracy_mean']
                std = data['accuracy_std']
                (color, linestyle, marker) = FIGURE_STYLES[method]
                axis.plot(rounds, mean, color=color, linestyle=linestyle, marker=marker, markevery=max(1, len(rounds) // 10), linewidth=1.8 if method == 'spade_dfl' else 1.1, markersize=2.8, label=DISPLAY_LABELS[method])
                axis.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
            axis.text(0.03, 0.04, titles[partition], transform=axis.transAxes, ha='left', va='bottom', fontsize=8, zorder=10)
            axis.set_xlabel('Rounds')
            axis.set_ylabel('Accuracy (%)')
            axis.set_xlim(0, int(np.max(rounds)))
            axis.set_xticks(np.arange(0, int(np.max(rounds)) + 1, 10))
            axis.set_ylim(ymin, ymax)
            axis.yaxis.set_major_locator(MultipleLocator(5))
            axis.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.grid(axis='y', color='#d9d9d9', linewidth=0.5, alpha=0.65)
            axis.legend(loc='lower right', ncol=1, fontsize=5, frameon=True, framealpha=0.94, edgecolor='#cccccc')
            fig.tight_layout(pad=0.02)
            figures[partition] = fig
    return figures


def _brain_renderer():
    FIGURE_STYLES = {'zo2_lt_admm_vr': ('#2166ac', '-', 'o'), 'zo2_lt_admm': ('#67a9cf', '--', 's'), 'zo2_dp_sgd': ('#1b9e77', '-.', '^'), 'zo2_dp_fedavg': ('#66c2a5', '--', 'D'), 'one_point_dsg': ('#e6ab02', '-', 'P'), 'one_point_dsgt': ('#f46d43', '-.', 'X'), 'spade_dfl': ('#d73027', '-', 'v')}

    VISIBLE_METHOD_LABELS = {'zo2_lt_admm_vr': 'LT-ADMM-VR', 'zo2_lt_admm': 'LT-ADMM', 'zo2_dp_sgd': 'DP-SGD', 'zo2_dp_fedavg': 'DP-FedAvg', 'one_point_dsg': '1P-DSG', 'one_point_dsgt': '1P-DSGT', 'spade_dfl': 'SPADE-DFL'}

    BRAIN_TUMOR_METHOD_IDS = ('zo2_lt_admm_vr', 'zo2_lt_admm', 'zo2_dp_sgd', 'zo2_dp_fedavg', 'one_point_dsg', 'one_point_dsgt', 'spade_dfl')

    FORMAL_PANEL_ORDER = ('iid', 'dir_0.6', 'dir_0.3', 'pathological')

    FORMAL_PANEL_LABELS = {'iid': 'IID', 'dir_0.6': 'Non-IID: Dir 0.6', 'dir_0.3': 'Non-IID: Dir 0.3', 'pathological': 'Pathological'}

    def formal_accuracy_ylim_percent(aggregates: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]) -> tuple[float, float]:
        """Keep the original 15--60% scale, extending only in five-point steps."""
        upper_observed = max((float(np.max(np.asarray(aggregates[panel][method]['accuracy_mean']) + np.asarray(aggregates[panel][method]['accuracy_std']))) for panel in FORMAL_PANEL_ORDER for method in BRAIN_TUMOR_METHOD_IDS))
        upper = max(60.0, 5.0 * math.ceil(upper_observed / 5.0))
        return (15.0, upper)

    def write_formal_accuracy_figures(aggregates: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]], output_dir: Path, *, n_rounds: int) -> dict[str, tuple[Path, ...]]:
        """Write the four original-format Fig. 4 panels from unsmoothed aggregates."""
        y_limits = formal_accuracy_ylim_percent(aggregates)
        marker_every = max(1, int(n_rounds) // 10)
        x_ticks = np.arange(0, int(n_rounds) + 1, 10)
        if int(n_rounds) < 10:
            x_ticks = np.arange(0, int(n_rounds) + 1, 1)
        settings = {'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 6.5, 'axes.labelsize': 7.0, 'xtick.labelsize': 6.0, 'ytick.labelsize': 6.0, 'legend.fontsize': 5.0, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'}
        outputs: dict[str, tuple[Path, ...]] = {}
        with plt.rc_context(settings):
            for panel in FORMAL_PANEL_ORDER:
                (figure, axis) = plt.subplots(figsize=(2.39, 1.79))
                for method in BRAIN_TUMOR_METHOD_IDS:
                    aggregate = aggregates[panel][method]
                    rounds = np.asarray(aggregate['round'], dtype=np.float64)
                    mean = np.asarray(aggregate['accuracy_mean'], dtype=np.float64)
                    std = np.asarray(aggregate['accuracy_std'], dtype=np.float64)
                    if not rounds.shape == mean.shape == std.shape:
                        raise ValueError(f'inconsistent aggregate shapes for {panel}/{method}')
                    (color, linestyle, marker) = FIGURE_STYLES[method]
                    axis.plot(rounds, mean, label=VISIBLE_METHOD_LABELS[method], color=color, linestyle=linestyle, marker=marker, markevery=marker_every, linewidth=1.05, markersize=2.6, markeredgewidth=0.35, zorder=4 if method == 'spade_dfl' else 3)
                    axis.fill_between(rounds, np.clip(mean - std, 0.0, 100.0), np.clip(mean + std, 0.0, 100.0), color=color, alpha=0.12, linewidth=0.0, zorder=1)
                axis.set_xlim(0, int(n_rounds))
                axis.set_xticks(x_ticks)
                axis.set_ylim(*y_limits)
                axis.yaxis.set_major_locator(MultipleLocator(5.0))
                axis.set_xlabel('Rounds')
                axis.set_ylabel('Accuracy (%)')
                axis.grid(True, color='#c7c7c7', alpha=0.45, linewidth=0.35)
                axis.text(0.02, 0.035, FORMAL_PANEL_LABELS[panel], transform=axis.transAxes, ha='left', va='bottom', fontsize=6.5)
                axis.legend(loc='lower right', frameon=True, framealpha=0.88, borderpad=0.25, labelspacing=0.2, handlelength=1.6, handletextpad=0.35)
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
                axis.tick_params(width=0.55, length=2.6)
                figure.subplots_adjust(left=0.18, right=0.985, bottom=0.225, top=0.97)
                outputs[panel] = figure
        return outputs
    return write_formal_accuracy_figures


def _distribution_renderers():
    PARTITION_ORDER = ('iid', 'dir_0.6', 'dir_0.3', 'pathological', 'dir_0.1')

    HETEROGENEITY_ORDER = ('iid', 'pathological', 'dir_0.6', 'dir_0.3', 'dir_0.1')

    LABELS = {'iid': 'IID', 'dir_0.6': 'Dir. 0.6', 'dir_0.3': 'Dir. 0.3', 'pathological': 'Mixed Path.', 'dir_0.1': 'Dir. 0.1'}

    COLORS = {'iid': '#3182bd', 'dir_0.6': '#6baed6', 'dir_0.3': '#ffb74d', 'pathological': '#ef5350', 'dir_0.1': '#756bb1'}

    MARKERS = {'iid': 'o', 'dir_0.6': 's', 'dir_0.3': '^', 'pathological': 'D', 'dir_0.1': 'P'}

    LINESTYLES = {'iid': '-', 'dir_0.6': '--', 'dir_0.3': '-.', 'pathological': ':', 'dir_0.1': (0, (5, 1.4))}

    def _plot_context():
        return plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 7.0, 'axes.labelsize': 8.0, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

    def _mean_t_ci(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = curves.mean(axis=0)
        standard_error = curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])
        half_width = float(stats.t.ppf(0.975, curves.shape[0] - 1)) * standard_error
        return (mean, mean - half_width, mean + half_width)

    def _round_count(study: DistributionStudy) -> int:
        return int(np.asarray(study.accuracy_by_seed['iid']).shape[1])

    def _validate(study: DistributionStudy) -> None:
        if tuple(study.partitions) != PARTITION_ORDER:
            raise ValueError(f'study partitions must be ordered as {PARTITION_ORDER}')
        if len(study.seeds) < 2:
            raise ValueError('at least two paired seeds are required')
        shape: tuple[int, int] | None = None
        for partition in PARTITION_ORDER:
            curves = np.asarray(study.accuracy_by_seed[partition], dtype=float)
            if curves.ndim != 2 or curves.shape[0] != len(study.seeds) or (not np.all(np.isfinite(curves))):
                raise ValueError(f'invalid seed-by-round accuracy for {partition}')
            if shape is None:
                shape = curves.shape
            elif curves.shape != shape:
                raise ValueError('all partitions must share aligned seed-by-round curves')
        if shape is None or shape[1] != 51:
            raise ValueError('the preview requires round 0 through round 50')

    def empirical_heterogeneity_order(study: DistributionStudy) -> tuple[str, ...]:
        """Order partitions by observed mean binary-label skew, low to high."""
        _validate(study)
        original_position = {partition: index for (index, partition) in enumerate(PARTITION_ORDER)}
        scores = {partition: float(1.0 - np.mean(np.asarray(study.label_entropy[partition], dtype=float))) for partition in PARTITION_ORDER}
        measured_order = tuple(sorted(PARTITION_ORDER, key=lambda p: (scores[p], original_position[p])))
        if measured_order != HETEROGENEITY_ORDER:
            raise ValueError(f'measured client-label skew contradicts the approved low-to-high heterogeneity order: {measured_order}')
        return HETEROGENEITY_ORDER

    def plot_accuracy_with_iid_inset(study: DistributionStudy, output_base: Path):
        """Plot five strong-DP SPADE curves and paired differences from IID."""
        _validate(study)
        rounds = np.arange(_round_count(study), dtype=float)
        iid = np.asarray(study.accuracy_by_seed['iid'], dtype=float) * 100.0
        source: dict[str, object] = {}
        visible_lower: list[float] = []
        visible_upper: list[float] = []
        with _plot_context():
            (figure, axis) = plt.subplots(figsize=(3.5, 2.65))
            inset = figure.add_axes([0.28, 0.55, 0.38, 0.29])
            for (index, partition) in enumerate(PARTITION_ORDER):
                curves = np.asarray(study.accuracy_by_seed[partition], dtype=float) * 100.0
                (mean, lower, upper) = _mean_t_ci(curves)
                visible_lower.append(float(np.min(lower[1:])))
                visible_upper.append(float(np.max(upper[1:])))
                axis.fill_between(rounds, lower, upper, color=COLORS[partition], alpha=0.05, linewidth=0)
                axis.plot(rounds, mean, color=COLORS[partition], linestyle=LINESTYLES[partition], marker=MARKERS[partition], markevery=(index, 10), markersize=2.8, linewidth=1.1, label=LABELS[partition])
                paired = curves - iid
                if partition != 'iid':
                    inset.plot(rounds, paired.mean(axis=0), color=COLORS[partition], linestyle=LINESTYLES[partition], linewidth=0.8)
                source[partition] = {'accuracy_by_seed_percent': curves.tolist(), 'mean_percent': mean.tolist(), 'confidence_interval_95_lower_percent': lower.tolist(), 'confidence_interval_95_upper_percent': upper.tolist(), 'paired_difference_vs_iid_pp': paired.tolist()}
            y_min = max(0.0, float(np.floor(min(visible_lower) - 0.5)))
            y_max = min(100.0, float(np.ceil(max(visible_upper) + 0.5)))
            axis.set_xlim(0, int(rounds[-1]))
            axis.set_ylim(y_min, max(y_min + 2.0, y_max))
            axis.set_xlabel('Rounds')
            axis.set_ylabel('Accuracy (%)')
            axis.grid(color='#dddddd', linewidth=0.4, alpha=0.55)
            axis.legend(loc='lower right', ncol=2, fontsize=5.0, title='mean ± 95% CI', title_fontsize=5.3)
            inset.axhline(0.0, color='#666666', linestyle='--', linewidth=0.6)
            inset.set_xlim(0, int(rounds[-1]))
            inset.set_title('Difference vs. IID (percentage points)', fontsize=5.5)
            inset.tick_params(labelsize=5.0)
            inset.grid(color='#e5e5e5', linewidth=0.3, alpha=0.45)
            figure.subplots_adjust(left=0.16, right=0.99, bottom=0.18, top=0.98)
            return figure

    def plot_label_heterogeneity(study: DistributionStudy, output_base: Path):
        """Plot the empirical client-level positive-label fractions."""
        _validate(study)
        empirical_order = empirical_heterogeneity_order(study)
        mean_label_skew = {partition: float(1.0 - np.mean(np.asarray(study.label_entropy[partition], dtype=float))) for partition in PARTITION_ORDER}
        rng = np.random.default_rng(20260827)
        with _plot_context():
            (figure, axis) = plt.subplots(figsize=(3.5, 2.65))
            positions = np.arange(1, len(empirical_order) + 1)
            values = [np.asarray(study.label_proportions[p], dtype=float) for p in empirical_order]
            boxes = axis.boxplot(values, positions=positions, widths=0.55, patch_artist=True, showfliers=False, medianprops={'color': '#333333', 'linewidth': 0.9}, whiskerprops={'color': '#777777', 'linewidth': 0.65}, capprops={'color': '#777777', 'linewidth': 0.65})
            for (patch, partition) in zip(boxes['boxes'], empirical_order):
                patch.set_facecolor(COLORS[partition])
                patch.set_alpha(0.22)
                patch.set_edgecolor(COLORS[partition])
            for (position, partition, observed) in zip(positions, empirical_order, values):
                select = observed if observed.size <= 1400 else rng.choice(observed, 1400, replace=False)
                axis.scatter(position + rng.uniform(-0.18, 0.18, select.size), select, s=3.2, color=COLORS[partition], alpha=0.13, linewidths=0)
            axis.set_xticks(positions, [LABELS[p] for p in empirical_order], rotation=28, ha='right')
            axis.set_ylim(-0.03, 1.03)
            axis.set_ylabel('Client p(y=7)')
            axis.set_xlabel('Client-data distribution')
            axis.grid(axis='y', color='#dddddd', linewidth=0.4, alpha=0.55)
            figure.subplots_adjust(left=0.17, right=0.99, bottom=0.33, top=0.98)
            return figure
    return (plot_accuracy_with_iid_inset, plot_label_heterogeneity)


def _volcano_renderer():
    def _plot_context():
        return plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 7.0, 'axes.labelsize': 8.0, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

    METHOD_LABELS = {'zo2_lt_admm_vr': 'LT-ADMM-VR', 'zo2_lt_admm': 'LT-ADMM', 'zo2_dp_sgd': 'DP-SGD', 'zo2_dp_fedavg': 'DP-FedAvg', 'one_point_dsg': '1P-DSG', 'one_point_dsgt': '1P-DSGT', 'spade_dfl': 'SPADE-DFL'}

    PARTITION_LABELS = {'iid': 'IID', 'dir_0.6': 'Dir. 0.6', 'dir_0.3': 'Dir. 0.3', 'pathological': 'Pathological'}

    def _annotate(axis: plt.Axes, records: tuple[MethodVolcanoRecord, ...], point_y_max: float) -> None:
        requests = (('spade_dfl', 'iid'), ('spade_dfl', 'dir_0.6'), ('spade_dfl', 'dir_0.3'), ('spade_dfl', 'pathological'), ('one_point_dsg', 'dir_0.3'), ('zo2_dp_sgd', 'dir_0.3'))
        chosen = []
        for (method, partition) in requests:
            subset = [record for record in records if record.method == method and record.partition == partition and (record.classification != 'unresolved')]
            if subset:
                chosen.append(max(subset, key=lambda record: record.minus_log10_q))
        label_specs = {('spade_dfl', 'pathological'): (0.68, 0.82, 'left', 0.1), ('spade_dfl', 'dir_0.3'): (0.78, 0.77, 'left', 0.06), ('spade_dfl', 'dir_0.6'): (0.88, 0.72, 'left', 0.03), ('spade_dfl', 'iid'): (0.98, 0.67, 'right', -0.05), ('one_point_dsg', 'dir_0.3'): (0.15, 0.76, 'left', -0.08), ('zo2_dp_sgd', 'dir_0.3'): (0.15, 0.7, 'left', -0.05)}
        for record in chosen:
            key = (record.method, record.partition)
            if key not in label_specs:
                continue
            (text_x, text_y, align, rad) = label_specs[key]
            is_left = record.mean_paired_accuracy_difference_pp < 0.0
            color = '#3561cc' if is_left else '#ef3b35'
            axis.annotate(f'{METHOD_LABELS[record.method]}, {PARTITION_LABELS[record.partition]}', xy=(record.mean_paired_accuracy_difference_pp, record.minus_log10_q), xytext=(text_x, text_y), xycoords='data', textcoords='axes fraction', ha=align, va='center', fontsize=5.0, color='#333333', bbox={'boxstyle': 'round,pad=0.12', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.72}, arrowprops={'arrowstyle': '-', 'color': color, 'linewidth': 0.45, 'alpha': 0.82, 'shrinkA': 2.0, 'shrinkB': 2.0, 'connectionstyle': f'arc3,rad={rad}'})

    def plot_method_volcano(records):
        """Render a screenshot-like, statistically defined method volcano."""
        categories = (('worse', 'Worse', '#3561cc'), ('unresolved', 'Not resolved', '#bdbdbd'), ('better', 'Better', '#ef3b35'))
        with _plot_context():
            (figure, axis) = plt.subplots(figsize=(3.5, 2.65))
            axis.axvline(0.0, color='#666666', linestyle='--', linewidth=0.7, zorder=1)
            axis.axhline(-np.log10(0.05), color='#666666', linestyle='--', linewidth=0.7, zorder=1)
            for (classification, label, color) in categories:
                selected = [record for record in records if record.classification == classification]
                axis.scatter([record.mean_paired_accuracy_difference_pp for record in selected], [record.minus_log10_q for record in selected], s=8.5, color=color, alpha=0.55 if classification != 'unresolved' else 0.42, linewidths=0, label=label, zorder=2)
            point_y_max = max((record.minus_log10_q for record in records))
            axis.set_ylim(bottom=-0.35, top=point_y_max + 2.15)
            _annotate(axis, records, point_y_max)
            axis.set_xlabel('Accuracy change vs. LT-ADMM-VR (percentage points)')
            axis.set_ylabel('−log10(BH-adjusted p-value)')
            axis.grid(color='#dddddd', linewidth=0.4, alpha=0.48)
            axis.legend(loc='upper center', ncol=3, fontsize=5.6, handletextpad=0.35, bbox_to_anchor=(0.5, 1.005))
            axis.text(0.98, 0.035, 'n=30 paired seeds; 1200 comparisons', transform=axis.transAxes, ha='right', va='bottom', fontsize=5.1, color='#555555')
            figure.subplots_adjust(left=0.19, right=0.99, bottom=0.21, top=0.98)
            return figure
    return plot_method_volcano
