"""Frozen manuscript output contracts; no training is permitted here."""
import copy
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

from spade_dfl_release import paper_outputs as p


def test_historical_manuscript_identity_is_reported_from_frozen_release(reference_root):
    expected = "4eafcd1bfd5ebb609f37d056fe0705589495d83c5abefcbeec097737dab7a9ab"
    data = p.load_inputs(reference_root)
    metadata = p._output_metadata("tables/table_II.tex", data)
    assert metadata["manuscript_sha256"] == expected
    assert p.load_manifest()["verification_sha256"] == "6ca441d19e25d5e78db6edae61c9a12b47d2c20059f681571287919246d61fdb"
    assert p.load_manifest()["export"]["svg_hashsalt"] == p.SVG_HASHSALT == "a8046ac75e26ea21eacba27cc77d75301d81d8be9af765c9de2952f1ad523829"
    verification = json.loads((reference_root / "paper_values/verification.json").read_text(encoding="utf-8"))
    assert verification["source_hashes"]["paper_tex/spade_dfl_v1.tex"] == expected
    source = json.loads((reference_root / "raw_manifests/source_manifest.json").read_text(encoding="utf-8"))
    assert (source["schema_version"], source["public_index_revision"]) == (1, 2)
    assert len(source["files"]) == 170
    # Lock the exact selected historical identity/size/hash records, not merely
    # a smaller count that could silently drop a scientific source dependency.
    canonical = json.dumps(source["files"], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    assert hashlib.sha256(canonical).hexdigest() == "47bdfe82e1dbfae90c652926959d540fc175a9cfa0408df848dab3fc2a68388c"
    indexed = {(r["source_id"], r["path"]): r for r in source["files"]}
    assert len(indexed) == 170
    for identity, digest in verification["source_hashes"].items():
        assert indexed[tuple(identity.split("/", 1))]["sha256"].lower() == digest.lower()
    paper = [r for r in source["files"] if r["source_id"] == "paper_tex"]
    assert len(paper) == 1 and paper[0]["bytes"] == 89380
    assert paper[0]["sha256"].lower() == expected


def test_prose_identity_change_does_not_change_svg_salt(tmp_path, monkeypatch):
    fig, axis = p.plt.subplots()
    axis.plot([0, 1], [0, 1])
    try:
        p._export_vectors({"sample": fig}, tmp_path / "before")
        monkeypatch.setattr(p, "MANIFEST_SHA256", "0" * 64)
        p._export_vectors({"sample": fig}, tmp_path / "after")
        assert p.sha256(tmp_path / "before/sample.svg") == p.sha256(tmp_path / "after/sample.svg")
    finally:
        p.plt.close(fig)


def test_real_half_up_boundaries():
    assert p.display("1.005") == "1.01"
    assert p.display("2.675") == "2.68"
    assert p.display("-1.005") == "-1.01"


def test_frozen_tables_and_shuffle(reference_root):
    data = p.load_inputs(reference_root)
    tables = p.render_tables(data)
    assert set(tables) == {"table_II.tex", "table_III.tex", "table_IV.tex"}
    assert sum(p.validate_tables(tables).values()) == 0
    assert r"{\pm}" in tables["table_II.tex"]
    assert r"{\pm}" not in tables["table_III.tex"]
    assert r"\pm" in tables["table_IV.tex"]
    shuffled = copy.deepcopy(data)
    for rows in shuffled.values():
        random.Random(19).shuffle(rows)
    assert p.render_tables(shuffled) == tables


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "ddof", "count"])
def test_main_invalid_keys_or_estimand(reference_root, change):
    data = p.load_inputs(reference_root)
    rows = data["main_final_accuracy.csv"]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "extra":
        rows[0]["method"] = "invented"
    elif change == "ddof":
        rows[0]["sd_ddof"] = "1"
    else:
        rows[0]["n_trajectories"] = "1"
    with pytest.raises(ValueError):
        p.validate_data(data)


@pytest.mark.parametrize("token", ["Notes:", r"\footnote", r"\begin{minipage}", r"\begin{threeparttable}", "tablenotes"])
def test_table_notes_rejected(reference_root, token):
    tables = p.render_tables(p.load_inputs(reference_root))
    tables["table_III.tex"] += token
    with pytest.raises(ValueError):
        p.validate_tables(tables)


def test_hash_preflight_before_output(reference_root, tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        p.write_all_paper_outputs(tmp_path / "missing", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_figure_scientific_contract(reference_root):
    data = p.load_inputs(reference_root)
    evidence = p.figure_evidence(data)
    assert evidence["main"] == {"rows": 5712, "panels": 16, "curves": 112, "points_per_curve": 51, "ddof": 0}
    assert evidence["figure7"]["rows"] == 7650
    assert evidence["figure7"]["curves"] == 5
    assert evidence["figure7"]["paired_inset_curves"] == 4
    assert evidence["figure8"] == {"rows": 150000, "partitions": 5, "display_points_per_partition": 1400, "display_seed": 20260827}
    assert evidence["figure9"]["classifications"] == {"better": 424, "worse": 410, "unresolved": 366}
    assert evidence["figure9"]["raw_rows"] == 42840
    assert evidence["figure9"]["comparisons"] == 1200
    assert evidence["figure10"]["rows"] == 8


def test_build_figures_exact_series(reference_root):
    data = p.load_inputs(reference_root)
    figures = p.build_figures(data)
    try:
        assert len(figures) == 20
        assert "figure_3/MNIST-dir 0.6" in figures
        assert "figure_6/Brain Tumor-pathological" in figures
        assert p.validate_figures(figures, data) == {"figure_series_differences": 0}
        fig7 = figures["figure_7/fig-distribution-accuracy"]
        assert fig7.axes[1].get_title() == "Difference vs. IID (percentage points)"
        assert all(t.get_fontsize() == 5 for t in fig7.axes[1].get_xticklabels())
        fig = figures["figure_3/MNIST-iid"]
        fig.axes[0].lines[0].set_ydata(np.zeros(51))
        with pytest.raises(ValueError):
            p.validate_figures(figures, data)
    finally:
        import matplotlib.pyplot as plt
        for fig in figures.values():
            plt.close(fig)


def test_atomic_publication_and_tamper(reference_root, tmp_path):
    output = tmp_path / "paper"
    manifest = p.write_all_paper_outputs(reference_root, output)
    assert manifest["status"] == "completed"
    assert manifest["verification"] == {"table_cell_differences": 0, "figure_series_differences": 0, "missing_outputs": 0}
    assert len(manifest["files"]) == 52
    for relative, record in manifest["files"].items():
        if relative.endswith(".csv"):
            with (output / relative).open(newline="", encoding="utf-8") as stream:
                actual_rows = sum(1 for _ in csv.DictReader(stream))
            assert record["source_rows"] == actual_rows
            assert record["source_rows_semantics"] == "csv_data_rows"
    assert p.verify_paper_outputs(reference_root, output) == manifest
    from spade_dfl_release.communication.plotting import configure_matplotlib
    with p.plt.rc_context():
        configure_matplotlib()
        caller = dict(p.matplotlib.rcParams)
        repeated = p.write_all_paper_outputs(reference_root, tmp_path / "polluted-rc")
        assert dict(p.matplotlib.rcParams) == caller
    assert {n: r["sha256"] for n, r in repeated["files"].items()} == {n: r["sha256"] for n, r in manifest["files"].items()}
    assert not list(output.rglob("*.png")) and not list(output.rglob("*.tiff"))
    with pytest.raises(ValueError):
        p.write_all_paper_outputs(reference_root, output)
    manifest_path = output / "PAPER_OUTPUT_MANIFEST.json"
    for extension in ("pdf", "svg"):
        relative = "figure_9/fig-method-volcano." + extension
        target = output / relative
        original = target.read_bytes()
        target.write_bytes(original + b"coordinated tamper")
        forged = copy.deepcopy(manifest)
        forged["files"][relative]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        forged["files"][relative]["bytes"] = target.stat().st_size
        manifest_path.write_text(json.dumps(forged), encoding="utf-8")
        with pytest.raises(ValueError, match="independent rendered figure"):
            p.verify_paper_outputs(reference_root, output)
        target.write_bytes(original)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = output / "figure_9/fig-method-volcano.svg"
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        p.verify_paper_outputs(reference_root, output)


def test_atomic_render_failure(reference_root, tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("injected render failure")
    monkeypatch.setattr(p, "build_figures", fail)
    output = tmp_path / "paper"
    with pytest.raises(RuntimeError, match="injected"):
        p.write_all_paper_outputs(reference_root, output)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_export_options_match_reviewed_sources():
    assert p._save_options("figure_10/fig_communication_saving_band") == {"bbox_inches": "tight", "pad_inches": 0.02, "facecolor": "white"}
    assert p._save_options("figure_6/Brain Tumor-iid") == {"bbox_inches": None}


def test_export_context_is_frozen_and_does_not_leak(frozen_data, tmp_path):
    from spade_dfl_release.communication.plotting import configure_matplotlib
    fingerprints = []
    for mode in ("default", "communication_renderer"):
        with p.plt.rc_context(p.matplotlib.rcParamsDefault):
            figures = p.build_figures(frozen_data)
            try:
                if mode == "communication_renderer":
                    configure_matplotlib()
                caller = dict(p.matplotlib.rcParams)
                selected = {n: f for n, f in figures.items() if n.startswith(("figure_7/", "figure_9/"))}
                p._export_vectors(selected, tmp_path / mode)
                assert dict(p.matplotlib.rcParams) == caller
                fingerprints.append({f.relative_to(tmp_path / mode).as_posix(): p.sha256(f) for f in (tmp_path / mode).rglob("*") if f.is_file()})
            finally:
                for fig in figures.values():
                    p.plt.close(fig)
    assert len(fingerprints[0]) == 4
    assert fingerprints[0] == fingerprints[1]


def test_qa_rasters_are_600dpi_and_separate(tmp_path):
    calls = []
    class Figure:
        def savefig(self, path, **kwargs):
            calls.append((str(path), kwargs))
    p.export_qa_rasters(Figure(), tmp_path / "preview")
    assert [Path(path).suffix for path, _ in calls] == [".png", ".tiff"]
    assert all(options["dpi"] == 600 for _, options in calls)


@pytest.fixture(scope="module")
def frozen_data():
    return p.load_inputs(p.ROOT / "reference_results")


@pytest.mark.parametrize("name,expected", [("s15_roundwise.csv", 42840), ("s15_wilcoxon_bh.csv", 1200)])
def test_figure9_csv_metadata_counts_its_own_rows(frozen_data, name, expected):
    record = p._output_metadata("figure_9/" + name, frozen_data)
    assert record["source_rows"] == expected
    assert record["source_rows_semantics"] == "csv_data_rows"
    assert record["input_rows_by_file"] == {name: expected}
    assert set(record["inputs"]) == {name}


@pytest.mark.parametrize("extension", ["pdf", "svg"])
def test_figure9_vector_metadata_counts_combined_inputs(frozen_data, extension):
    record = p._output_metadata("figure_9/fig-method-volcano." + extension, frozen_data)
    assert record["source_rows"] == 44040
    assert record["source_rows_semantics"] == "combined_figure_input_rows"
    assert record["input_rows_by_file"] == {"s15_roundwise.csv": 42840, "s15_wilcoxon_bh.csv": 1200}


def test_output_row_metadata_is_derived_from_selected_records(frozen_data):
    changed = dict(frozen_data)
    changed["s15_wilcoxon_bh.csv"] = frozen_data["s15_wilcoxon_bh.csv"][:-1]
    assert p._output_metadata("figure_9/s15_wilcoxon_bh.csv", changed)["source_rows"] == 1199
    assert p._output_metadata("figure_9/fig-method-volcano.pdf", changed)["source_rows"] == 44039


def test_figure8_boxplot_and_figure9_structure_mutations(frozen_data):
    import matplotlib.pyplot as plt
    figures = p.build_figures(frozen_data)
    try:
        box = figures["figure_8/fig-client-label-heterogeneity"].axes[0]
        # Five lines per group: two whiskers, two caps, then the median.
        for index in (0, 1, 4):
            line = box.lines[index]
            original = line.get_ydata().copy()
            line.set_ydata(np.asarray(original) + 0.01)
            with pytest.raises(ValueError, match="Figure 8"):
                p.validate_figures(figures, frozen_data)
            line.set_ydata(original)
        patch = box.patches[0]
        original = patch.get_path().vertices.copy()
        for index in (0, 2):
            patch.get_path().vertices[index, 1] += 0.01
            with pytest.raises(ValueError, match="Figure 8"):
                p.validate_figures(figures, frozen_data)
            patch.get_path().vertices[:] = original
        volcano = figures["figure_9/fig-method-volcano"].axes[0]
        for index, coordinate in ((0, "x"), (1, "y")):
            line = volcano.lines[index]
            original = getattr(line, "get_" + coordinate + "data")()
            getattr(line, "set_" + coordinate + "data")([0.1, 0.1])
            with pytest.raises(ValueError, match="Figure 9"):
                p.validate_figures(figures, frozen_data)
            getattr(line, "set_" + coordinate + "data")(original)
        annotation = volcano.texts[0]
        original = annotation.get_text()
        annotation.set_text("Wrong identity")
        with pytest.raises(ValueError, match="Figure 9"):
            p.validate_figures(figures, frozen_data)
        annotation.set_text(original)
        original = volcano.get_xlabel()
        volcano.set_xlabel("Accuracy change vs. SPADE-DFL")
        with pytest.raises(ValueError, match="Figure 9"):
            p.validate_figures(figures, frozen_data)
        volcano.set_xlabel(original)
        annotation.remove()
        with pytest.raises(ValueError, match="Figure 9 annotation count"):
            p.validate_figures(figures, frozen_data)
    finally:
        for fig in figures.values():
            plt.close(fig)


@pytest.mark.parametrize("name", ["main_final_accuracy.csv", "main_roundwise.csv", "sensitivity_cells.csv", "s15_roundwise.csv", "s15_client_proportions.csv", "s15_wilcoxon_bh.csv", "communication_summary.csv"])
def test_same_count_duplicate_logical_key(frozen_data, name):
    data = dict(frozen_data)
    data[name] = list(data[name])
    data[name][-1] = dict(data[name][0])
    with pytest.raises(ValueError):
        p.validate_data(data)


@pytest.mark.parametrize("field,value", [("ddof", "0"), ("n_seed", "25"), ("n_fold_seed", "5"), ("formal_authorized", "True"), ("validation_only", "False"), ("previous_test_exposure", "False")])
def test_sensitivity_estimand_mutations(frozen_data, field, value):
    data = dict(frozen_data)
    data["sensitivity_cells.csv"] = [dict(r) for r in data["sensitivity_cells.csv"]]
    row = next(r for r in data["sensitivity_cells.csv"] if r["dataset"] == "alzheimer")
    row[field] = value
    with pytest.raises(ValueError):
        p.validate_data(data)


def test_shared_anchor_once_and_synthetic_sensitivity_values(frozen_data):
    data = dict(frozen_data)
    data["sensitivity_cells.csv"] = [dict(r) for r in data["sensitivity_cells.csv"]]
    anchors = [r for r in data["sensitivity_cells.csv"] if (r["n"], r["epsilon"]) == ("100", "32")]
    assert len(anchors) == 4
    for row in anchors:
        row["mean_percent"], row["sd_percent"] = "12.345", "0.005"
    tables = p.render_tables(data)
    for name in ("table_III.tex", "table_IV.tex"):
        assert tables[name].count(r"$12.35\pm0.01$") == 4
        assert r"\mathbf" not in tables[name] and r"\underline" not in tables[name]
    with pytest.raises(ValueError):
        p.validate_tables(tables)


def _synthetic_methods():
    return [{"method": m, "mean_percent": str(10 + i * 3), "sd_percent": "1.005"} for i, m in enumerate(p.METHODS)]


@pytest.mark.parametrize("kind", ["raw_best", "display_best", "raw_other", "display_other", "not_best"])
def test_method_ties_and_not_best(kind):
    rows = _synthetic_methods()
    if kind == "raw_best":
        rows[-1]["mean_percent"] = rows[-2]["mean_percent"]
    elif kind == "display_best":
        rows[-1]["mean_percent"], rows[-2]["mean_percent"] = "28.004", "28.003"
    elif kind == "raw_other":
        rows[-2]["mean_percent"] = rows[-3]["mean_percent"]
    elif kind == "display_other":
        rows[-2]["mean_percent"], rows[-3]["mean_percent"] = "25.004", "25.003"
    else:
        rows[-1]["mean_percent"] = "0"
    with pytest.raises(ValueError):
        p.method_cells(rows)


def test_unrounded_delta_and_no_result_constants():
    rows = _synthetic_methods()
    rows[-1]["mean_percent"], rows[-2]["mean_percent"] = "28.004", "25.006"
    rendered = p.method_cells(rows)
    assert rendered[-1] == "$+3.00$"
    assert rendered[-2] == r"$\mathbf{28.00{\pm}1.01}$"
    assert rendered[-3] == r"$\underline{25.01{\pm}1.01}$"


@pytest.mark.parametrize("old,new", [("Final-round", "Last-round"), ("tab:distribution_final_accuracy", "wrong"), ("Dataset & Distribution", "Distribution & Dataset"), (r"\mathbf", r"\underline"), (r"\underline", ""), (r"llcccccccc", "lccc")])
def test_table_structure_and_obsolete_helper_rejected(frozen_data, old, new):
    tables = p.render_tables(frozen_data)
    tables["table_II.tex"] = tables["table_II.tex"].replace(old, new)
    with pytest.raises(ValueError):
        p.validate_tables(tables)


def test_presentation_manifest_mutation_fails(tmp_path, monkeypatch):
    manifest = p.load_manifest()
    manifest["semantics"]["main_sd_ddof"] = 1
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(p, "MANIFEST_PATH", path)
    with pytest.raises(ValueError, match="semantic hash"):
        p.load_manifest()


def test_dynamic_communication_label(frozen_data):
    from spade_dfl_release.communication.plotting import saving_label
    rows = p._communication_rows(frozen_data)
    old = saving_label(rows)
    for row in rows:
        if row["schedule"] != "tau=1":
            row["transmitted_scalars"] *= 2
    assert saving_label(rows) != old


def test_renderer_closure_and_mutation(tmp_path, monkeypatch):
    assert p.verify_renderer_lineage() == 13
    assert p.MANIFEST_SHA256 == "5a184c73b1c8dd4509a450e08c6ba4f8b48b21e5f51bffdc4b56b3370f8ed009"
    source = Path(p.__file__).read_text(encoding="utf-8")
    assert "linewidth=1.05" in source
    altered = tmp_path / "altered.py"
    altered.write_text(source.replace("linewidth=1.05", "linewidth=9.05"), encoding="utf-8")
    monkeypatch.setattr(p, "__file__", str(altered))
    with pytest.raises(ValueError, match="renderer AST"):
        p.verify_renderer_lineage()


def test_main_inset_and_visible_structure_mutations(frozen_data):
    figures = p.build_figures(frozen_data)
    try:
        axis = figures["figure_3/MNIST-iid"].axes[0]
        inset = axis.child_axes[0]
        original = inset.lines[0].get_ydata().copy()
        inset.lines[0].set_ydata(np.zeros(51))
        with pytest.raises(ValueError):
            p.validate_figures(figures, frozen_data)
        inset.lines[0].set_ydata(original)
        axis.set_xlabel("Wrong axis")
        with pytest.raises(ValueError):
            p.validate_figures(figures, frozen_data)
    finally:
        import matplotlib.pyplot as plt
        for fig in figures.values():
            plt.close(fig)


def test_input_tamper_and_pooling_certificate_fail_before_output(reference_root, tmp_path):
    import shutil
    reference = tmp_path / "reference"
    (reference / "paper_values").mkdir(parents=True)
    (reference / "aggregates").mkdir()
    verification = reference / "paper_values/verification.json"
    shutil.copyfile(reference_root / "paper_values/verification.json", verification)
    target = reference / "aggregates/communication_summary.csv"
    target.write_bytes((reference_root / "aggregates/communication_summary.csv").read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="input hash"):
        p.write_all_paper_outputs(reference, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    certificate = json.loads(verification.read_text())
    certificate["semantics"]["alzheimer_sensitivity"] = "unweighted fold mean"
    verification.write_text(json.dumps(certificate))
    with pytest.raises(ValueError, match="verification hash"):
        p.load_inputs(reference)


def test_rendering_isolates_global_rc_and_repeated_geometry(frozen_data):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from spade_dfl_release.communication.plotting import configure_matplotlib
    with mpl.rc_context(mpl.rcParamsDefault):
        before = dict(mpl.rcParams)
        first = p.build_figures(frozen_data)
        try:
            assert dict(mpl.rcParams) == before
            expected = {name: [tuple(ax.get_position().bounds) for ax in fig.axes] for name, fig in first.items()}
        finally:
            for fig in first.values():
                plt.close(fig)
        configure_matplotlib()
        caller = dict(mpl.rcParams)
        second = p.build_figures(frozen_data)
        try:
            assert dict(mpl.rcParams) == caller
            assert {name: [tuple(ax.get_position().bounds) for ax in fig.axes] for name, fig in second.items()} == expected
        finally:
            for fig in second.values():
                plt.close(fig)
