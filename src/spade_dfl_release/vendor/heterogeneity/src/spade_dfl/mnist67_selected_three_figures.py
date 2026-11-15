"""Build the three author-approved MNIST 6/7 paper figures from saved runs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .distribution_study import load_distribution_study
from .mnist67_method_volcano import (
    METHOD_ORDER as COMPARISON_METHODS,
    PARTITION_ORDER as COMPARISON_PARTITIONS,
    plot_method_volcano_from_result_root,
)
from .mnist67_spade_distribution import (
    PARTITION_ORDER as DISTRIBUTION_PARTITIONS,
    plot_accuracy_with_iid_inset,
    plot_label_heterogeneity,
)


FIGURE_IDS = (
    "figure1_distribution_accuracy",
    "figure2_client_label_heterogeneity",
    "figure3_method_volcano",
)
DISTRIBUTION_METHOD = "spade_dfl_private"
PRIVATE_COMPARISON_METHODS = {"zo2_dp_sgd", "zo2_dp_fedavg"}


def build_selected_three_figures(
    distribution_root: Path,
    comparison_root: Path,
    output_dir: Path,
) -> Mapping[str, tuple[Path, Path, Path, Path]]:
    """Audit 30-seed results, then render exactly the three approved figures."""

    distribution = Path(distribution_root).resolve()
    comparison = Path(comparison_root).resolve()
    output = Path(output_dir).resolve()
    audit_formal_results(distribution, comparison)
    study = load_distribution_study(
        distribution,
        DISTRIBUTION_PARTITIONS,
        threshold=0.9,
        method=DISTRIBUTION_METHOD,
        n_clients=1000,
    )
    output.mkdir(parents=True, exist_ok=True)
    bundles = {
        FIGURE_IDS[0]: plot_accuracy_with_iid_inset(study, output / FIGURE_IDS[0]),
        FIGURE_IDS[1]: plot_label_heterogeneity(study, output / FIGURE_IDS[1]),
        FIGURE_IDS[2]: plot_method_volcano_from_result_root(
            comparison, output / FIGURE_IDS[2]
        ),
    }
    _write_figure_manifest(output, distribution, comparison, bundles)
    return bundles


def audit_formal_results(distribution_root: Path, comparison_root: Path) -> None:
    """Fail closed unless both formal suites match their frozen contracts."""

    _audit_suite(
        Path(distribution_root),
        methods=(DISTRIBUTION_METHOD,),
        partitions=DISTRIBUTION_PARTITIONS,
        seeds=tuple(range(30)),
        rounds=50,
        private_methods={DISTRIBUTION_METHOD},
    )
    _audit_suite(
        Path(comparison_root),
        methods=COMPARISON_METHODS,
        partitions=COMPARISON_PARTITIONS,
        seeds=tuple(range(30)),
        rounds=50,
        private_methods=PRIVATE_COMPARISON_METHODS,
    )


def audit_smoke_results(distribution_root: Path, comparison_root: Path) -> None:
    """Check the one-round execution path without making paper claims."""

    _audit_suite(
        Path(distribution_root),
        methods=(DISTRIBUTION_METHOD,),
        partitions=DISTRIBUTION_PARTITIONS,
        seeds=(0,),
        rounds=1,
        private_methods={DISTRIBUTION_METHOD},
    )
    _audit_suite(
        Path(comparison_root),
        methods=COMPARISON_METHODS,
        partitions=COMPARISON_PARTITIONS,
        seeds=(0,),
        rounds=1,
        private_methods=PRIVATE_COMPARISON_METHODS,
    )


def _audit_suite(
    root: Path,
    *,
    methods: Sequence[str],
    partitions: Sequence[str],
    seeds: Sequence[int],
    rounds: int,
    private_methods: set[str],
) -> None:
    root = Path(root).resolve()
    manifest = _read_json(root / "suite_manifest.json")
    suite = manifest.get("suite_config")
    records = manifest.get("runs")
    if not isinstance(suite, dict) or not isinstance(records, list):
        raise ValueError(f"malformed suite manifest: {root}")
    declared_methods = tuple(str(item) for item in suite.get("methods", ()))
    declared_partitions = tuple(str(item) for item in suite.get("partitions", ()))
    declared_seeds = tuple(int(item) for item in suite.get("seeds", ()))
    if declared_methods != tuple(methods):
        raise ValueError(f"method contract mismatch: {root}")
    if set(declared_partitions) != set(partitions) or len(declared_partitions) != len(partitions):
        raise ValueError(f"partition contract mismatch: {root}")
    if declared_seeds != tuple(seeds):
        raise ValueError(f"paired-seed contract mismatch: {root}")
    expected = {
        (partition, method, int(seed))
        for partition in partitions
        for method in methods
        for seed in seeds
    }
    observed: dict[tuple[str, str, int], Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"malformed run record: {root}")
        identity = (
            str(record.get("partition")),
            str(record.get("method")),
            int(record.get("seed", -1)),
        )
        if identity in observed:
            raise ValueError(f"duplicate run identity: {identity}")
        observed[identity] = record
    if set(observed) != expected:
        raise ValueError(
            f"run matrix mismatch: expected {len(expected)}, observed {len(observed)}"
        )
    for identity in sorted(expected):
        record = observed[identity]
        run_dir = _resolve_run_dir(root, record.get("run_dir"))
        _audit_history(run_dir / "history.csv", rounds)
        summary = _read_json(run_dir / "summary.json")
        method = identity[1]
        if int(summary.get("communication_rounds", -1)) != _expected_communication_rounds(
            method, rounds
        ):
            raise ValueError(f"wrong round count for {identity}")
        if int(summary.get("gradient_evaluations", -1)) != 0:
            raise ValueError(f"analytic gradient access detected for {identity}")
        if float(summary.get("query_count", 0.0)) <= 0.0:
            raise ValueError(f"missing function-query count for {identity}")
        metadata = summary.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("gradient_access") is not False:
            raise ValueError(f"black-box declaration missing for {identity}")
        expected_oracle = "symmetric_two_point" if method.startswith("zo2_") else "one_point"
        if metadata.get("gradient_oracle") != expected_oracle:
            raise ValueError(f"wrong zero-order estimator for {identity}")
        if method == "zo2_dp_sgd":
            if int(metadata.get("outer_round_count", -1)) != rounds:
                raise ValueError(f"DP-SGD aligned update count is wrong for {identity}")
            if metadata.get("x_axis_caveat") != "optimization-step index; no network communication":
                raise ValueError(f"DP-SGD communication caveat is missing for {identity}")
        if method in {"spade_dfl", DISTRIBUTION_METHOD} and metadata.get("vr_mode") != "persistent_saga":
            raise ValueError(f"persistent SAGA declaration missing for {identity}")
        if method in private_methods:
            if summary.get("privacy_valid") is not True:
                raise ValueError(f"invalid privacy accounting for {identity}")
            if not np.isclose(float(summary.get("epsilon", np.nan)), 8.0):
                raise ValueError(f"epsilon must equal 8 for {identity}")
            if not np.isclose(float(summary.get("delta", np.nan)), 1e-5):
                raise ValueError(f"delta must equal 1e-5 for {identity}")


def _audit_history(path: Path, rounds: int) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != rounds + 1:
        raise ValueError(f"history must contain rounds 0 through {rounds}: {path}")
    round_key = "round" if rows and "round" in rows[0] else "iteration"
    try:
        observed = [int(row[round_key]) for row in rows]
        accuracy = np.asarray([float(row["accuracy"]) for row in rows], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"malformed history: {path}") from error
    if observed != list(range(rounds + 1)) or not np.all(np.isfinite(accuracy)):
        raise ValueError(f"invalid round axis or accuracy values: {path}")


def _expected_communication_rounds(method: str, aligned_rounds: int) -> int:
    """Return actual network rounds; DP-SGD has aligned steps but no network."""

    return 0 if method == "zo2_dp_sgd" else int(aligned_rounds)


def _resolve_run_dir(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("run record lacks run_dir")
    declared = Path(value)
    try:
        index = declared.parts.index("runs")
    except ValueError as error:
        raise ValueError(f"run_dir lacks a runs anchor: {value}") from error
    candidate = (root / Path(*declared.parts[index:])).resolve()
    try:
        candidate.relative_to((root / "runs").resolve())
    except ValueError as error:
        raise ValueError(f"run_dir escapes result root: {value}") from error
    return candidate


def _write_figure_manifest(
    output: Path,
    distribution: Path,
    comparison: Path,
    bundles: Mapping[str, tuple[Path, Path, Path, Path]],
) -> None:
    records = {}
    for figure_id, paths in bundles.items():
        records[figure_id] = [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ]
    (output / "SELECTED_FIGURE_MANIFEST.json").write_text(
        json.dumps(
            {
                "dataset": "MNIST digits 6 and 7",
                "figure_ids": list(FIGURE_IDS),
                "distribution_result_root": str(distribution),
                "comparison_result_root": str(comparison),
                "figure_sources": {
                    FIGURE_IDS[0]: "five-distribution strong-DP SPADE-DFL suite",
                    FIGURE_IDS[1]: "client label histograms from the same five-distribution suite",
                    FIGURE_IDS[2]: "four-distribution seven-method all-black-box comparison suite",
                },
                "files": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload
