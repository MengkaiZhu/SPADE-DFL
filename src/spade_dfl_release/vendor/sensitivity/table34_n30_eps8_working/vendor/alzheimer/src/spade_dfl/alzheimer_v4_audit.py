"""Independent read-only audit for the frozen Alzheimer v4 evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .alzheimer_contract import ALZHEIMER_METHOD_IDS
from .alzheimer_data import build_stratified_alzheimer_fold
from .alzheimer_suite import ALZHEIMER_PARTITIONS, DISPLAY_LABELS
from .alzheimer_v3_tuning import _canonical_sha256, _load_v3_tuning_config
from .alzheimer_v4_cross_validation import (
    _final_method_config,
    _load_contract,
    _record_beta,
    _selected_candidates,
)

_POINTS = 51
_TOLERANCE = 1e-12


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact is not an object: {path}")
    return payload


def _read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != _POINTS:
        raise ValueError(f"history does not contain {_POINTS} rows: {path}")
    if not rows or not rows[0]:
        raise ValueError(f"history has no columns: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in rows[0]
    }


def _append_close_error(
    errors: list[str],
    *,
    label: str,
    actual: float,
    expected: float,
) -> None:
    if not np.isclose(actual, expected, atol=_TOLERANCE, rtol=0.0):
        errors.append(label)


def compare_accuracy_source(
    path: Path,
    aggregates: Mapping[str, Mapping[str, np.ndarray]],
    errors: list[str],
) -> None:
    """Compare plotted source rows with independently recomputed arrays."""
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError) as error:
        errors.append(f"cannot read figure source: {type(error).__name__}")
        return
    expected_count = sum(
        len(np.asarray(aggregate["round"])) for aggregate in aggregates.values()
    )
    if len(rows) != expected_count:
        errors.append("figure source row count mismatch")
        return
    by_key = {
        (str(row.get("method_id")), int(str(row.get("round")))): row
        for row in rows
    }
    for method, aggregate in aggregates.items():
        rounds = np.asarray(aggregate["round"], dtype=np.int64)
        means = np.asarray(aggregate["accuracy_mean"], dtype=np.float64)
        stds = np.asarray(aggregate["accuracy_std"], dtype=np.float64)
        for round_index, mean, std in zip(rounds, means, stds):
            key = (method, int(round_index))
            row = by_key.get(key)
            if row is None:
                errors.append(
                    f"figure source missing: {method} round {round_index}"
                )
                continue
            if row.get("method") != DISPLAY_LABELS[method]:
                errors.append(
                    f"figure label mismatch: {method} round {round_index}"
                )
            _append_close_error(
                errors,
                label=f"figure source mismatch: {method} round {round_index} mean",
                actual=float(str(row["mean"])),
                expected=float(mean),
            )
            _append_close_error(
                errors,
                label=f"figure source mismatch: {method} round {round_index} std",
                actual=float(str(row["std"])),
                expected=float(std),
            )


def _compare_npz(
    path: Path,
    expected: Mapping[str, np.ndarray],
    errors: list[str],
) -> None:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) != set(expected):
                errors.append(f"aggregate columns mismatch: {path}")
                return
            for name, values in expected.items():
                actual = np.asarray(loaded[name])
                if actual.shape != values.shape or not np.allclose(
                    actual,
                    values,
                    atol=_TOLERANCE,
                    rtol=0.0,
                ):
                    errors.append(f"aggregate mismatch: {path.name} {name}")
    except (OSError, ValueError) as error:
        errors.append(f"cannot read aggregate: {path} ({type(error).__name__})")


def _expected_config_hashes(
    results_root: Path,
    config_path: Path,
) -> tuple[
    Any,
    dict[tuple[str, str, int], str],
]:
    contract = _load_contract(config_path)
    tuning = _load_v3_tuning_config(contract.tuning_config_path)
    base = replace(
        tuning.experiment,
        n_rounds=contract.rounds,
        seeds=contract.seeds,
        output_dir=str(results_root.resolve()),
    )
    candidates = _selected_candidates(contract)
    hashes: dict[tuple[str, str, int], str] = {}
    for partition in contract.partitions:
        for method in ALZHEIMER_METHOD_IDS:
            candidate = candidates[method]
            common = _final_method_config(base, method, candidate, partition)
            for seed in contract.seeds:
                exact = (
                    _record_beta(common, candidate, seed)
                    if method == "spade_dfl"
                    else common
                )
                hashes[(partition, method, seed)] = _canonical_sha256(
                    exact.to_dict()
                )
    return contract, hashes


def _audit_folds(
    contract: Any,
    manifest: Mapping[str, object],
    errors: list[str],
) -> None:
    declared_rows = manifest.get("fold_audit")
    if not isinstance(declared_rows, list) or len(declared_rows) != 5:
        errors.append("suite fold audit is missing or incomplete")
        return
    declared = {
        int(row["fold"]): row
        for row in declared_rows
        if isinstance(row, dict) and "fold" in row
    }
    all_test_indices: list[np.ndarray] = []
    for fold in range(contract.n_folds):
        split = build_stratified_alzheimer_fold(
            contract.data_path,
            n_folds=contract.n_folds,
            fold_index=fold,
            split_seed=contract.split_seed,
        )
        all_test_indices.append(np.asarray(split.test_indices, dtype=np.int64))
        row = declared.get(fold)
        if row is None:
            errors.append(f"suite fold audit missing fold {fold}")
            continue
        train_hash = hashlib.sha256(
            np.asarray(split.model_indices, dtype="<i8").tobytes()
        ).hexdigest()
        test_hash = hashlib.sha256(
            np.asarray(split.test_indices, dtype="<i8").tobytes()
        ).hexdigest()
        if row.get("train_indices_sha256") != train_hash:
            errors.append(f"fold {fold} train index hash mismatch")
        if row.get("test_indices_sha256") != test_hash:
            errors.append(f"fold {fold} test index hash mismatch")
        if int(row.get("train_count", -1)) != len(split.model_indices):
            errors.append(f"fold {fold} train count mismatch")
        if int(row.get("test_count", -1)) != len(split.test_indices):
            errors.append(f"fold {fold} test count mismatch")
        if not set(split.model_indices.tolist()).isdisjoint(
            split.test_indices.tolist()
        ):
            errors.append(f"fold {fold} has train/test overlap")
    concatenated = np.concatenate(all_test_indices)
    if (
        concatenated.size != 2149
        or np.unique(concatenated).size != 2149
        or set(concatenated.tolist()) != set(range(2149))
    ):
        errors.append("five test folds do not cover the 2149-row cohort exactly once")


def _audit_final_metrics(
    path: Path,
    expected: Mapping[tuple[str, str], Mapping[str, float]],
    errors: list[str],
) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError) as error:
        errors.append(f"cannot read final metrics: {type(error).__name__}")
        return
    if len(rows) != 28:
        errors.append("final metrics row count mismatch")
        return
    for row in rows:
        key = (str(row["partition"]), str(row["method_id"]))
        reference = expected.get(key)
        if reference is None:
            errors.append(f"unexpected final metrics row: {key}")
            continue
        if row["method"] != DISPLAY_LABELS[key[1]]:
            errors.append(f"final metrics label mismatch: {key}")
        for field in (
            "final_accuracy_mean",
            "final_accuracy_std",
            "query_count_total",
            "transmitted_bytes_total",
        ):
            _append_close_error(
                errors,
                label=f"final metrics mismatch: {key} {field}",
                actual=float(row[field]),
                expected=float(reference[field]),
            )


def audit_v4_cross_validation(
    results_root: Path,
    config_path: Path,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Recompute configs, raw curves, aggregates, figure sources, and ranks."""
    results_root = Path(results_root).resolve()
    config_path = Path(config_path).resolve()
    errors: list[str] = []
    manifest = _read_json(results_root / "suite_manifest.json")
    contract, expected_hashes = _expected_config_hashes(results_root, config_path)
    candidates = _selected_candidates(contract)
    if manifest.get("protocol_version") != contract.protocol_version:
        errors.append("suite protocol version mismatch")
    if manifest.get("curve_postprocessing") != "none":
        errors.append("suite claims curve post-processing")
    _audit_folds(contract, manifest, errors)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    audited_runs = 0
    expected_dirs: set[Path] = set()
    for fold in range(contract.n_folds):
        for partition in contract.partitions:
            for method in ALZHEIMER_METHOD_IDS:
                for seed in contract.seeds:
                    run_dir = (
                        results_root
                        / "raw"
                        / f"fold_{fold}"
                        / partition
                        / method
                        / f"seed_{seed}"
                    )
                    expected_dirs.add(run_dir)
                    try:
                        config = _read_json(run_dir / "config.json")
                        environment = _read_json(run_dir / "environment.json")
                        summary = _read_json(run_dir / "summary.json")
                        history = _read_history(run_dir / "history.csv")
                        with np.load(
                            run_dir / "states.npz", allow_pickle=False
                        ) as states:
                            if not states.files or any(
                                not np.all(np.isfinite(states[name]))
                                for name in states.files
                            ):
                                errors.append(f"nonfinite states: {run_dir}")
                        if any(
                            values.shape != (_POINTS,)
                            or not np.all(np.isfinite(values))
                            for values in history.values()
                        ):
                            errors.append(f"nonfinite or invalid history: {run_dir}")
                        actual_hash = _canonical_sha256(config)
                        expected_hash = expected_hashes[(partition, method, seed)]
                        dataset = environment.get("dataset")
                        environment_hash = (
                            dataset.get("config_sha256")
                            if isinstance(dataset, dict)
                            else None
                        )
                        if actual_hash != expected_hash:
                            errors.append(f"frozen config mismatch: {run_dir}")
                        if environment_hash != actual_hash:
                            errors.append(f"environment config hash mismatch: {run_dir}")
                        if (
                            not isinstance(dataset, dict)
                            or dataset.get("fold") != fold
                            or dataset.get("partition") != partition
                            or dataset.get("candidate_id")
                            != candidates[method].candidate_id
                        ):
                            errors.append(f"environment provenance mismatch: {run_dir}")
                        grouped[(partition, method)].append(
                            {
                                "history": history,
                                "query_count": int(summary["query_count"]),
                                "transmitted_bytes": int(
                                    summary["transmitted_bytes"]
                                ),
                            }
                        )
                        audited_runs += 1
                    except (
                        csv.Error,
                        json.JSONDecodeError,
                        KeyError,
                        OSError,
                        TypeError,
                        ValueError,
                    ) as error:
                        errors.append(
                            f"cannot audit run {run_dir}: {type(error).__name__}"
                        )
    discovered_dirs = {
        path.parent
        for path in (results_root / "raw").glob(
            "fold_*/*/*/seed_*/config.json"
        )
    }
    if discovered_dirs != expected_dirs:
        errors.append("raw run directory set differs from frozen 420-run contract")

    all_aggregates: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    final_expected: dict[tuple[str, str], dict[str, float]] = {}
    ranks: dict[str, int] = {}
    for partition in contract.partitions:
        partition_aggregates: dict[str, dict[str, np.ndarray]] = {}
        for method in ALZHEIMER_METHOD_IDS:
            rows = grouped[(partition, method)]
            repetitions = contract.n_folds * len(contract.seeds)
            if len(rows) != repetitions:
                errors.append(f"run count mismatch: {partition} {method}")
                continue
            accuracy = np.stack(
                [
                    np.asarray(row["history"]["accuracy"], dtype=np.float64)
                    for row in rows
                ]
            )
            loss = np.stack(
                [
                    np.asarray(row["history"]["loss"], dtype=np.float64)
                    for row in rows
                ]
            )
            stationarity = np.stack(
                [
                    np.asarray(
                        row["history"]["stationarity_consensus"],
                        dtype=np.float64,
                    )
                    for row in rows
                ]
            )
            aggregate = {
                "round": np.arange(_POINTS, dtype=np.int64),
                "accuracy_mean": accuracy.mean(axis=0),
                "accuracy_std": accuracy.std(axis=0, ddof=0),
                "loss_mean": loss.mean(axis=0),
                "loss_std": loss.std(axis=0, ddof=0),
                "stationarity_consensus_mean": stationarity.mean(axis=0),
                "stationarity_consensus_std": stationarity.std(axis=0, ddof=0),
            }
            partition_aggregates[method] = aggregate
            _compare_npz(
                results_root / "aggregates" / partition / f"{method}.npz",
                aggregate,
                errors,
            )
            final_expected[(partition, method)] = {
                "final_accuracy_mean": float(aggregate["accuracy_mean"][-1]),
                "final_accuracy_std": float(aggregate["accuracy_std"][-1]),
                "query_count_total": float(
                    sum(int(row["query_count"]) for row in rows)
                ),
                "transmitted_bytes_total": float(
                    sum(int(row["transmitted_bytes"]) for row in rows)
                ),
            }
        all_aggregates[partition] = partition_aggregates
        if set(partition_aggregates) == set(ALZHEIMER_METHOD_IDS):
            ordered = sorted(
                ALZHEIMER_METHOD_IDS,
                key=lambda method: (
                    -float(
                        partition_aggregates[method]["accuracy_mean"][-1]
                    ),
                    method,
                ),
            )
            ranks[partition] = ordered.index("spade_dfl") + 1
        compare_accuracy_source(
            results_root
            / "figures"
            / partition
            / "accuracy.source_data.csv",
            partition_aggregates,
            errors,
        )
    _audit_final_metrics(
        results_root / "final_metrics.csv",
        final_expected,
        errors,
    )

    expected_rank_map = {partition: 1 for partition in ALZHEIMER_PARTITIONS}
    expected_runs = (
        contract.n_folds
        * len(contract.partitions)
        * len(ALZHEIMER_METHOD_IDS)
        * len(contract.seeds)
    )
    mechanical = audited_runs == expected_runs and not errors
    scientific = mechanical and ranks == expected_rank_map
    summary: dict[str, object] = {
        "protocol_version": contract.protocol_version,
        "protocol_role": contract.protocol_role,
        "expected_run_count": expected_runs,
        "audited_run_count": audited_runs,
        "folds": contract.n_folds,
        "seeds": list(contract.seeds),
        "rounds": contract.rounds,
        "partitions": list(contract.partitions),
        "method_ids": list(ALZHEIMER_METHOD_IDS),
        "curve_postprocessing": "none",
        "aggregation": "raw_fold_seed_mean_and_population_std_ddof_0",
        "spade_partition_ranks": ranks,
        "mechanical_acceptance": mechanical,
        "scientific_acceptance": scientific,
        "errors": errors,
        "evaluation_limitation": (
            "Five-fold stratified cross-validation on one cohort; not an "
            "independent external cohort."
        ),
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        target = output / "audit_summary.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(target)
    return summary
