"""Execution and artifact gates for the frozen MNIST 6/7 paper suite."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .mnist67_paper_protocol import (
    METHOD_IDS,
    PARTITION_IDS,
    load_and_validate_protocol,
)


PRIVATE_METHODS = {"zo2_dp_sgd", "zo2_dp_fedavg", "spade_dfl_private"}
REQUIRED_HISTORY_COLUMNS = {
    "iteration",
    "accuracy",
    "loss",
    "stationarity",
    "consensus_error",
    "penalty_energy",
}


class IncompleteSuiteError(ValueError):
    """Raised when saved artifacts cannot support the requested evidence."""


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    run_count: int
    expected_run_count: int
    history_points: int
    inference_status: str


def expected_run_identities(
    suite: Mapping[str, object],
) -> tuple[tuple[str, str, int], ...]:
    """Return the Cartesian run matrix in deterministic protocol order."""

    methods = tuple(suite.get("methods", ()))
    partitions = tuple(suite.get("partitions", ()))
    raw_seeds = tuple(suite.get("seeds", ()))
    if methods != METHOD_IDS:
        raise IncompleteSuiteError("suite method order is not the frozen protocol")
    if partitions != PARTITION_IDS:
        raise IncompleteSuiteError("suite partition order is not the frozen protocol")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds):
        raise IncompleteSuiteError("suite seeds must be integers")
    seeds = tuple(int(seed) for seed in raw_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise IncompleteSuiteError("suite seeds must be nonempty and unique")
    return tuple(
        (partition, method, seed)
        for partition in partitions
        for method in methods
        for seed in seeds
    )


def validate_result_root(result_root: Path, *, formal: bool) -> ValidationReport:
    """Validate every run before statistics or paper figures are generated."""

    root = Path(result_root).resolve()
    manifest = _read_json(root / "suite_manifest.json", "suite manifest")
    suite = manifest.get("suite_config")
    runs = manifest.get("runs")
    if not isinstance(suite, dict) or not isinstance(runs, list):
        raise IncompleteSuiteError("suite manifest must contain suite_config and runs")
    expected = expected_run_identities(suite)
    expected_set = set(expected)
    observed: dict[tuple[str, str, int], dict[str, object]] = {}
    for run in runs:
        if not isinstance(run, dict):
            raise IncompleteSuiteError("run records must be JSON objects")
        try:
            identity = (
                str(run["partition"]),
                str(run["method"]),
                int(run["seed"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IncompleteSuiteError("malformed run identity") from error
        if identity in observed:
            raise IncompleteSuiteError(f"duplicate run identity: {identity}")
        observed[identity] = run
    missing = expected_set - set(observed)
    unexpected = set(observed) - expected_set
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"{len(missing)} missing")
        if unexpected:
            parts.append(f"{len(unexpected)} unexpected")
        raise IncompleteSuiteError(", ".join(parts) + " run identities")
    seeds = tuple(int(seed) for seed in suite["seeds"])
    if formal and seeds != tuple(range(30)):
        raise IncompleteSuiteError("formal evidence requires seeds 0 through 29")
    history_points = (
        51
        if suite.get("inference_status") == "formal_30_paired_seeds"
        else int(suite.get("n_rounds_override", 0)) + 1
    )
    for identity in expected:
        _validate_run(root, identity, observed[identity], history_points)
    return ValidationReport(
        passed=True,
        run_count=len(observed),
        expected_run_count=len(expected),
        history_points=history_points,
        inference_status=str(suite.get("inference_status", "")),
    )


def run_or_resume(
    base_path: Path,
    suite_path: Path,
    parameters_path: Path,
    output_root: Path,
) -> ValidationReport:
    """Validate inputs, execute/resume the suite, then validate every artifact."""

    protocol = load_and_validate_protocol(base_path, suite_path, parameters_path)
    from .paper_suite import run_paper_suite

    run_paper_suite(Path(suite_path), Path(output_root))
    formal = protocol.suite["inference_status"] == "formal_30_paired_seeds"
    return validate_result_root(output_root, formal=formal)


def _validate_run(
    root: Path,
    identity: tuple[str, str, int],
    record: Mapping[str, object],
    history_points: int,
) -> None:
    partition, method, seed = identity
    run_dir_raw = record.get("run_dir")
    if not isinstance(run_dir_raw, str) or not run_dir_raw:
        raise IncompleteSuiteError(f"missing run_dir for {identity}")
    declared = Path(run_dir_raw)
    run_dir = declared.resolve() if declared.is_absolute() else (root / declared).resolve()
    required = tuple(run_dir / name for name in (
        "history.csv", "summary.json", "config.json", "environment.json"
    ))
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise IncompleteSuiteError(f"missing artifacts for {identity}: {missing}")
    rows = _read_history(required[0], identity)
    if len(rows) != history_points:
        raise IncompleteSuiteError(
            f"history length for {identity} must be {history_points}"
        )
    summary = _read_json(required[1], f"summary for {identity}")
    if int(summary.get("gradient_evaluations", -1)) != 0:
        raise IncompleteSuiteError(f"analytic gradient use detected for {identity}")
    if float(summary.get("query_count", 0)) <= 0:
        raise IncompleteSuiteError(f"missing function queries for {identity}")
    if float(summary.get("transmitted_bytes", -1)) < 0:
        raise IncompleteSuiteError(f"invalid transmitted bytes for {identity}")
    if not np.isfinite(float(summary.get("wall_time", np.nan))):
        raise IncompleteSuiteError(f"invalid wall time for {identity}")
    metadata = summary.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("gradient_access") is not False:
        raise IncompleteSuiteError(f"black-box declaration missing for {identity}")
    expected_oracle = "symmetric_two_point" if method.startswith("zo2_") else "one_point"
    if metadata.get("gradient_oracle") != expected_oracle:
        raise IncompleteSuiteError(f"wrong oracle declaration for {identity}")
    if method == "spade_dfl_private" and metadata.get("vr_mode") != "persistent_saga":
        raise IncompleteSuiteError(f"SPADE persistent SAGA declaration missing for {identity}")
    if method in PRIVATE_METHODS:
        if summary.get("privacy_valid") is not True:
            raise IncompleteSuiteError(f"invalid privacy accounting for {identity}")
        epsilon = float(summary.get("epsilon", np.nan))
        delta = float(summary.get("delta", np.nan))
        if not np.isclose(epsilon, 8.0) or not np.isclose(delta, 1e-5):
            raise IncompleteSuiteError(f"wrong privacy budget for {identity}")
        if summary.get("privacy_scope") in {None, "none"}:
            raise IncompleteSuiteError(f"missing privacy scope for {identity}")


def _read_history(
    path: Path, identity: tuple[str, str, int]
) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise IncompleteSuiteError(f"cannot read history for {identity}") from error
    if not rows or not REQUIRED_HISTORY_COLUMNS <= set(rows[0]):
        raise IncompleteSuiteError(f"history columns missing for {identity}")
    iterations = []
    for row in rows:
        try:
            iterations.append(int(row["iteration"]))
            values = np.asarray(
                [float(row[name]) for name in REQUIRED_HISTORY_COLUMNS - {"iteration"}],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise IncompleteSuiteError(f"malformed history for {identity}") from error
        if not np.all(np.isfinite(values)):
            raise IncompleteSuiteError(f"non-finite history for {identity}")
        accuracy = float(row["accuracy"])
        if accuracy < 0.0 or accuracy > 1.0:
            raise IncompleteSuiteError(f"accuracy outside [0,1] for {identity}")
    if iterations != list(range(len(rows))):
        raise IncompleteSuiteError(f"non-canonical round axis for {identity}")
    return rows


def _read_json(path: Path, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IncompleteSuiteError(f"cannot read {name}") from error
    if not isinstance(payload, dict):
        raise IncompleteSuiteError(f"{name} must be a JSON object")
    return payload
