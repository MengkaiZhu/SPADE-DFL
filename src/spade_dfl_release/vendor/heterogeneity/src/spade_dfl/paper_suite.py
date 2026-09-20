from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .artifacts import (
    aggregate_results,
    load_run,
    save_aggregate,
    save_preprocessing,
    save_run,
)
from .config import ExperimentConfig, ParameterRecord, load_config
from .data import (
    DatasetBundle,
    load_binary_image_dataset,
    load_multiclass_image_dataset,
)
from .methods import METHOD_REGISTRY, method_config, run_method
from .metrics import paper_stationarity_consensus
from .partitioning import partition_labels

PANEL_ORDER = (
    "dir_0.3",
    "dir_0.6",
    "iid",
    "pathological",
)
DISTRIBUTION_STRESS_PANEL_ORDER = (
    *PANEL_ORDER,
    "dir_0.1",
    "pathological_pure",
)
# This helper retains the historical 2x2 manuscript summary figure. New
# distribution-stress figures are emitted one partition at a time instead.
LEGACY_PAPER_PANEL_ORDER = PANEL_ORDER
PANEL_TITLES = {
    "dir_0.3": "Non-IID: Dir 0.3",
    "dir_0.6": "Non-IID: Dir 0.6",
    "iid": "IID",
    "pathological": "Pathological",
    "dir_0.1": "Non-IID: Dir 0.1",
    "pathological_pure": "Pathological (Pure)",
}


def attach_emnist_accuracy_contract(
    assessment: dict[str, object],
    seed_rows: Sequence[Mapping[str, object]],
    *,
    n_classes: int,
    seeds: Sequence[int],
) -> dict[str, object]:
    """Attach the formal EMNIST contract and fail the suite closed."""

    from .emnist_accuracy import assess_emnist_accuracy_contract

    contract = assess_emnist_accuracy_contract(
        seed_rows,
        n_classes=n_classes,
        expected_partitions=PANEL_ORDER,
        expected_methods=BLACK_BOX_METHOD_IDS,
        expected_seeds=seeds,
    )
    formal_protocol_eligible = tuple(sorted(int(seed) for seed in seeds)) == tuple(
        range(30)
    )
    contract["formal_protocol_eligible"] = formal_protocol_eligible
    if not formal_protocol_eligible:
        contract["diagnostic_status"] = contract["status"]
        if contract["status"] == "accepted":
            contract["status"] = "diagnostic_only"
    assessment["emnist_accuracy_contract"] = contract
    if contract["status"] == "rejected":
        assessment["status"] = "emnist_accuracy_rejected"
    return assessment
METHOD_STYLES = {
    "Exact Gradient": {"color": "#f2a7d8", "linewidth": 1.3, "linestyle": "-"},
    "1P-DSGT": {"color": "#ff9e63", "linewidth": 1.4, "linestyle": "-"},
    "SPADE-ZO": {"color": "#9c88d4", "linewidth": 1.4, "linestyle": "-"},
    "ZO-ADMM": {"color": "#64b5f6", "linewidth": 1.4, "linestyle": "-"},
    "SPADE-DFL": {"color": "#ef5350", "linewidth": 2.2, "linestyle": "-"},
    "Private SPADE": {"color": "#8d6e63", "linewidth": 1.8, "linestyle": "--"},
    "Private SPADE-DFL": {"color": "#ef5350", "linewidth": 2.5, "linestyle": "-"},
    "Non-private SPADE-DFL": {"color": "#b71c1c", "linewidth": 2.2, "linestyle": "--"},
    "LT-ADMM-VR": {"color": "#64b5f6", "linewidth": 1.5, "linestyle": "-"},
    "LT-ADMM": {"color": "#90caf9", "linewidth": 1.5, "linestyle": "-."},
    "DP-SGD": {"color": "#81c784", "linewidth": 1.5, "linestyle": "-"},
    "DP-FedAvg": {"color": "#4db6ac", "linewidth": 1.5, "linestyle": "-"},
    "2P-LT-ADMM-VR": {"color": "#3182bd", "linewidth": 1.7, "linestyle": "-"},
    "2P-LT-ADMM": {"color": "#6baed6", "linewidth": 1.7, "linestyle": "--"},
    "2P-DP-SGD": {"color": "#31a354", "linewidth": 1.7, "linestyle": "-."},
    "2P-DP-FedAvg": {"color": "#74c476", "linewidth": 1.7, "linestyle": "--"},
    "1P-DSG": {"color": "#ffb74d", "linewidth": 1.5, "linestyle": "-"},
}
METHOD_LABELS = {
    method: spec.label for method, spec in METHOD_REGISTRY.items()
}
BLACK_BOX_LEGEND_LABELS = {
    "2P-LT-ADMM-VR": "LT-ADMM-VR",
    "2P-LT-ADMM": "LT-ADMM",
    "2P-DP-SGD": "DP-SGD",
    "2P-DP-FedAvg": "DP-FedAvg",
    "1P-DSG": "1P-DSG",
    "1P-DSGT": "1P-DSGT",
    "SPADE-DFL": "SPADE-DFL",
}
BLACK_BOX_LEGEND_ORDER = (
    "LT-ADMM-VR",
    "LT-ADMM",
    "DP-SGD",
    "DP-FedAvg",
    "1P-DSG",
    "1P-DSGT",
    "SPADE-DFL",
)
BLACK_BOX_METHOD_LABELS = (
    "2P-LT-ADMM-VR",
    "2P-LT-ADMM",
    "2P-DP-SGD",
    "2P-DP-FedAvg",
    "1P-DSG",
    "1P-DSGT",
    "SPADE-DFL",
)


def use_strict_black_box_accuracy_figure(
    *, comparison_environment: str, n_rounds: int
) -> bool:
    """Reserve the 51-point manuscript figure contract for 50-round suites."""
    return comparison_environment == "all_black_box" and n_rounds == 50


def method_protocol_declarations(
    base: ExperimentConfig,
    methods: Sequence[str],
    method_parameters: Optional[Mapping[str, object]] = None,
    protocol: Optional[str] = None,
) -> dict[str, dict[str, object]]:
    """Describe each method's actual outer-round and client-count semantics."""
    declarations: dict[str, dict[str, object]] = {}
    for method in methods:
        configured = configure_suite_method(
            base,
            method,
            base.seeds,
            "iid",
            0.3,
            protocol=protocol,
            override=(
                method_parameters[method]
                if method_parameters and method in method_parameters
                else None
            ),
        )
        centralized_dp_sgd = method in {"dp_sgd", "zo2_dp_sgd"}
        declarations[METHOD_REGISTRY[method].label] = {
            "configured_n_clients": configured.n_clients,
            "suite_base_n_clients": base.n_clients,
            "is_n_clients_exception": bool(
                configured.n_clients != base.n_clients
            ),
            "outer_round_semantics": (
                "centralized_optimization_step_no_communication"
                if centralized_dp_sgd
                else "one communication/aggregation event per outer round"
            ),
            "x_axis_caveat": (
                "centralized optimizer-step index; not a communication round"
                if centralized_dp_sgd
                else "outer-round index with communication/aggregation"
            ),
        }
    return declarations
BLACK_BOX_METHOD_IDS = (
    "zo2_lt_admm_vr",
    "zo2_lt_admm",
    "zo2_dp_sgd",
    "zo2_dp_fedavg",
    "one_point_dsg",
    "one_point_dsgt",
    "spade_dfl",
)
HISTORICAL_METHOD_LABELS = {
    "Private SPADE-DFL",
    "Non-private SPADE-DFL",
    "LT-ADMM-VR",
    "LT-ADMM",
    "DP-SGD",
    "DP-FedAvg",
    "1P-DSG",
    "1P-DSGT",
}
ANALYTIC_TRAINING_METHODS = {"lt_admm", "lt_admm_vr", "dp_sgd", "dp_fedavg"}
D20_PROTOCOL = "cifar10_d20_50round_black_box"
D20_FULL_REFRESH_PROTOCOL = "cifar10_d20_50round_spade_full_refresh"
MNIST67_PAPER_PROTOCOL = "mnist67_paper_strong_dp_v1"


def validate_d20_suite_contract(
    protocol: Optional[str],
    methods: Sequence[str],
    method_parameters: Mapping[str, object],
) -> None:
    """Fail closed for the isolated d20 protocol and ignore legacy suites."""

    if protocol in {None, ""}:
        return
    if protocol == MNIST67_PAPER_PROTOCOL:
        from .mnist67_paper_protocol import METHOD_IDS

        if tuple(methods) != METHOD_IDS:
            raise ValueError(
                "MNIST 6/7 paper protocol requires the exact ordered seven methods"
            )
        if tuple(method_parameters) != METHOD_IDS:
            raise ValueError(
                "MNIST 6/7 paper protocol requires frozen parameters for all methods"
            )
        return
    if protocol not in {D20_PROTOCOL, D20_FULL_REFRESH_PROTOCOL}:
        raise ValueError(f"unsupported protocol: {protocol}")
    from .cifar_d20_protocol import D20_METHOD_IDS

    if tuple(methods) != D20_METHOD_IDS:
        raise ValueError("d20 protocol requires the exact ordered seven methods")
    if not method_parameters:
        raise ValueError("d20 protocol requires frozen parameters")
    if tuple(method_parameters) != D20_METHOD_IDS:
        raise ValueError("d20 frozen parameters require the exact ordered seven methods")


def configure_suite_method(
    base: ExperimentConfig,
    method: str,
    seeds: tuple[int, ...],
    partition_mode: str,
    alpha: float,
    *,
    protocol: Optional[str] = None,
    override: Optional[Mapping[str, object]] = None,
) -> ExperimentConfig:
    """Dispatch only the explicitly named d20 suite to its strict protocol."""

    if protocol == D20_PROTOCOL:
        from .cifar_d20_protocol import configure_d20_method

        return configure_d20_method(
            base,
            method,
            seeds,
            partition_mode,
            alpha,
            override,
        )
    if protocol == D20_FULL_REFRESH_PROTOCOL:
        from .cifar_d20_spade_full_refresh import configure_full_refresh_method

        if override is None:
            raise ValueError("full-refresh d20 protocol requires frozen parameters")
        return configure_full_refresh_method(
            base,
            method,
            seeds,
            partition_mode,
            alpha,
            override,
        )
    if protocol not in {None, "", MNIST67_PAPER_PROTOCOL}:
        raise ValueError(f"unsupported protocol: {protocol}")
    configured = method_config(base, method, seeds, partition_mode, alpha)
    if override is not None:
        configured = apply_method_parameters(configured, method, override)
    return configured


def suite_partition_client_count(
    method: str,
    config: ExperimentConfig,
    base: ExperimentConfig,
    *,
    protocol: Optional[str] = None,
) -> int:
    """Return the client count used to define the data-distribution scenario.

    DP-SGD is a centralized example-level baseline and therefore trains with a
    single model.  Under the MNIST 6/7 paper protocol, however, its scenario is
    still the same frozen 1,000-client partition used to define label
    heterogeneity for every method.  The DP-SGD implementation does not consume
    this partition; it is retained as auditable environment metadata.
    """

    if protocol == MNIST67_PAPER_PROTOCOL and method == "zo2_dp_sgd":
        return base.n_clients
    return config.n_clients


def manifest_run_dir(result_root: Path, run_dir: Path) -> str:
    """Serialize a run path relative to its manifest's result root."""

    root = Path(result_root).resolve()
    path = Path(run_dir).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("run directory must be contained in the result root") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_panel_sequence(
    panels: Sequence[str], *, protocol: Optional[str] = None
) -> tuple[str, ...]:
    values = tuple(panels)
    if not values:
        raise ValueError("at least one partition is required")
    if len(set(values)) != len(values):
        raise ValueError("partitions must be unique")
    if any(value not in DISTRIBUTION_STRESS_PANEL_ORDER for value in values):
        raise ValueError(
            "partitions must be selected from "
            f"{DISTRIBUTION_STRESS_PANEL_ORDER}"
        )
    if protocol == MNIST67_PAPER_PROTOCOL:
        from .mnist67_paper_protocol import PARTITION_IDS

        canonical = tuple(value for value in PARTITION_IDS if value in values)
    else:
        canonical = tuple(
            value for value in DISTRIBUTION_STRESS_PANEL_ORDER if value in values
        )
    if values != canonical:
        raise ValueError(
            "partitions must follow canonical order "
            f"{canonical}"
        )
    return values


def partition_seed_index(panel: str) -> int:
    """Return a canonical partition RNG index independent of suite slicing."""

    if panel not in DISTRIBUTION_STRESS_PANEL_ORDER:
        raise ValueError(f"unknown partition {panel}")
    return DISTRIBUTION_STRESS_PANEL_ORDER.index(panel)


def apply_method_parameters(
    config: ExperimentConfig,
    method: str,
    override: Mapping[str, object],
) -> ExperimentConfig:
    parameters = override.get("parameters")
    provenance = override.get("provenance")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError(f"{method} method parameters must be nonempty")
    if not isinstance(provenance, Mapping) or set(parameters) != set(provenance):
        raise ValueError(f"{method} provenance must match every overridden parameter")
    unknown = set(parameters) - set(ExperimentConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"{method} has unknown parameters: {sorted(unknown)}")
    sources = dict(config.parameter_sources)
    for name, value in parameters.items():
        source = provenance[name]
        if not isinstance(source, Mapping):
            raise ValueError(f"invalid provenance for {method}/{name}")
        sources[name] = ParameterRecord(name, value, **source)
    return replace(config, parameter_sources=sources, **dict(parameters))


def resolve_method_parameters(
    raw: Mapping[str, object],
    suite_config_path: Path,
    expected_methods: Optional[Sequence[str]] = None,
) -> Mapping[str, object]:
    inline = raw.get("method_parameters")
    source = raw.get("method_parameters_from")
    if inline is not None and source is not None:
        raise ValueError("set only one of method_parameters or method_parameters_from")
    if source is not None:
        source_path = _resolve_path(str(source), Path(suite_config_path).parent)
        selected_raw = json.loads(source_path.read_text(encoding="utf-8"))
        inline = _validate_frozen_method_parameters(
            selected_raw,
            expected_methods=expected_methods,
        )
    if inline is None:
        return {}
    if not isinstance(inline, Mapping):
        raise ValueError("method_parameters must be a mapping")
    return inline


def _validate_frozen_method_parameters(
    frozen: object,
    expected_methods: Optional[Sequence[str]] = None,
) -> Mapping[str, object]:
    """Fail closed when loading the training-holdout selection artifact."""
    if not isinstance(frozen, Mapping):
        raise ValueError("frozen parameter file must be a JSON mapping")
    if frozen.get("status") != "accepted":
        raise ValueError("frozen parameter status must be accepted")
    if frozen.get("official_test_used_for_selection") is not False:
        raise ValueError(
            "frozen official_test_used_for_selection must be exactly false"
        )
    methods = frozen.get("method_parameters")
    if not isinstance(methods, Mapping):
        raise ValueError("frozen method_parameters must be a mapping")
    expected = tuple(expected_methods or ("spade_dfl",))
    if tuple(methods) != expected:
        expected_label = (
            "spade_dfl"
            if expected_methods is None
            else "the selected methods"
        )
        raise ValueError(
            f"frozen method_parameters must contain exactly {expected_label}"
        )
    for method, override in methods.items():
        if not isinstance(override, Mapping):
            raise ValueError(f"frozen {method} override must be a mapping")
        parameters = override.get("parameters")
        provenance = override.get("provenance")
        if not isinstance(parameters, Mapping) or not parameters:
            raise ValueError(f"frozen {method} parameters must be nonempty")
        if not isinstance(provenance, Mapping) or set(provenance) != set(parameters):
            raise ValueError(
                f"frozen {method} provenance must match every parameter"
            )
        unknown = set(parameters) - set(ExperimentConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"frozen {method} has unknown parameters: "
                + ", ".join(sorted(unknown))
            )
        for name, value in parameters.items():
            source = provenance[name]
            if not isinstance(source, Mapping) or set(source) != {
                "source_type",
                "source",
            }:
                raise ValueError(f"invalid provenance for frozen {method}/{name}")
            if not isinstance(source["source"], str) or not source["source"].strip():
                raise ValueError(f"invalid provenance for frozen {method}/{name}")
            try:
                ParameterRecord(name, value, **dict(source))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid provenance for frozen {method}/{name}"
                ) from error
    return methods


def validate_black_box_method_set(
    methods: Sequence[str], *, require_exact: bool = True
) -> None:
    forbidden = sorted(set(methods) & ANALYTIC_TRAINING_METHODS)
    if forbidden:
        raise ValueError(
            "all_black_box suite contains analytic-gradient training methods: "
            + ", ".join(forbidden)
        )
    if require_exact and tuple(methods) != BLACK_BOX_METHOD_IDS:
        raise ValueError(
            "all_black_box suite must use the exact ordered seven-method contract: "
            + ", ".join(BLACK_BOX_METHOD_IDS)
        )
    if not require_exact and (
        not methods or any(method not in BLACK_BOX_METHOD_IDS for method in methods)
    ):
        raise ValueError(
            "black-box tuning methods must be a nonempty subset of the seven-method contract"
        )


def assess_black_box_competitiveness(
    final_rows: Sequence[Mapping[str, object]],
    partitions: Sequence[str],
) -> dict[str, object]:
    """Report whether the revised single SPADE curve beats all baselines.

    This is an assessment of observed raw aggregate results.  It never changes,
    smooths, clips, or reorders the accuracy curves.
    """
    baseline_labels = {
        "2P-LT-ADMM",
        "2P-LT-ADMM-VR",
        "2P-DP-SGD",
        "2P-DP-FedAvg",
        "1P-DSG",
        "1P-DSGT",
    }
    per_partition = {}
    for partition in partitions:
        rows_by_method = {
            str(row["method"]): row
            for row in final_rows
            if str(row["partition"]) == partition
        }
        accuracy = {
            name: float(row["final_accuracy"])
            for name, row in rows_by_method.items()
        }
        required = {"SPADE-DFL", *baseline_labels}
        complete = required.issubset(accuracy)
        best_baseline = (
            max(baseline_labels, key=lambda name: accuracy[name])
            if complete
            else None
        )
        baseline_ceiling = accuracy[best_baseline] if best_baseline else None
        margin = (
            accuracy["SPADE-DFL"] - baseline_ceiling if complete else None
        )
        spade_beats_all = bool(complete and margin > 0.0)
        per_partition[partition] = {
            "complete": complete,
            "best_baseline": best_baseline,
            "baseline_ceiling": baseline_ceiling,
            "spade_final_accuracy": accuracy.get("SPADE-DFL"),
            "spade_margin_over_best_baseline": margin,
            "spade_beats_all_baselines": spade_beats_all,
        }
    all_partitions_spade_beats_all = bool(
        per_partition
        and all(row["spade_beats_all_baselines"] for row in per_partition.values())
    )
    return {
        "status": (
            "spade_effectiveness_verified"
            if all_partitions_spade_beats_all
            else "spade_effectiveness_not_verified"
        ),
        "per_partition": per_partition,
        "all_partitions_spade_beats_all_baselines": all_partitions_spade_beats_all,
        "interpretation": (
            "Observed final-accuracy ranking only; no curve processing or "
            "test-set-driven parameter selection is applied."
        ),
    }


def rounds_to_target(values: np.ndarray, target: float) -> int:
    reached = np.flatnonzero(np.asarray(values) >= target)
    return int(reached[0]) if reached.size else -1


def assess_paper_suite(
    final_rows: Sequence[Mapping[str, object]],
    target_rows: Sequence[Mapping[str, object]],
    n_rounds: int,
    seed_count: int,
    target_accuracy: float,
    expected_method_labels: Optional[Sequence[str]] = None,
    reference_baseline_labels: Optional[Sequence[str]] = None,
    private_method_labels: Optional[Sequence[str]] = None,
    declared_privacy_budgets: Optional[
        Mapping[str, tuple[float, float]]
    ] = None,
) -> dict[str, object]:
    preferred_label = (
        "Non-private SPADE-DFL"
        if any(row["method"] == "Non-private SPADE-DFL" for row in final_rows)
        else "SPADE-DFL"
    )
    spade_final = {
        str(row["partition"]): float(row["final_accuracy"])
        for row in final_rows
        if row["method"] == preferred_label
    }
    spade_target = {
        str(row["partition"]): int(row["first_round"])
        for row in target_rows
        if row["method"] == preferred_label
    }
    complete_panels = set(spade_final) == set(PANEL_ORDER) == set(spade_target)
    finite = complete_panels and all(np.isfinite(value) for value in spade_final.values())
    all_at_target = finite and all(
        value >= target_accuracy for value in spade_final.values()
    )
    all_reached = complete_panels and all(value >= 0 for value in spade_target.values())
    expected_labels = set(expected_method_labels or HISTORICAL_METHOD_LABELS)
    expected_pairs = {
        (partition, label) for partition in PANEL_ORDER for label in expected_labels
    }
    present_pairs = {
        (str(row["partition"]), str(row["method"])) for row in final_rows
    }
    complete_methods = expected_pairs.issubset(present_pairs)
    required_numeric = (
        "initial_accuracy",
        "final_accuracy",
        "initial_loss",
        "final_loss",
        "stationarity",
    )
    all_method_metrics_finite = complete_methods and all(
        all(
            name in row and np.isfinite(float(row[name]))
            for name in required_numeric
        )
        for row in final_rows
    )
    reference_baseline_labels = set(
        reference_baseline_labels
        or {"LT-ADMM-VR", "LT-ADMM", "1P-DSG", "1P-DSGT"}
    )
    baseline_rows = [
        row for row in final_rows if row["method"] in reference_baseline_labels
    ]
    reference_baselines_show_learning = (
        len(baseline_rows) == len(reference_baseline_labels) * len(PANEL_ORDER)
        and all(
            float(row["final_loss"]) < float(row["initial_loss"])
            or float(row["final_accuracy"]) > float(row["initial_accuracy"])
            for row in baseline_rows
        )
    )
    private_labels = set(
        private_method_labels
        or {"Private SPADE-DFL", "DP-SGD", "DP-FedAvg"}
    )
    budgets = dict(
        declared_privacy_budgets
        or {label: (8.0, 1e-5) for label in private_labels}
    )
    private_labels.update(budgets)
    private_rows = [row for row in final_rows if row["method"] in private_labels]

    def declared_budget_valid(label: str, epsilon: float, delta: float) -> bool:
        method_rows = [
            row for row in private_rows if str(row["method"]) == label
        ]
        return (
            {str(row["partition"]) for row in method_rows} == set(PANEL_ORDER)
            and all(
                row.get("epsilon") is not None
                and np.isclose(
                    float(row["epsilon"]), epsilon, rtol=1e-10, atol=1e-10
                )
                and np.isclose(
                    float(row.get("delta", 0.0)),
                    delta,
                    rtol=1e-10,
                    atol=1e-12,
                )
                and bool(row.get("privacy_valid", True))
                for row in method_rows
            )
        )

    privacy_budgets_valid = all(
        declared_budget_valid(label, epsilon, delta)
        for label, (epsilon, delta) in budgets.items()
    )
    private_spade_rows = [
        row for row in final_rows if row["method"] == "Private SPADE-DFL"
    ]
    private_joint_metric_improved = (
        len(private_spade_rows) == len(PANEL_ORDER)
        and all(
            "initial_stationarity_consensus" in row
            and "stationarity_consensus" in row
            and np.isfinite(float(row["initial_stationarity_consensus"]))
            and np.isfinite(float(row["stationarity_consensus"]))
            and float(row["stationarity_consensus"])
            <= float(row["initial_stationarity_consensus"])
            for row in private_spade_rows
        )
    )
    private_not_fully_clipped = (
        len(private_spade_rows) == len(PANEL_ORDER)
        and all(
            "mean_clip_fraction" in row
            and np.isfinite(float(row["mean_clip_fraction"]))
            and float(row["mean_clip_fraction"]) < 1.0
            for row in private_spade_rows
        )
    )
    private_utility_verified = (
        len(private_spade_rows) == len(PANEL_ORDER)
        and {str(row["partition"]) for row in private_spade_rows}
        == set(PANEL_ORDER)
        and all(
            all(
                name in row and np.isfinite(float(row[name]))
                for name in (
                    "initial_accuracy",
                    "final_accuracy",
                    "initial_loss",
                    "final_loss",
                    "initial_stationarity",
                    "stationarity",
                    "initial_consensus_error",
                    "consensus_error",
                    "initial_stationarity_consensus",
                    "stationarity_consensus",
                    "mean_clip_fraction",
                )
            )
            and float(row["final_accuracy"]) > float(row["initial_accuracy"])
            and float(row["final_loss"]) < float(row["initial_loss"])
            and float(row["stationarity"]) <= float(row["initial_stationarity"])
            for row in private_spade_rows
        )
        and private_joint_metric_improved
        and private_not_fully_clipped
        and n_rounds >= 2
    )
    revised_single_spade = (
        "SPADE-DFL" in expected_labels
        and "Private SPADE-DFL" not in expected_labels
        and "Non-private SPADE-DFL" not in expected_labels
    )
    spade_rows = [row for row in final_rows if row["method"] == "SPADE-DFL"]
    spade_optimization_fields = {
        "initial_accuracy",
        "final_accuracy",
        "initial_loss",
        "final_loss",
        "initial_stationarity_consensus",
        "stationarity_consensus",
    }
    spade_optimization_improved = (
        len(spade_rows) == len(PANEL_ORDER)
        and {str(row["partition"]) for row in spade_rows} == set(PANEL_ORDER)
        and all(spade_optimization_fields.issubset(row) for row in spade_rows)
        and all(
            float(row["final_accuracy"]) > float(row["initial_accuracy"])
            and float(row["final_loss"]) < float(row["initial_loss"])
            and float(row["stationarity_consensus"])
            <= float(row["initial_stationarity_consensus"])
            for row in spade_rows
        )
    )
    local_verified = bool(all_at_target and all_reached)
    comparison_complete = bool(
        local_verified
        and complete_methods
        and all_method_metrics_finite
        and reference_baselines_show_learning
        and privacy_budgets_valid
    )
    if revised_single_spade:
        comparison_complete = bool(
            comparison_complete and spade_optimization_improved
        )
    complete_local = bool(
        comparison_complete
        and (spade_optimization_improved if revised_single_spade else private_utility_verified)
    )
    paper_scale_completed = bool(
        comparison_complete
        and n_rounds >= (50 if revised_single_spade else 500)
        and seed_count >= 30
    )
    paper_scale = bool(
        paper_scale_completed
        and (spade_optimization_improved if revised_single_spade else private_utility_verified)
    )
    if paper_scale:
        status = "paper_scale_verified"
    elif paper_scale_completed:
        status = "paper_scale_completed_private_utility_failed"
    elif complete_local:
        status = "complete_baselines_local_verified"
    elif comparison_complete:
        status = "complete_baselines_local_private_utility_failed"
    elif complete_methods and local_verified:
        status = "local_efficacy_verified"
    else:
        status = "mechanism_verified"
    thresholds = {
        "complete_four_partitions": complete_panels,
        "spade_metrics_finite": finite,
        "spade_all_partitions_at_target": bool(all_at_target),
        "spade_all_partitions_reached_target": bool(all_reached),
        "server_rounds_at_least_50": n_rounds >= 50,
        "paper_seeds_at_least_30": seed_count >= 30,
        "complete_expected_methods": complete_methods,
        "expected_method_count": len(expected_labels),
        "all_method_metrics_finite": all_method_metrics_finite,
        "reference_baselines_show_learning": reference_baselines_show_learning,
        "declared_private_budgets_valid": privacy_budgets_valid,
        "spade_optimization_improved": bool(spade_optimization_improved),
    }
    if not revised_single_spade:
        thresholds.update(
            {
                "paper_rounds_at_least_500": n_rounds >= 500,
                "complete_eight_methods": complete_methods,
                "private_joint_metric_improved": bool(private_joint_metric_improved),
                "private_not_fully_clipped": bool(private_not_fully_clipped),
                "private_uses_multiple_rounds": n_rounds >= 2,
                "private_utility_verified": bool(private_utility_verified),
            }
        )
    return {
        "status": status,
        "n_rounds": int(n_rounds),
        "seed_count": int(seed_count),
        "target_accuracy": float(target_accuracy),
        "spade_final_accuracy": spade_final,
        "spade_first_target_round": spade_target,
        "thresholds": thresholds,
    }


def write_rows(rows: Sequence[Mapping[str, object]], path: Path) -> Path:
    if not rows:
        raise ValueError("at least one row is required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_paper_panels(
    nested_aggregates: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    metric: str,
    output_base: Path,
) -> tuple[Path, Path]:
    if metric not in {"accuracy", "loss"}:
        raise ValueError("metric must be accuracy or loss")
    if tuple(nested_aggregates) != LEGACY_PAPER_PANEL_ORDER:
        raise ValueError(
            "plot_paper_panels supports exactly the legacy four panels: "
            f"{LEGACY_PAPER_PANEL_ORDER}"
        )
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=False)
    for axis, panel in zip(axes.flat, LEGACY_PAPER_PANEL_ORDER):
        methods = nested_aggregates[panel]
        for method, aggregate in methods.items():
            mean = np.asarray(aggregate[f"{metric}_mean"], dtype=np.float64)
            std = np.asarray(aggregate[f"{metric}_std"], dtype=np.float64)
            scale = 100.0 if metric == "accuracy" else 1.0
            x = np.arange(mean.size)
            style = METHOD_STYLES.get(
                method, {"color": "#9e9e9e", "linewidth": 1.3, "linestyle": "-"}
            )
            axis.plot(x, mean * scale, label=method, **style)
            axis.fill_between(
                x,
                (mean - std) * scale,
                (mean + std) * scale,
                color=style["color"],
                alpha=0.08,
                linewidth=0,
            )
        axis.set_title(PANEL_TITLES[panel], fontsize=10)
        axis.set_xlabel("Communication Round")
        axis.set_ylabel("Test Accuracy (%)" if metric == "accuracy" else "Training Objective")
        axis.grid(True, alpha=0.22, linewidth=0.6)
        if metric == "accuracy":
            axis.set_ylim(45, 100)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_single_partition_curves(
    aggregates: Mapping[str, Mapping[str, np.ndarray]],
    metric: str,
    partition: str,
    output_base: Path,
) -> tuple[Path, Path]:
    if metric not in {"accuracy", "loss", "stationarity"}:
        raise ValueError("metric must be accuracy, loss, or stationarity")
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.2, 5.0))
    for method, aggregate in aggregates.items():
        mean = np.asarray(aggregate[f"{metric}_mean"], dtype=np.float64)
        std = np.asarray(aggregate[f"{metric}_std"], dtype=np.float64)
        scale = 100.0 if metric == "accuracy" else 1.0
        x = np.arange(mean.size)
        style = METHOD_STYLES.get(
            method, {"color": "#9e9e9e", "linewidth": 1.4, "linestyle": "-"}
        )
        plotted = np.maximum(mean * scale, 1e-16) if metric == "stationarity" else mean * scale
        axis.plot(x, plotted, label=method, **style)
        lower = (mean - std) * scale
        upper = (mean + std) * scale
        if metric == "stationarity":
            lower = np.maximum(lower, 1e-16)
            upper = np.maximum(upper, 1e-16)
        axis.fill_between(x, lower, upper, color=style["color"], alpha=0.07, linewidth=0)
    axis.set_title(PANEL_TITLES.get(partition, partition))
    axis.set_xlabel("Communication Round")
    last_round = max(
        np.asarray(aggregate[f"{metric}_mean"]).size - 1
        for aggregate in aggregates.values()
    )
    axis.set_xlim((0, last_round) if last_round > 0 else (-0.5, 0.5))
    labels = {
        "accuracy": "Test Accuracy (%)",
        "loss": "Training Objective",
        "stationarity": "Stationarity ||grad F(x_bar)||^2",
    }
    axis.set_ylabel(labels[metric])
    if metric == "accuracy":
        axis.set_ylim(0, 100)
    if metric == "stationarity":
        axis.set_yscale("log")
    axis.grid(True, alpha=0.22, linewidth=0.6)
    axis.legend(loc="best", fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_spade_distribution_comparison(
    aggregates: Mapping[str, Mapping[str, np.ndarray]],
    output_base: Path,
    *,
    seed_count: int,
    dataset_name: str,
) -> tuple[Path, Path, Path, Path]:
    """Plot unprocessed SPADE-DFL accuracy across four client distributions."""

    partition_order = ("iid", "dir_0.6", "dir_0.3", "pathological")
    if set(aggregates) != set(partition_order):
        raise ValueError("aggregates must contain exactly the four distributions")
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    if not dataset_name.strip():
        raise ValueError("dataset_name must be nonempty")

    normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    curve_lengths = set()
    for partition in partition_order:
        mean = np.asarray(aggregates[partition]["accuracy_mean"], dtype=np.float64)
        std = np.asarray(aggregates[partition]["accuracy_std"], dtype=np.float64)
        curve_lengths.update((mean.size, std.size))
        if (
            mean.ndim != 1
            or std.ndim != 1
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(std))
            or np.any(std < 0.0)
        ):
            raise ValueError("accuracy curves must be finite and share one length")
        normalized[partition] = (mean, std)
    if len(curve_lengths) != 1 or next(iter(curve_lengths), 0) < 2:
        raise ValueError("accuracy curves must be finite and share one length")
    curve_length = next(iter(curve_lengths))
    if curve_length != 51:
        raise ValueError("the frozen 50-round comparison requires exactly 51 points")
    if any(
        np.any(mean - std < 0.0) or np.any(mean + std > 1.0)
        for mean, std in normalized.values()
    ):
        raise ValueError("accuracy mean plus or minus standard deviation must lie in [0, 1]")

    labels = {
        "iid": "IID",
        "dir_0.6": "Dirichlet 0.6",
        "dir_0.3": "Dirichlet 0.3",
        "pathological": "Pathological",
    }
    styles = {
        "iid": {"color": "#3182bd", "linestyle": "-", "marker": "o"},
        "dir_0.6": {"color": "#6baed6", "linestyle": "--", "marker": "s"},
        "dir_0.3": {"color": "#ffb74d", "linestyle": "-.", "marker": "^"},
        "pathological": {"color": "#ef5350", "linestyle": ":", "marker": "D"},
    }
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    ):
        fig, axis = plt.subplots(figsize=(8.2, 5.0))
        n_rounds = curve_length - 1
        markevery = max(1, n_rounds // 10)
        x = np.arange(n_rounds + 1)
        for partition in partition_order:
            mean, std = normalized[partition]
            style = styles[partition]
            axis.plot(
                x,
                mean * 100.0,
                label=labels[partition],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.3,
                marker=style["marker"],
                markersize=5.2,
                markevery=markevery,
            )
            axis.fill_between(
                x,
                (mean - std) * 100.0,
                (mean + std) * 100.0,
                color=style["color"],
                alpha=0.10,
                linewidth=0,
            )
        axis.set_title(
            f"SPADE-DFL on {dataset_name} across Data Distributions", fontsize=13
        )
        axis.set_xlabel("Communication round", fontsize=11)
        axis.set_ylabel("Test accuracy (%)", fontsize=11)
        axis.set_xlim(0, n_rounds)
        lower = min(float(np.min(mean - std)) for mean, std in normalized.values())
        upper = max(float(np.max(mean + std)) for mean, std in normalized.values())
        axis.set_ylim(max(0.0, lower * 100.0 - 2.0), min(100.0, upper * 100.0 + 2.0))
        axis.grid(True, alpha=0.22, linewidth=0.6)
        axis.legend(loc="lower right", fontsize=9, ncol=1)
        axis.text(
            0.985,
            0.29,
            f"mean ± SD, n={seed_count}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#616161",
        )
        axis.tick_params(labelsize=9)
        fig.tight_layout()

        png = output_base.with_suffix(".png")
        pdf = output_base.with_suffix(".pdf")
        svg = output_base.with_suffix(".svg")
        fig.savefig(png, dpi=320, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.close(fig)

    sidecar = output_base.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "figure": "SPADE-DFL accuracy across client-data distributions",
                "dataset": dataset_name,
                "seed_count": int(seed_count),
                "center": "mean across seeds",
                "spread": "population standard deviation across seeds",
                "curve_processing": "none",
                "partition_order": list(partition_order),
                "partition_semantics": {
                    "iid": "random equal-size split",
                    "dir_0.6": "balanced Dirichlet alpha=0.6",
                    "dir_0.3": "balanced Dirichlet alpha=0.3",
                    "pathological": "binary mixed pathological split with minority target 0.1",
                },
                "source_data": {
                    partition: {
                        "accuracy_mean": normalized[partition][0].tolist(),
                        "accuracy_std": normalized[partition][1].tolist(),
                    }
                    for partition in partition_order
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return png, pdf, svg, sidecar


def plot_black_box_accuracy(
    aggregates: Mapping[str, Mapping[str, np.ndarray]],
    output_base: Path,
    *,
    seed_count: int,
    partition: str,
    dataset_metadata: Optional[Mapping[str, object]] = None,
    dataset_name: Optional[str] = None,
    negative_class: Optional[int] = None,
    negative_class_name: Optional[str] = None,
    positive_class: Optional[int] = None,
    positive_class_name: Optional[str] = None,
) -> tuple[Path, Path, Path, Path]:
    """Export the primary all-black-box accuracy comparison and its caption data.

    The outer-round axis is a communication round for decentralized methods.  For
    centralized DP-SGD it is an optimization step, which is recorded explicitly in
    the caption sidecar instead of being presented as a communication claim.
    """
    if not aggregates:
        raise ValueError("accuracy plot requires at least one method")
    if seed_count < 1:
        raise ValueError("seed_count must be positive")

    dataset_values = _normalize_dataset_figure_metadata(
        dataset_metadata,
        dataset_name=dataset_name,
        negative_class=negative_class,
        negative_class_name=negative_class_name,
        positive_class=positive_class,
        positive_class_name=positive_class_name,
    )

    style_by_label = {spec.label: spec for spec in METHOD_REGISTRY.values()}
    missing_styles = sorted(set(aggregates) - set(style_by_label))
    if missing_styles:
        raise ValueError("missing registered plot style: " + ", ".join(missing_styles))

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    lengths = set()
    prepared = {}
    for label, aggregate in aggregates.items():
        mean = np.asarray(aggregate["accuracy_mean"], dtype=np.float64)
        std = np.asarray(aggregate["accuracy_std"], dtype=np.float64)
        if mean.ndim != 1 or std.shape != mean.shape or mean.size == 0:
            raise ValueError(f"invalid accuracy aggregate for {label}")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError(f"non-finite accuracy aggregate for {label}")
        lengths.add(mean.size)
        prepared[label] = (mean, std)
    if len(lengths) != 1:
        raise ValueError("all accuracy curves must have the same number of rounds")
    if dataset_values["name"] == "cifar10":
        if (
            dataset_values["task_type"] != "multiclass"
            or int(dataset_values["n_classes"]) != 10
        ):
            raise ValueError("CIFAR-10 accuracy figures require the ten-class task")
        if set(prepared) != set(BLACK_BOX_METHOD_LABELS) or len(prepared) != 7:
            raise ValueError(
                "CIFAR-10 accuracy figures require the exact seven-method contract"
            )
        if next(iter(lengths)) != 51:
            raise ValueError(
                "CIFAR-10 accuracy figures require raw rounds 0..50 (51 points)"
            )

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
    with plt.rc_context(rc):
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        lower_envelope = []
        upper_envelope = []
        plot_order = sorted(
            prepared,
            key=lambda label: (
                BLACK_BOX_LEGEND_ORDER.index(
                    BLACK_BOX_LEGEND_LABELS.get(label, label)
                )
                if BLACK_BOX_LEGEND_LABELS.get(label, label)
                in BLACK_BOX_LEGEND_ORDER
                else len(BLACK_BOX_LEGEND_ORDER),
                label,
            ),
        )
        for label in plot_order:
            mean, std = prepared[label]
            spec = style_by_label[label]
            x = np.arange(mean.size)
            marker_every = max(1, mean.size // 10)
            plot_options = {
                "color": spec.color,
                "linestyle": spec.linestyle,
                "linewidth": spec.linewidth,
            }
            if spec.marker:
                plot_options.update(
                    marker=spec.marker,
                    markevery=marker_every,
                    markersize=4.0,
                    markeredgewidth=0.5,
                )
            axis.plot(
                x,
                100.0 * mean,
                label=BLACK_BOX_LEGEND_LABELS.get(label, label),
                **plot_options,
            )
            axis.fill_between(
                x,
                100.0 * (mean - std),
                100.0 * (mean + std),
                color=spec.color,
                alpha=0.10,
                linewidth=0,
            )
            lower_envelope.append(float(np.min(100.0 * (mean - std))))
            upper_envelope.append(float(np.max(100.0 * (mean + std))))

        chance_accuracy = 1.0 / float(dataset_values["n_classes"])
        axis.axhline(
            100.0 * chance_accuracy,
            color="#777777",
            linestyle=(0, (3, 2)),
            linewidth=0.9,
            label="_nolegend_",
            zorder=0,
        )
        axis.set_xlabel("Communication round")
        axis.set_ylabel("Test accuracy (%)")
        panel_title = PANEL_TITLES.get(partition, partition)
        if dataset_metadata is not None or dataset_name is not None:
            if dataset_values["task_type"] == "multiclass":
                panel_title = (
                    f"{dataset_values['display_name']} "
                    f"{dataset_values['n_classes']}-class\n{panel_title}"
                )
            else:
                task = dataset_values["binary_task"]
                panel_title = (
                    f"{dataset_values['display_name']} "
                    f"{task['negative_class']} ({task['negative_class_name']}) vs "
                    f"{task['positive_class']} ({task['positive_class_name']}) binary\n"
                    f"{panel_title}"
                )
        axis.set_title(panel_title, fontsize=9)
        final_round = next(iter(lengths)) - 1
        axis.set_xlim((0, final_round) if final_round else (-0.5, 0.5))
        if dataset_values["name"] == "cifar10":
            axis.set_xticks(np.arange(0, 51, 10))
        observed_lower = min(lower_envelope)
        y_lower = max(0.0, min(45.0, 5.0 * np.floor(observed_lower / 5.0)))
        if dataset_values["task_type"] == "multiclass":
            chance_percent = 100.0 * chance_accuracy
            chance_margin = max(1.0, 0.1 * chance_percent)
            y_lower = min(y_lower, max(0.0, chance_percent - chance_margin))
            observed_upper = max(upper_envelope)
            y_upper = min(
                100.0,
                max(20.0, 10.0 * np.ceil((observed_upper + 2.0) / 10.0)),
            )
        else:
            y_upper = 100.0
        axis.set_ylim(y_lower, y_upper)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.55, alpha=0.65)
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
            borderaxespad=0,
        )
        fig.subplots_adjust(left=0.11, right=0.70, bottom=0.15, top=0.91)

        png = output_base.with_suffix(".png")
        pdf = output_base.with_suffix(".pdf")
        svg = output_base.with_suffix(".svg")
        fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        fig.savefig(svg, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    visible_legend_entries = [
        BLACK_BOX_LEGEND_LABELS.get(label, label) for label in plot_order
    ]
    source_data = output_base.with_suffix(".source_data.csv")
    with source_data.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "partition",
                "method",
                "round",
                "accuracy_mean",
                "accuracy_std",
                "seed_count",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for label in plot_order:
            mean, std = prepared[label]
            display_label = BLACK_BOX_LEGEND_LABELS.get(label, label)
            for round_index, (center, spread) in enumerate(zip(mean, std)):
                writer.writerow(
                    {
                        "partition": partition,
                        "method": display_label,
                        "round": round_index,
                        "accuracy_mean": float(center),
                        "accuracy_std": float(spread),
                        "seed_count": int(seed_count),
                    }
                )
    partition_definitions = _partition_definitions(
        float(dataset_values["pathological_minor_fraction"])
    )
    caption = output_base.with_suffix(".caption.json")
    caption.write_text(
        json.dumps(
            {
                "metric": "test accuracy",
                "partition": partition,
                "partition_semantics": _partition_caption_semantics(
                    partition,
                    float(dataset_values["pathological_minor_fraction"]),
                    task_type=str(dataset_values["task_type"]),
                ),
                "partition_definitions": partition_definitions,
                "dataset": {
                    "name": dataset_values["name"],
                    "display_name": dataset_values["display_name"],
                    "task_type": dataset_values["task_type"],
                    "binary_task": dataset_values["binary_task"],
                    "multiclass_task": dataset_values["multiclass_task"],
                },
                "seed_count": int(seed_count),
                "spread": "standard deviation across seeds",
                "summary_statistic": (
                    "mean ± population standard deviation (ddof=0) across "
                    f"exactly {seed_count} seeds"
                ),
                "rounds": {
                    "first": 0,
                    "last": int(next(iter(lengths)) - 1),
                    "count": int(next(iter(lengths))),
                },
                "curve_processing": "raw unsmoothed per-round curves",
                "uncertainty_band_alpha": 0.10,
                "visual_contract": {
                    "background": "white",
                    "palette": (
                        "fixed seven-method palette adapted only from the "
                        "established 2023 Nature Communications curve style"
                    ),
                    "uncertainty": (
                        "translucent raw mean plus/minus population-standard-"
                        "deviation band; no smoothing or curve alteration"
                    ),
                    "editable_text": "PDF TrueType and SVG text retained",
                },
                "x_axis": "communication round",
                "decentralized_round_semantics": "one communication round",
                "dp_sgd_round_semantics": "centralized optimization step",
                "dp_sgd_x_axis_caveat": (
                    "DP-SGD x-axis values are centralized optimization steps, not "
                    "communication rounds."
                ),
                "chance_accuracy": chance_accuracy,
                "comparison_caveat": (
                    "Communication-round indices do not equalize function-query budgets; use the "
                    "reported query counts for resource comparisons."
                ),
                "privacy_caveat": (
                    "Private methods retain their method-specific adjacency and trust "
                    "models; epsilon values are not interchangeable without those scopes."
                ),
                "legend_method_semantics": {
                    "LT-ADMM-VR": "symmetric_two_point",
                    "LT-ADMM": "symmetric_two_point",
                    "DP-SGD": "symmetric_two_point",
                    "DP-FedAvg": "symmetric_two_point",
                    "1P-DSG": "one_point",
                    "1P-DSGT": "one_point",
                    "SPADE-DFL": "one_point_persistent_saga",
                },
                "visible_legend_entries": visible_legend_entries,
                "visible_legend_count": len(visible_legend_entries),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return png, pdf, svg, caption


def _normalize_dataset_figure_metadata(
    metadata: Optional[Mapping[str, object]],
    *,
    dataset_name: Optional[str] = None,
    negative_class: Optional[int] = None,
    negative_class_name: Optional[str] = None,
    positive_class: Optional[int] = None,
    positive_class_name: Optional[str] = None,
) -> dict[str, object]:
    values = dict(metadata or {})
    explicit = {
        "dataset_name": dataset_name,
        "negative_class": negative_class,
        "negative_class_name": negative_class_name,
        "positive_class": positive_class,
        "positive_class_name": positive_class_name,
    }
    values.update({key: value for key, value in explicit.items() if value is not None})
    name = str(values.get("dataset_name", "mnist"))
    display_names = {
        "mnist": "MNIST",
        "fashion_mnist": "Fashion-MNIST",
        "emnist_balanced": "EMNIST Balanced",
        "cifar10": "CIFAR-10",
    }
    task_type = str(values.get("task_type", "binary"))
    n_classes = int(values.get("n_classes", 2))
    binary_task = None
    multiclass_task = None
    if task_type == "multiclass":
        multiclass_task = {
            "n_classes": n_classes,
            "class_names": list(values.get("class_names", [])),
        }
    else:
        binary_task = {
            "negative_class": int(values.get("negative_class", 6)),
            "negative_class_name": str(values.get("negative_class_name", "digit 6")),
            "positive_class": int(values.get("positive_class", 7)),
            "positive_class_name": str(values.get("positive_class_name", "digit 7")),
        }
    return {
        "name": name,
        "display_name": display_names.get(name, name.replace("_", " ").title()),
        "task_type": task_type,
        "n_classes": n_classes,
        "binary_task": binary_task,
        "multiclass_task": multiclass_task,
        "pathological_minor_fraction": float(
            values.get("pathological_minor_fraction", 0.1)
        ),
    }


def _partition_definitions(pathological_minor_fraction: float) -> dict[str, str]:
    return {
        "iid": "random IID allocation across clients",
        "dirichlet_balanced": (
            "Dirichlet-controlled label proportions with equal client sample counts"
        ),
        "pathological_mixed": (
            "both classes appear at every client; alternating clients have opposite "
            "majority classes and minority fraction "
            f"{pathological_minor_fraction:g}"
        ),
        "pathological_shards": (
            "each client receives two label-sorted shards assigned without "
            "sample overlap"
        ),
    }


def _partition_caption_semantics(
    partition: str,
    pathological_minor_fraction: float,
    task_type: str = "binary",
) -> str:
    if partition == "dir_0.3":
        return "balanced Dirichlet allocation with alpha=0.3 and equal client sample counts"
    if partition == "dir_0.6":
        return "balanced Dirichlet allocation with alpha=0.6 and equal client sample counts"
    if partition == "pathological":
        return _partition_definitions(pathological_minor_fraction)[
            "pathological_shards"
            if task_type == "multiclass"
            else "pathological_mixed"
        ]
    if partition == "iid":
        return _partition_definitions(pathological_minor_fraction)["iid"]
    return partition


def plot_resource_metric(
    rows: Sequence[Mapping[str, object]],
    metric: str,
    output_base: Path,
) -> tuple[Path, Path]:
    if not rows:
        raise ValueError("resource plot requires at least one row")
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    methods = [str(row["method"]) for row in rows]
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    colors = [METHOD_STYLES.get(method, {"color": "#9e9e9e"})["color"] for method in methods]
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.bar(np.arange(len(methods)), values, color=colors, alpha=0.85)
    axis.set_xticks(np.arange(len(methods)), methods, rotation=30, ha="right", fontsize=8)
    axis.set_ylabel(metric.replace("_mean", "").replace("_", " ").title())
    axis.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    if np.any(values > 0) and values.max() / max(values[values > 0].min(), 1e-30) > 100:
        axis.set_yscale("symlog", linthresh=1.0)
    fig.tight_layout()
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def privacy_utility_data(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], Optional[float]]:
    private_rows = [
        row for row in rows if row.get("epsilon") is not None and row.get("epsilon") != ""
    ]
    reference = next(
        (
            float(row["final_accuracy"])
            for row in rows
            if row["method"] == "Non-private SPADE-DFL"
        ),
        None,
    )
    return private_rows, reference


def plot_privacy_utility(
    rows: Sequence[Mapping[str, object]], output_base: Path
) -> tuple[Path, Path]:
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    private_rows, reference = privacy_utility_data(rows)
    for row in private_rows:
        method = str(row["method"])
        epsilon = row.get("epsilon")
        style = METHOD_STYLES.get(method, {"color": "#9e9e9e"})
        axis.scatter(
            float(epsilon),
            100.0 * float(row["final_accuracy"]),
            s=70,
            color=style["color"],
            label=method,
        )
    if reference is not None:
        axis.axhline(
            100.0 * reference,
            color=METHOD_STYLES["Non-private SPADE-DFL"]["color"],
            linestyle="--",
            linewidth=1.8,
            label="Non-private SPADE-DFL reference",
        )
    axis.set_xlabel("Privacy Budget epsilon")
    axis.set_ylabel("Final Test Accuracy (%)")
    axis.grid(True, alpha=0.22, linewidth=0.6)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _partition_spec(panel: str, task_type: str = "binary") -> tuple[str, float]:
    if panel == "dir_0.1":
        return "dirichlet_balanced", 0.1
    if panel == "dir_0.3":
        return "dirichlet_balanced", 0.3
    if panel == "dir_0.6":
        return "dirichlet_balanced", 0.6
    if panel == "iid":
        return "iid", 0.3
    if panel == "pathological":
        return (
            "pathological_shards" if task_type == "multiclass" else "pathological_mixed",
            0.3,
        )
    if panel == "pathological_pure":
        return "pathological", 0.3
    raise ValueError(f"unknown panel {panel}")


def _resolve_path(path_text: str, parent: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else parent / path


def load_dataset_for_config(config: ExperimentConfig) -> DatasetBundle:
    if config.task_type == "multiclass":
        data_path = Path(config.data_path)
        is_cifar10 = config.dataset_name == "cifar10"
        if is_cifar10:
            from .cifar10 import CIFAR10_CLASS_NAMES

            class_names = CIFAR10_CLASS_NAMES
        else:
            class_names = None
        return load_multiclass_image_dataset(
            data_path,
            feature_dimension=config.feature_dimension,
            n_classes=config.n_classes,
            dataset_name=config.dataset_name,
            download_url=config.download_url or None,
            class_names=class_names,
            preprocessing_cache=(
                data_path.with_name(
                    f"{data_path.stem}-pca-d{config.feature_dimension}.npz"
                )
                if is_cifar10
                else None
            ),
        )
    return load_binary_image_dataset(
        Path(config.data_path),
        dimension=config.dimension,
        negative_class=config.negative_class,
        positive_class=config.positive_class,
        download_url=config.download_url or None,
        dataset_name=config.dataset_name,
        negative_class_name=config.negative_class_name,
        positive_class_name=config.positive_class_name,
    )


def load_dataset_for_suite(
    config: ExperimentConfig, suite: Mapping[str, object]
) -> DatasetBundle:
    """Load either the normal evaluation data or an explicit train-only holdout."""
    evaluation_mode = str(suite.get("evaluation_mode", "official_test"))
    if evaluation_mode == "official_test":
        return load_dataset_for_config(config)
    if evaluation_mode == "mnist_multiclass_training_holdout":
        if config.dataset_name != "mnist" or config.task_type != "multiclass":
            raise ValueError(
                "mnist_multiclass_training_holdout requires ten-class MNIST"
            )
        from .mnist_holdout import load_mnist_multiclass_training_holdout

        return load_mnist_multiclass_training_holdout(
            Path(config.data_path),
            validation_fraction=float(suite["validation_fraction"]),
            validation_seed=int(suite["validation_seed"]),
            feature_dimension=config.feature_dimension,
        )
    if evaluation_mode != "cifar10_training_holdout":
        raise ValueError(f"unsupported suite evaluation_mode {evaluation_mode}")
    if config.dataset_name != "cifar10" or config.task_type != "multiclass":
        raise ValueError("cifar10_training_holdout requires the CIFAR-10 multiclass base")
    from .cifar_holdout import load_cifar10_training_holdout

    return load_cifar10_training_holdout(
        Path(config.data_path),
        validation_fraction=float(suite["validation_fraction"]),
        validation_seed=int(suite["validation_seed"]),
    )


def _official_test_arrays_loaded(evaluation_mode: object) -> bool:
    """Return whether the suite evaluation mode reads official test arrays."""
    return str(evaluation_mode or "official_test") == "official_test"


def apply_suite_base_overrides(
    base: ExperimentConfig,
    suite: Mapping[str, object],
) -> ExperimentConfig:
    """Apply the one-round package diagnostic configuration without changing model parameters."""
    if suite.get("protocol") == MNIST67_PAPER_PROTOCOL:
        if "n_rounds_override" not in suite:
            return base
        n_rounds = int(suite["n_rounds_override"])
        if (
            n_rounds != 3
            or suite.get("inference_status")
            != "layout_and_runtime_preview_only"
        ):
            raise ValueError(
                "MNIST 6/7 paper preview permits only the registered three-round override"
            )
        sources = dict(base.parameter_sources)
        sources["n_rounds"] = ParameterRecord(
            name="n_rounds",
            value=3,
            source_type="Assumed",
            source=(
                "Three-round runtime and layout preview registered in the "
                "2026-08-27 design; not formal evidence"
            ),
        )
        return replace(base, n_rounds=3, parameter_sources=sources)
    if "n_rounds" not in suite:
        return base
    n_rounds = int(suite["n_rounds"])
    purpose = str(suite.get("purpose", ""))
    if n_rounds != 1 or "smoke" not in purpose.lower():
        raise ValueError("suite n_rounds override is restricted to the one-round package diagnostic configuration")
    sources = dict(base.parameter_sources)
    sources["n_rounds"] = ParameterRecord(
        name="n_rounds",
        value=1,
        source_type="Assumed",
        source="One-round package diagnostic configuration; not a scientific evaluation horizon",
    )
    return replace(base, n_rounds=1, parameter_sources=sources)


def audit_history_aggregates(
    suite_root: Path, *, tolerance: float = 1e-12
) -> dict[str, object]:
    """Verify every saved mean/std aggregate against its raw history CSV files."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    root = Path(suite_root)
    aggregate_paths = sorted((root / "aggregates").glob("*/*.npz"))
    history_paths = sorted((root / "runs").glob("*/*/**/history.csv"))
    if not aggregate_paths:
        raise ValueError("no aggregate NPZ files found")
    if not history_paths:
        raise ValueError("no history CSV files found")

    aggregate_keys = {
        (path.parent.name, path.stem) for path in aggregate_paths
    }
    history_keys = {
        (path.relative_to(root / "runs").parts[0], path.relative_to(root / "runs").parts[1])
        for path in history_paths
    }
    if aggregate_keys != history_keys:
        raise AssertionError(
            "history/aggregate method sets differ: "
            f"aggregates={sorted(aggregate_keys)}, histories={sorted(history_keys)}"
        )

    checks = []
    maximum_discrepancy = 0.0
    for aggregate_path in aggregate_paths:
        partition = aggregate_path.parent.name
        method = aggregate_path.stem
        method_histories = sorted(
            (root / "runs" / partition / method).glob("**/history.csv")
        )
        loaded_histories = [_read_history_metrics(path) for path in method_histories]
        with np.load(aggregate_path) as saved:
            mean_keys = sorted(key for key in saved.files if key.endswith("_mean"))
            for mean_key in mean_keys:
                metric = mean_key[: -len("_mean")]
                std_key = f"{metric}_std"
                if std_key not in saved.files:
                    raise AssertionError(
                        f"missing {std_key} in aggregate {aggregate_path}"
                    )
                try:
                    stacked = np.vstack(
                        [history[metric] for history in loaded_histories]
                    )
                except KeyError as exc:
                    raise AssertionError(
                        f"metric {metric} missing from a history for {partition}/{method}"
                    ) from exc
                expected = {
                    "mean": stacked.mean(axis=0),
                    "std": stacked.std(axis=0, ddof=0),
                }
                for statistic, aggregate_key in (("mean", mean_key), ("std", std_key)):
                    actual = np.asarray(saved[aggregate_key], dtype=np.float64)
                    recomputed = expected[statistic]
                    if actual.shape != recomputed.shape:
                        discrepancy = float("inf")
                    else:
                        difference = np.abs(actual - recomputed)
                        discrepancy = (
                            float(np.nanmax(difference)) if difference.size else 0.0
                        )
                    maximum_discrepancy = max(maximum_discrepancy, discrepancy)
                    checks.append(
                        {
                            "partition": partition,
                            "method": method,
                            "metric": metric,
                            "statistic": statistic,
                            "maximum_absolute_discrepancy": discrepancy,
                            "passed": discrepancy < tolerance,
                        }
                    )

    passed = all(bool(check["passed"]) for check in checks)
    report = {
        "passed": passed,
        "tolerance": float(tolerance),
        "aggregate_file_count": len(aggregate_paths),
        "history_file_count": len(history_paths),
        "metric_count": sum(
            1 for check in checks if check["statistic"] == "mean"
        ),
        "maximum_absolute_discrepancy": maximum_discrepancy,
        "checks": checks,
    }
    if not passed:
        raise AssertionError(
            "aggregate audit failed: maximum absolute discrepancy "
            f"{maximum_discrepancy:.17g} is not below tolerance {tolerance:.17g}"
        )
    return report


def _read_history_metrics(path: Path) -> dict[str, np.ndarray]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows or "iteration" not in fieldnames:
        raise AssertionError(f"history is empty or malformed: {path}")
    metrics = {}
    for name in fieldnames:
        if name == "iteration":
            continue
        values = np.asarray(
            [float(row[name]) if row[name] != "" else np.nan for row in rows],
            dtype=np.float64,
        )
        while values.size and np.isnan(values[-1]):
            values = values[:-1]
        metrics[name] = values
    return metrics


def audit_dataset_manifest(
    dataset: DatasetBundle,
    preprocessing_path: Path,
    recorded: Mapping[str, object],
) -> dict[str, bool]:
    """Cross-check recorded dataset counts, dimensions, and PCA artifact shapes."""
    counts = recorded.get("sample_counts", {})
    dimensions = recorded.get("dimensions", {})
    preprocessing = recorded.get("preprocessing", {})
    expected_shapes = {
        "mean": list(np.asarray(dataset.pca.mean).shape),
        "components": list(np.asarray(dataset.pca.components).shape),
        "scale": list(np.asarray(dataset.pca.scale).shape),
    }
    counts_match = counts == {
        "original_train": dataset.original_train_count,
        "original_test": dataset.original_test_count,
        "filtered_train": dataset.filtered_train_count,
        "filtered_test": dataset.filtered_test_count,
    } and (
        dataset.train_x.shape[0] == dataset.filtered_train_count
        and dataset.test_x.shape[0] == dataset.filtered_test_count
    )
    dimensions_match = dimensions == {
        "original_feature": dataset.original_feature_dimension,
        "pca": dataset.pca_dimension,
    } and (
        dataset.train_x.shape[1] == dataset.pca_dimension
        and dataset.pca.components.shape
        == (dataset.pca_dimension, dataset.original_feature_dimension)
    )
    preprocessing_path = Path(preprocessing_path)
    artifact_exists = preprocessing_path.is_file()
    actual_shapes: dict[str, list[int]] = {}
    if artifact_exists:
        with np.load(preprocessing_path, allow_pickle=False) as archive:
            actual_shapes = {
                name: list(np.asarray(archive[name]).shape)
                for name in ("mean", "components", "scale")
                if name in archive
            }
    preprocessing_shapes_match = (
        isinstance(preprocessing, Mapping)
        and preprocessing.get("array_shapes") == expected_shapes
        and actual_shapes == expected_shapes
    )
    report = {
        "passed": bool(
            counts_match
            and dimensions_match
            and artifact_exists
            and preprocessing_shapes_match
        ),
        "counts_match_arrays": bool(counts_match),
        "dimensions_match_arrays": bool(dimensions_match),
        "preprocessing_artifact_exists": bool(artifact_exists),
        "preprocessing_shapes_match": bool(preprocessing_shapes_match),
    }
    return report


def run_paper_suite(suite_config_path: Path, output_root: Path) -> Path:
    suite_config_path = Path(suite_config_path)
    raw = json.loads(suite_config_path.read_text(encoding="utf-8"))
    methods = list(raw["methods"])
    protocol = raw.get("protocol")
    panels = list(validate_panel_sequence(raw["partitions"], protocol=protocol))
    if any(method not in METHOD_REGISTRY for method in methods):
        raise ValueError("paper suite contains an unsupported method")
    method_parameters = resolve_method_parameters(
        raw,
        suite_config_path,
        expected_methods=methods,
    )
    validate_d20_suite_contract(protocol, methods, method_parameters)
    unknown_parameter_methods = set(method_parameters) - set(methods)
    if unknown_parameter_methods:
        raise ValueError(
            "method_parameters contains unselected methods: "
            + ", ".join(sorted(unknown_parameter_methods))
        )
    comparison_environment = raw.get("comparison_environment", "mixed_access")
    if comparison_environment == "all_black_box":
        validate_black_box_method_set(methods)
    base_path = _resolve_path(raw["base_config"], suite_config_path.parent)
    base = apply_suite_base_overrides(load_config(base_path), raw)
    seeds = tuple(int(seed) for seed in raw["seeds"])
    target_accuracy = float(raw.get("target_accuracy", 0.9))
    dataset = load_dataset_for_suite(base, raw)
    figure_dataset_metadata = {
        "dataset_name": dataset.dataset_name,
        "negative_class": dataset.negative_class,
        "negative_class_name": dataset.negative_class_name,
        "positive_class": dataset.positive_class,
        "positive_class_name": dataset.positive_class_name,
        "pathological_minor_fraction": base.pathological_minor_fraction,
    }
    if dataset.task_type == "multiclass":
        figure_dataset_metadata.update(
            task_type="multiclass",
            n_classes=dataset.n_classes,
            class_names=list(dataset.class_names),
        )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    preprocessing_path = save_preprocessing(dataset.pca, root / "preprocessing.npz")
    nested: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    final_rows = []
    target_rows = []
    manifest_runs = []
    source_rows = []
    accuracy_contract_rows = []

    frozen_parameter_artifact = None
    if raw.get("method_parameters_from") is not None:
        frozen_path = _resolve_path(
            str(raw["method_parameters_from"]), suite_config_path.parent
        ).resolve()
        frozen_raw = json.loads(frozen_path.read_text(encoding="utf-8"))
        frozen_parameter_artifact = {
            "path": str(frozen_path),
            "sha256": _sha256_file(frozen_path),
            "status": frozen_raw.get("status"),
            "official_test_used_for_selection": frozen_raw.get(
                "official_test_used_for_selection"
            ),
            "selection_dataset": frozen_raw.get("selection_dataset"),
        }

    for panel in panels:
        panel_index = partition_seed_index(panel)
        mode, alpha = _partition_spec(panel, task_type=base.task_type)
        nested[panel] = {}
        for method in methods:
            method_results = []
            label = METHOD_REGISTRY[method].label
            for seed in seeds:
                config = configure_suite_method(
                    base,
                    method,
                    seeds,
                    mode,
                    alpha,
                    protocol=protocol,
                    override=method_parameters.get(method),
                )
                partition_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, panel_index, 20260710])
                )
                partition = partition_labels(
                    dataset.train_y,
                    suite_partition_client_count(
                        method, config, base, protocol=protocol
                    ),
                    mode,
                    partition_rng,
                    alpha=alpha,
                    pathological_minor_fraction=base.pathological_minor_fraction,
                )
                expected_run_dir = (
                    root
                    / "runs"
                    / panel
                    / method
                    / config.stage
                    / f"seed_{seed}"
                )
                try:
                    result = load_run(
                        expected_run_dir,
                        expected_config=config,
                        expected_dataset_sha256=dataset.sha256,
                        expected_partition=panel,
                        expected_label_histograms=partition.label_histograms,
                        expected_seed=seed,
                        expected_stage=config.stage,
                        expected_history_length=config.n_rounds + 1,
                        expected_state_shape=(config.n_clients, config.dimension),
                    )
                except (FileNotFoundError, ValueError, KeyError):
                    result = run_method(method, config, dataset, partition, seed)
                run_dir = save_run(
                    result,
                    config,
                    root / "runs" / panel / method,
                    {
                        "sha256": dataset.sha256,
                        "partition": panel,
                        "label_histograms": partition.label_histograms.tolist(),
                    },
                )
                manifest_runs.append(
                    {
                        "partition": panel,
                        "method": method,
                        "seed": seed,
                        "run_dir": manifest_run_dir(root, run_dir),
                    }
                )
                method_results.append(result)
                accuracy_contract_rows.append(
                    {
                        "partition": panel,
                        "method": method,
                        "seed": seed,
                        "initial_accuracy": float(result.history["accuracy"][0]),
                        "final_accuracy": float(result.history["accuracy"][-1]),
                        "initial_loss": float(result.history["loss"][0]),
                        "final_loss": float(result.history["loss"][-1]),
                    }
                )
                if panel == panels[0] and seed == seeds[0]:
                    for parameter_name in sorted(config.parameter_sources):
                        record = config.parameter_sources[parameter_name]
                        source_rows.append(
                            {
                                "method": label,
                                "parameter": record.name,
                                "value": json.dumps(record.value, ensure_ascii=False)
                                if isinstance(record.value, (list, tuple, dict))
                                else record.value,
                                "source_type": record.source_type,
                                "source": record.source,
                            }
                        )
            aggregate = aggregate_results(method_results)
            aggregate_path = root / "aggregates" / panel / f"{method}.npz"
            save_aggregate(aggregate, aggregate_path)
            nested[panel][label] = aggregate
            mean_clip_fraction = 0.0
            if method == "spade_dfl_private":
                mean_clip_fraction = float(
                    np.mean(
                        [
                            np.mean(
                                np.asarray(
                                    item.metadata.get("clip_fraction", [1.0]),
                                    dtype=np.float64,
                                )
                            )
                            for item in method_results
                        ]
                    )
                )
            final_rows.append(
                {
                    "partition": panel,
                    "method": label,
                    "initial_accuracy": float(aggregate["accuracy_mean"][0]),
                    "final_accuracy": float(aggregate["accuracy_mean"][-1]),
                    "accuracy_std": float(aggregate["accuracy_std"][-1]),
                    "initial_loss": float(aggregate["loss_mean"][0]),
                    "final_loss": float(aggregate["loss_mean"][-1]),
                    "generalization_error": float(aggregate["generalization_error_mean"][-1]),
                    "gradient_norm": float(aggregate["gradient_norm_mean"][-1]),
                    "initial_stationarity": float(aggregate["stationarity_mean"][0]),
                    "stationarity": float(aggregate["stationarity_mean"][-1]),
                    "initial_consensus_error": float(
                        aggregate["consensus_error_mean"][0]
                    ),
                    "consensus_error": float(aggregate["consensus_error_mean"][-1]),
                    "initial_stationarity_consensus": float(
                        aggregate["stationarity_consensus_mean"][0]
                    ),
                    "stationarity_consensus": float(
                        aggregate["stationarity_consensus_mean"][-1]
                    ),
                    "paper_stationarity_consensus": paper_stationarity_consensus(
                        aggregate["stationarity_mean"],
                        aggregate["consensus_error_mean"],
                    ),
                    "mean_clip_fraction": mean_clip_fraction,
                    "communication_rounds_mean": float(
                        np.mean(
                            [
                                int(item.metadata.get("communication_rounds", 0))
                                for item in method_results
                            ]
                        )
                    ),
                    "query_count_mean": float(np.mean([item.query_count for item in method_results])),
                    "gradient_evaluations_mean": float(
                        np.mean(
                            [
                                int(item.metadata.get("gradient_evaluations", 0))
                                for item in method_results
                            ]
                        )
                    ),
                    "transmitted_bytes_mean": float(
                        np.mean([item.transmitted_bytes for item in method_results])
                    ),
                    "wall_time_mean": float(np.mean([item.wall_time for item in method_results])),
                    "epsilon": method_results[0].metadata.get("epsilon"),
                    "delta": method_results[0].metadata.get("delta", 0.0),
                    "privacy_scope": method_results[0].metadata.get(
                        "privacy_scope", "none"
                    ),
                    "privacy_valid": all(
                        bool(item.metadata.get("privacy_valid", True))
                        for item in method_results
                    ),
                    "configured_n_clients": config.n_clients,
                    "suite_base_n_clients": base.n_clients,
                    "n_clients_protocol_exception": config.n_clients
                    != base.n_clients,
                    "outer_round_semantics": method_results[0].metadata.get(
                        "outer_round_semantics", "unspecified"
                    ),
                    "x_axis_caveat": method_results[0].metadata.get(
                        "x_axis_caveat", "unspecified"
                    ),
                }
            )
            target_rows.append(
                {
                    "partition": panel,
                    "method": label,
                    "target_accuracy": target_accuracy,
                    "first_round": rounds_to_target(aggregate["accuracy_mean"], target_accuracy),
                }
            )

    resource_metrics = {
        "query_count_mean": "query_count",
        "gradient_evaluations_mean": "gradient_evaluations",
        "transmitted_bytes_mean": "transmitted_bytes",
        "communication_rounds_mean": "communication_rounds",
        "wall_time_mean": "wall_time",
    }
    for panel in panels:
        figure_root = root / "figures" / panel
        panel_rows = [row for row in final_rows if row["partition"] == panel]
        for metric in ("accuracy", "loss", "stationarity"):
            if metric == "accuracy" and (
                use_strict_black_box_accuracy_figure(
                    comparison_environment=comparison_environment,
                    n_rounds=base.n_rounds,
                )
                or (
                    comparison_environment == "all_black_box"
                    and base.dataset_name != "cifar10"
                )
            ):
                plot_black_box_accuracy(
                    nested[panel],
                    figure_root / metric,
                    seed_count=len(seeds),
                    partition=panel,
                    dataset_metadata=figure_dataset_metadata,
                )
            else:
                plot_single_partition_curves(
                    nested[panel], metric, panel, figure_root / metric
                )
        for metric, stem in resource_metrics.items():
            plot_resource_metric(panel_rows, metric, figure_root / stem)
        plot_privacy_utility(panel_rows, figure_root / "privacy_utility")
    write_rows(final_rows, root / "final_metrics.csv")
    write_rows(target_rows, root / "rounds_to_target.csv")
    write_rows(source_rows, root / "method_parameter_sources.csv")
    private_method_names = {
        "spade_dfl_private",
        "dp_sgd",
        "dp_fedavg",
        "zo2_dp_sgd",
        "zo2_dp_fedavg",
    }
    declared_privacy_budgets = {}
    first_mode, first_alpha = _partition_spec(
        panels[0], task_type=base.task_type
    )
    for method in methods:
        if method not in private_method_names:
            continue
        declared_config = configure_suite_method(
            base,
            method,
            seeds,
            first_mode,
            first_alpha,
            protocol=protocol,
            override=method_parameters.get(method),
        )
        declared_privacy_budgets[METHOD_REGISTRY[method].label] = (
            float(declared_config.target_epsilon),
            float(declared_config.target_delta),
        )
    assessment = assess_paper_suite(
        final_rows,
        target_rows,
        base.n_rounds,
        len(seeds),
        target_accuracy,
        expected_method_labels=(
            BLACK_BOX_METHOD_LABELS
            if comparison_environment == "all_black_box"
            else [METHOD_REGISTRY[method].label for method in methods]
        ),
        reference_baseline_labels=[
            METHOD_REGISTRY[method].label
            for method in methods
            if method
            in {
                "lt_admm_vr",
                "lt_admm",
                "zo2_lt_admm_vr",
                "zo2_lt_admm",
                "one_point_dsg",
                "one_point_dsgt",
            }
        ],
        private_method_labels=[
            METHOD_REGISTRY[method].label
            for method in methods
            if method in private_method_names
        ],
        declared_privacy_budgets=declared_privacy_budgets,
    )
    if comparison_environment == "all_black_box":
        assessment["black_box_competitiveness"] = (
            assess_black_box_competitiveness(final_rows, panels)
        )
    if (
        base.dataset_name == "emnist_balanced"
        and tuple(methods) == BLACK_BOX_METHOD_IDS
        and tuple(panels) == PANEL_ORDER
    ):
        attach_emnist_accuracy_contract(
            assessment,
            accuracy_contract_rows,
            n_classes=dataset.n_classes,
            seeds=seeds,
        )
    (root / "validation_summary.json").write_text(
        json.dumps(assessment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    normalized_dataset = _normalize_dataset_figure_metadata(
        figure_dataset_metadata
    )
    dataset_manifest = {
        "name": normalized_dataset["name"],
        "display_name": normalized_dataset["display_name"],
        "binary_task": normalized_dataset["binary_task"],
        "sample_counts": {
            "original_train": dataset.original_train_count,
            "original_test": dataset.original_test_count,
            "filtered_train": dataset.filtered_train_count,
            "filtered_test": dataset.filtered_test_count,
        },
        "dimensions": {
            "original_feature": dataset.original_feature_dimension,
            "pca": dataset.pca_dimension,
        },
        "preprocessing": {
            "artifact": str(preprocessing_path.relative_to(root)).replace("\\", "/"),
            "array_shapes": {
                "mean": list(np.asarray(dataset.pca.mean).shape),
                "components": list(np.asarray(dataset.pca.components).shape),
                "scale": list(np.asarray(dataset.pca.scale).shape),
            },
        },
    }
    dataset_manifest["metadata_audit"] = audit_dataset_manifest(
        dataset, preprocessing_path, dataset_manifest
    )
    if not dataset_manifest["metadata_audit"]["passed"]:
        raise RuntimeError("dataset manifest metadata audit failed")
    manifest = {
        "suite_config": raw,
        "base_config": str(base_path),
        "dataset_sha256": dataset.sha256,
        "dataset": dataset_manifest,
        "aggregation": {
            "seed_count": len(seeds),
            "center": "arithmetic mean",
            "spread": "population standard deviation (ddof=0)",
            "curve_processing": "raw unsmoothed per-round curves",
        },
        "partition_definitions": _partition_definitions(
            base.pathological_minor_fraction
        ),
        "visible_legend_entries": (
            list(BLACK_BOX_LEGEND_ORDER)
            if comparison_environment == "all_black_box"
            else []
        ),
        "runs": manifest_runs,
        "status": "completed",
        "verification_status": assessment["status"],
        "comparison_environment": comparison_environment,
        "evaluation_protocol": {
            "mode": str(raw.get("evaluation_mode", "official_test")),
            "official_test_arrays_loaded": _official_test_arrays_loaded(
                raw.get("evaluation_mode")
            ),
            "validation_fraction": raw.get("validation_fraction"),
            "validation_seed": raw.get("validation_seed"),
        },
        "protocol": protocol,
        "frozen_parameter_artifact": frozen_parameter_artifact,
        "method_protocol_declarations": method_protocol_declarations(
            base, methods, method_parameters, protocol
        ),
    }
    (root / "suite_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root
