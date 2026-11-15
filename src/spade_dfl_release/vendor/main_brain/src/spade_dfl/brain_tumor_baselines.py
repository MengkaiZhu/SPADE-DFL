from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .config import ExperimentConfig, ParameterRecord


def baseline_candidate_grid() -> dict[str, list[dict[str, object]]]:
    admm_rows = [
        {
            "eta0": float(eta0),
            "rho": float(rho),
            "batch_size": 1,
            "provenance": (
                "Reference [13] Table V step sizes {0.1,1,2,3,4,5}; "
                "rho training-holdout grid {0.25,0.5,1,2} because the "
                "paper gives only a positive ADMM penalty condition for "
                "the MRI transfer; batch=1 and tau=2 are Reference [13]"
            ),
        }
        for eta0 in (0.1, 1.0, 2.0, 3.0, 4.0, 5.0)
        for rho in (0.25, 0.5, 1.0, 2.0)
    ]
    return {
        "zo2_lt_admm_vr": [dict(row) for row in admm_rows],
        "zo2_lt_admm": [dict(row) for row in admm_rows],
        "zo2_dp_sgd": [
            {
                "eta0": eta0,
                "batch_size": 56,
                "clip_norm": 4.0,
                "provenance": (
                    "Reference [24] reports stable learning rates in "
                    "[0.01,0.07] with a peak at 0.05; batch=56 preserves "
                    "its q=0.01 lot fraction on 5600 MRI training images; "
                    "clip norm 4 is the Reference [24] value"
                ),
            }
            for eta0 in (0.01, 0.03, 0.05, 0.07)
        ],
        "zo2_dp_fedavg": [
            {
                "eta0": eta0,
                "batch_size": 10,
                "local_epochs": 4,
                "client_clip_mode": "round_median",
                "provenance": (
                    "Reference [47] official code defaults to batch=10 and "
                    "local epochs=4 and Algorithm 1 uses round-median client "
                    "clipping; eta is a four-point two-point-adaptation grid "
                    "centered on the official CNN learning rate 0.001"
                ),
            }
            for eta0 in (0.00025, 0.0005, 0.001, 0.002)
        ],
    }


def apply_frozen_baseline_parameters(
    config: ExperimentConfig,
    method: str,
    frozen: Mapping[str, Mapping[str, object]],
) -> ExperimentConfig:
    if method not in frozen:
        return config
    selected = frozen[method]
    supported = {
        "eta0",
        "rho",
        "batch_size",
        "local_epochs",
        "client_clip_mode",
        "clip_norm",
    }
    values = {name: selected[name] for name in supported if name in selected}
    sources = dict(config.parameter_sources)
    for name, value in values.items():
        if method == "zo2_dp_fedavg" and name in {
            "batch_size",
            "local_epochs",
            "client_clip_mode",
        }:
            source_type = "Reference"
            source = (
                "Reference [47] official code defaults and Algorithm 1 "
                "round-median clipping"
            )
        elif method == "zo2_dp_sgd" and name in {"batch_size", "clip_norm"}:
            source_type = "TheoryDerived" if name == "batch_size" else "Reference"
            source = (
                "Reference [24] q=0.01 scaled to 5600 training images"
                if name == "batch_size"
                else "Reference [24] clipping threshold 4"
            )
        else:
            source_type = "Validation"
            source = (
                "Frozen training-only inner-PCA holdout selection from a "
                "reference-anchored grid; official test excluded"
            )
        sources[name] = ParameterRecord(name, value, source_type, source)
    return replace(config, parameter_sources=sources, **values)

