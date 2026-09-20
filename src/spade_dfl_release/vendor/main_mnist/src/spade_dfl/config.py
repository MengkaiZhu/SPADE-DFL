from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


SOURCE_TYPES = {
    "SPADE-DFL",
    "Reference",
    "Assumed",
    "Validation",
    "TheoryDerived",
}
STAGES = {
    "single_zo",
    "single_vr",
    "multi_zo_admm",
    "multi_vr_admm",
    "spade_dfl",
    "spade_dfl_private",
    "exact_gradient",
    "one_point_dsgt",
    "one_point_dsg",
    "lt_admm",
    "lt_admm_vr",
    "dp_sgd",
    "dp_fedavg",
    "zo2_lt_admm",
    "zo2_lt_admm_vr",
    "zo2_dp_sgd",
    "zo2_dp_fedavg",
}
OBJECTIVE_MODELS = {"paper_sigmoid", "lt_admm_nonconvex_logistic"}


@dataclass(frozen=True)
class ParameterRecord:
    name: str
    value: Any
    source_type: str
    source: str

    def __post_init__(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(
                f"source_type must be one of {sorted(SOURCE_TYPES)}, got {self.source_type!r}"
            )


@dataclass(frozen=True)
class ExperimentConfig:
    stage: str = "single_zo"
    objective_model: str = "paper_sigmoid"
    dimension: int = 10
    n_rounds: int = 1000
    tau: int = 1
    batch_size: int = 31
    n_clients: int = 1
    seeds: tuple = (0, 1, 2)
    eta0: float = 0.15
    eta_exponent: float = 0.51
    mu0: float = 3.5
    mu_exponent: float = 0.17
    exact_eta0: float = 2.5
    regularization: float = 0.1
    sigma_u: float = 0.01
    query_noise_std: float = 1.0
    rho: float = 1.0
    beta: float = 0.0
    clip_norm: float = 1.0
    clip_exponent: float = 0.0
    dp_noise_std: float = 0.0
    target_epsilon: float = 0.0
    target_delta: float = 0.0
    local_epochs: int = 1
    shared_initialization: bool = False
    initialization: str = "independent_uniform"
    privacy_allocation: str = "theorem_weighted"
    graph_topology: str = "ring"
    graph_probability: float = 0.3
    partition_mode: str = "iid"
    dirichlet_alpha: float = 0.3
    pathological_minor_fraction: float = 0.1
    vr_mode: str = "persistent_saga"
    data_path: str = "data/mnist.npz"
    output_dir: str = "results"
    download_url: str = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    parameter_sources: Mapping[str, ParameterRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {sorted(STAGES)}, got {self.stage!r}")
        if self.objective_model not in OBJECTIVE_MODELS:
            raise ValueError(
                "objective_model must be one of "
                f"{sorted(OBJECTIVE_MODELS)}, got {self.objective_model!r}"
            )
        if self.objective_model == "lt_admm_nonconvex_logistic" and self.stage not in {
            "one_point_dsg",
            "one_point_dsgt",
        } and (
            self.sigma_u != 0.0 or self.query_noise_std != 0.0
        ):
            raise ValueError(
                "lt_admm_nonconvex_logistic requires sigma_u=0 and "
                "query_noise_std=0 as in Reference [13]"
            )
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.n_rounds <= 0 or self.tau <= 0 or self.batch_size <= 0:
            raise ValueError("n_rounds, tau, and batch_size must be positive")
        if self.n_clients <= 0:
            raise ValueError("n_clients must be positive")
        if self.vr_mode not in {"full_refresh", "persistent_saga"}:
            raise ValueError("vr_mode must be full_refresh or persistent_saga")
        if self.eta0 <= 0 or self.mu0 <= 0:
            raise ValueError("eta0 and mu0 must be positive")
        if self.rho <= 0:
            raise ValueError("rho must be positive")
        if self.clip_exponent < 0:
            raise ValueError("clip_exponent cannot be negative")
        if self.dp_noise_std < 0 or self.target_epsilon < 0:
            raise ValueError("privacy noise and target epsilon cannot be negative")
        if self.dp_noise_std > 0 and self.target_epsilon > 0:
            raise ValueError("set either dp_noise_std or target_epsilon, not both")
        if self.target_delta < 0 or self.target_delta >= 1:
            raise ValueError("target_delta must be zero or lie in (0, 1)")
        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be positive")
        if self.initialization not in {
            "independent_uniform",
            "shared_uniform",
            "shared_zero",
        }:
            raise ValueError(
                "initialization must be independent_uniform, shared_uniform, "
                "or shared_zero"
            )
        if self.shared_initialization and self.initialization != "independent_uniform":
            raise ValueError(
                "shared_initialization legacy flag cannot be combined with initialization"
            )
        if self.privacy_allocation not in {
            "theorem_weighted",
            "uniform_zcdp",
        }:
            raise ValueError(
                "privacy_allocation must be theorem_weighted or uniform_zcdp"
            )
        if self.graph_topology not in {"ring", "erdos_renyi"}:
            raise ValueError("graph_topology must be ring or erdos_renyi")
        if not 0.0 < self.graph_probability <= 1.0:
            raise ValueError("graph_probability must lie in (0, 1]")
        if self.partition_mode not in {
            "iid",
            "dirichlet",
            "dirichlet_balanced",
            "pathological",
            "pathological_mixed",
        }:
            raise ValueError(
                "partition_mode must be iid, dirichlet, dirichlet_balanced, "
                "pathological, or pathological_mixed"
            )
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        if not 0.0 < self.pathological_minor_fraction < 0.5:
            raise ValueError("pathological_minor_fraction must lie in (0, 0.5)")

    @property
    def cz(self) -> float:
        return 1.0 / self.dimension

    @classmethod
    def binary_defaults(
        cls,
        dimension: int = 10,
        stage: str = "single_zo",
        **overrides: Any,
    ) -> "ExperimentConfig":
        eta0 = 1.5 / dimension
        base: Dict[str, Any] = {
            "stage": stage,
            "dimension": dimension,
            "eta0": eta0,
            "batch_size": 31,
        }
        base.update(overrides)
        records = _binary_parameter_sources(base)
        base["parameter_sources"] = records
        return cls(**base)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        result["parameter_sources"] = {
            name: asdict(record) if isinstance(record, ParameterRecord) else dict(record)
            for name, record in self.parameter_sources.items()
        }
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        values = dict(raw)
        values["seeds"] = tuple(values.get("seeds", (0, 1, 2)))
        source_raw = values.get("parameter_sources", {})
        if not source_raw:
            source_raw = {
                name: asdict(record)
                for name, record in _binary_parameter_sources(values).items()
            }
        values["parameter_sources"] = {
            name: value
            if isinstance(value, ParameterRecord)
            else ParameterRecord(**value)
            for name, value in source_raw.items()
        }
        return cls(**values)


def reference20_base_config(
    n_rounds: int = 50,
    seeds: Sequence[int] = (0, 1, 2),
) -> ExperimentConfig:
    """Return the Reference [20] MNIST 6/7 environment.

    Values absent from Reference [20] remain explicitly classified as
    validation-selected or theory-derived instead of being presented as paper
    hyperparameters.
    """
    values: Dict[str, Any] = {
        "stage": "multi_vr_admm",
        "objective_model": "paper_sigmoid",
        "dimension": 10,
        "n_rounds": int(n_rounds),
        "tau": 2,
        "batch_size": 1,
        "n_clients": 31,
        "seeds": tuple(int(seed) for seed in seeds),
        "eta0": 0.15,
        "eta_exponent": 0.51,
        "mu0": 3.5,
        "mu_exponent": 0.17,
        "regularization": 0.1,
        "sigma_u": 0.01,
        "query_noise_std": 1.0,
        "rho": 1.0,
        "clip_norm": 1.0,
        "clip_exponent": 0.0,
        "target_epsilon": 8.0,
        "target_delta": 1e-5,
        "local_epochs": 1,
        "initialization": "independent_uniform",
        "graph_topology": "erdos_renyi",
        "graph_probability": 0.3,
        "partition_mode": "iid",
        "dirichlet_alpha": 0.3,
        "vr_mode": "persistent_saga",
    }
    sources = _binary_parameter_sources(values)
    sources.update(
        {
            "batch_size": ParameterRecord(
                "batch_size",
                1,
                "Reference",
                "References [20] and [21] use one stochastic sample/query per local update",
            ),
            "rho": ParameterRecord(
                "rho",
                1.0,
                "Validation",
                "SPADE-DFL requires rho > 0 but gives no MNIST value; initial member of the preregistered holdout grid",
            ),
            "clip_norm": ParameterRecord(
                "clip_norm",
                1.0,
                "Validation",
                "Initial placeholder replaced by the preregistered training-holdout update-norm quantile",
            ),
            "clip_exponent": ParameterRecord(
                "clip_exponent",
                0.0,
                "Validation",
                "Initial member of the preregistered clipping-decay grid {0, 0.17, 0.51}",
            ),
            "target_epsilon": ParameterRecord(
                "target_epsilon",
                8.0,
                "Reference",
                "Reference [24] reports epsilon=8 for MNIST",
            ),
            "target_delta": ParameterRecord(
                "target_delta",
                1e-5,
                "Reference",
                "Reference [24] reports delta=1e-5 for MNIST",
            ),
            "partition_mode": ParameterRecord(
                "partition_mode",
                "iid",
                "Assumed",
                "User-required supplemental IID/non-IID comparison factor",
            ),
            "dirichlet_alpha": ParameterRecord(
                "dirichlet_alpha",
                0.3,
                "Assumed",
                "User-required supplemental Dirichlet alpha values 0.3 and 0.6",
            ),
            "initialization": ParameterRecord(
                "initialization",
                "independent_uniform",
                "Reference",
                "Reference [20], Sec. 5.1 initializes each client in U([-1,1]^d)",
            ),
        }
    )
    return ExperimentConfig(parameter_sources=sources, **values)


def _binary_parameter_sources(values: Mapping[str, Any]) -> Dict[str, ParameterRecord]:
    dimension = int(values["dimension"])
    rows = {
        "objective_model": ParameterRecord(
            "objective_model",
            values.get("objective_model", "paper_sigmoid"),
            "Reference",
            "Reference [13], Sec. IV nonconvex logistic objective"
            if values.get("objective_model") == "lt_admm_nonconvex_logistic"
            else "Reference [20], Eq. (21)",
        ),
        "dimension": ParameterRecord(
            "dimension", dimension, "Reference", "Reference [20], Sec. 5: lossy encoding to d=10"
        ),
        "eta0": ParameterRecord(
            "eta0",
            values.get("eta0", 1.5 / dimension),
            "Reference"
            if (
                values.get("objective_model") == "lt_admm_nonconvex_logistic"
                and math.isclose(float(values.get("eta0", 0.0)), 0.1)
            )
            or (
                values.get("objective_model", "paper_sigmoid") == "paper_sigmoid"
                and math.isclose(
                    float(values.get("eta0", 1.5 / dimension)),
                    1.5 / dimension,
                )
            )
            else "Validation",
            "Reference [13], Table V includes gamma=0.1"
            if values.get("objective_model") == "lt_admm_nonconvex_logistic"
            and math.isclose(float(values.get("eta0", 0.0)), 0.1)
            else "Reference [20], Sec. 5.1 eta0=1.5, converted by cz=1/d for SPADE Eq. (10)"
            if values.get("objective_model", "paper_sigmoid") == "paper_sigmoid"
            and math.isclose(
                float(values.get("eta0", 1.5 / dimension)), 1.5 / dimension
            )
            else "Training-holdout-selected initial step scale",
        ),
        "eta_exponent": ParameterRecord(
            "eta_exponent", values.get("eta_exponent", 0.51), "Reference", "Reference [20], Sec. 5.1"
        ),
        "mu0": ParameterRecord("mu0", values.get("mu0", 3.5), "Reference", "Reference [20], Sec. 5.1"),
        "mu_exponent": ParameterRecord(
            "mu_exponent", values.get("mu_exponent", 0.17), "Reference", "Reference [20], Sec. 5.1"
        ),
        "regularization": ParameterRecord(
            "regularization",
            values.get("regularization", 0.1),
            "Reference",
            "Reference [13], Sec. IV sets epsilon=0.01"
            if values.get("objective_model") == "lt_admm_nonconvex_logistic"
            else "Reference [20], Eq. (21), Sec. 5.1",
        ),
        "sigma_u": ParameterRecord(
            "sigma_u",
            values.get("sigma_u", 0.01),
            "Reference",
            "Reference [13], Sec. IV deterministic objective (no multiplicative query noise)"
            if values.get("objective_model") == "lt_admm_nonconvex_logistic"
            else "Reference [20], Sec. 5.1",
        ),
        "query_noise_std": ParameterRecord(
            "query_noise_std",
            values.get("query_noise_std", 1.0),
            "Reference",
            "Reference [13], Sec. IV deterministic objective (no additive query noise)"
            if values.get("objective_model") == "lt_admm_nonconvex_logistic"
            else "Reference [20], Sec. 5.1",
        ),
        "n_rounds": ParameterRecord(
            "n_rounds",
            values.get("n_rounds", 1000),
            "Reference" if int(values.get("n_rounds", 1000)) == 1000 else "Assumed",
            "Reference [20], Fig. 1 iteration axis"
            if int(values.get("n_rounds", 1000)) == 1000
            else "Fixed total privacy-budget horizon selected by MNIST stability ablation",
        ),
        "batch_size": ParameterRecord(
            "batch_size",
            values.get("batch_size", 31),
            "Reference"
            if int(values.get("batch_size", 31)) == 1
            else "Validation",
            "References [20] and [21] use one stochastic sample/query per update"
            if int(values.get("batch_size", 31)) == 1
            else "Training-holdout-selected SPADE Eq. (15) mini-batch; constrained below each client's sample count",
        ),
        "seeds": ParameterRecord(
            "seeds",
            list(values.get("seeds", [0, 1, 2])),
            "Reference" if len(values.get("seeds", [0, 1, 2])) == 30 else "Assumed",
            "Reference [20], Sec. 5.1 uses 30 simulation instances" if len(values.get("seeds", [0, 1, 2])) == 30 else "Reduced local smoke budget",
        ),
        "pca": ParameterRecord(
            "pca", True, "Assumed", "Autoencoder architecture is absent; train-only PCA replacement"
        ),
        "rho": ParameterRecord("rho", values.get("rho", 1.0), "Validation", "SPADE requires rho > 0 but provides no MNIST numerical value"),
        "n_clients": ParameterRecord(
            "n_clients",
            values.get("n_clients", 1),
            "Reference"
            if int(values.get("n_clients", 1)) in {10, 31}
            else "Assumed",
            "LT-ADMM-VR Sec. IV uses a ring with N=10"
            if int(values.get("n_clients", 1)) == 10
            else "Reference [20], Sec. 5.1 uses N=31 for MNIST digits 6/7"
            if int(values.get("n_clients", 1)) == 31
            else "Layered local validation size",
        ),
        "tau": ParameterRecord(
            "tau",
            values.get("tau", 1),
            "Reference" if int(values.get("tau", 1)) in {2, 8} else "Validation",
            "LT-ADMM-VR Sec. IV selects tau=2"
            if int(values.get("tau", 1)) == 2
            else "Reference [13], Fig. 1 reports finite optimum at tau=8"
            if int(values.get("tau", 1)) == 8
            else "Training-holdout-selected SPADE local-update count",
        ),
        "vr_mode": ParameterRecord(
            "vr_mode",
            values.get("vr_mode", "persistent_saga"),
            "SPADE-DFL" if values.get("vr_mode") == "full_refresh" else "SPADE-DFL",
            "Algorithm 1 full refresh" if values.get("vr_mode") == "full_refresh" else "Sec. X-A practical persistent-SAGA variant",
        ),
        "beta": ParameterRecord(
            "beta", "derived", "TheoryDerived", "Midpoint 1.5/(tau*lambda_u*rho) of LT-ADMM theorem interval"
        ),
        "clip_norm": ParameterRecord(
            "clip_norm",
            values.get("clip_norm", 1.0),
            "Assumed",
            "SPADE Eq. (20) gives no numerical MNIST model-update clipping radius",
        ),
        "clip_exponent": ParameterRecord(
            "clip_exponent",
            values.get("clip_exponent", 0.0),
            "Validation",
            "SPADE Eq. (20) permits {C_k} but does not provide a numerical decay exponent",
        ),
        "dp_noise_std": ParameterRecord(
            "dp_noise_std",
            values.get("dp_noise_std", 0.0),
            "Assumed",
            "Direct SPADE model-release noise; no numerical value is supplied by the paper",
        ),
        "target_epsilon": ParameterRecord(
            "target_epsilon",
            values.get("target_epsilon", 0.0),
            "Assumed",
            "When positive, sigma is derived exactly from SPADE Eqs. (25)-(27)",
        ),
        "target_delta": ParameterRecord(
            "target_delta",
            values.get("target_delta", 0.0),
            "Assumed",
            "Unified DP comparison uses delta=1e-5 from Reference [24] MNIST",
        ),
        "pathological_minor_fraction": ParameterRecord(
            "pathological_minor_fraction",
            values.get("pathological_minor_fraction", 0.1),
            "Assumed",
            "Legacy binary mixed-label pathological option; not used by the reference-consistent profile",
        ),
        "local_epochs": ParameterRecord(
            "local_epochs",
            values.get("local_epochs", 1),
            "Assumed",
            "DP-FedAvg local epoch count for common outer-round comparison",
        ),
        "shared_initialization": ParameterRecord(
            "shared_initialization",
            values.get("shared_initialization", False),
            "Assumed",
            "SPADE-DFL Algorithm 1 permits a common initial x_i,0; enabled only after empirical validation",
        ),
        "initialization": ParameterRecord(
            "initialization",
            values.get("initialization", "independent_uniform"),
            "Reference"
            if values.get("initialization", "independent_uniform")
            == "independent_uniform"
            else "Validation",
            "Reference [20], Sec. 5.1 uses independent U([-1,1]^d) initial models"
            if values.get("initialization", "independent_uniform")
            == "independent_uniform"
            else "Common public initialization selected on the training holdout and shared fairly by all methods",
        ),
        "privacy_allocation": ParameterRecord(
            "privacy_allocation",
            values.get("privacy_allocation", "theorem_weighted"),
            "Assumed",
            "Per-round zCDP allocation policy; total privacy follows SPADE Eqs. (25)-(27)",
        ),
        "graph_topology": ParameterRecord(
            "graph_topology",
            values.get("graph_topology", "ring"),
            "Reference"
            if values.get("graph_topology") == "erdos_renyi"
            else "Assumed",
            "Reference [20], Sec. 5.1 (digits 6/7) uses a connected "
            "Erdos-Renyi graph; ring is retained as the backward-compatible default",
        ),
        "graph_probability": ParameterRecord(
            "graph_probability",
            values.get("graph_probability", 0.3),
            "Reference"
            if values.get("graph_topology", "ring") == "erdos_renyi"
            and abs(float(values.get("graph_probability", 0.3)) - 0.3) <= 1e-12
            and int(values.get("n_clients", 1)) in {31, 50, 100}
            else "Assumed",
            "Reference [20], Sec. 5.1 (digits 6/7): connected "
            "Erdos-Renyi graph with n=31 and p=0.3; the same paper also "
            "uses n=50/100 with p=0.3 for digits 1/2",
        ),
        "partition_mode": ParameterRecord(
            "partition_mode",
            values.get("partition_mode", "iid"),
            "Assumed",
            "User-required supplemental IID/non-IID comparison factor",
        ),
        "dirichlet_alpha": ParameterRecord(
            "dirichlet_alpha",
            values.get("dirichlet_alpha", 0.3),
            "Assumed",
            "User-required supplemental Dirichlet alpha values 0.3 and 0.6",
        ),
    }
    return rows


def load_config(path: Path) -> ExperimentConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig.from_dict(raw)
