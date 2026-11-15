from collections.abc import Sequence

ALZHEIMER_METHOD_IDS = (
    "zo2_lt_admm_vr",
    "zo2_lt_admm",
    "zo2_dp_sgd",
    "zo2_dp_fedavg",
    "one_point_dsg",
    "one_point_dsgt",
    "spade_dfl",
)


def validate_alzheimer_method_contract(methods: Sequence[str]) -> None:
    if tuple(methods) != ALZHEIMER_METHOD_IDS:
        raise ValueError(
            "Alzheimer suite must use the exact ordered seven-method contract"
        )
