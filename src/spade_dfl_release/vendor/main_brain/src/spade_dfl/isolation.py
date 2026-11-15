from __future__ import annotations

import hashlib
from pathlib import Path


BRAIN_TUMOR_METHOD_IDS = (
    "zo2_lt_admm_vr",
    "zo2_lt_admm",
    "zo2_dp_sgd",
    "zo2_dp_fedavg",
    "one_point_dsg",
    "one_point_dsgt",
    "spade_dfl",
)


def assert_write_path(path: Path, experiment_root: Path) -> Path:
    resolved = Path(path).resolve()
    root = Path(experiment_root).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"write path is outside Brain Tumor experiment root: {resolved}")
    return resolved


def snapshot_workspace(root: Path, excluded_root: Path) -> dict[str, str]:
    workspace = Path(root).resolve()
    excluded = Path(excluded_root).resolve()
    if workspace != excluded and workspace not in excluded.parents:
        raise ValueError("excluded root must lie within workspace")
    snapshot: dict[str, str] = {}
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved == excluded or excluded in resolved.parents:
            continue
        relative = resolved.relative_to(workspace).as_posix()
        snapshot[relative] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return snapshot
