from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path

from .paths import release_root

MODULES = {
    "main": "spade_dfl_release.main_comparison",
    "sensitivity": "spade_dfl_release.sensitivity",
    "heterogeneity": "spade_dfl_release.heterogeneity",
    "communication": "spade_dfl_release.communication",
}


def execute_family(family: str, mode: str, output: Path | None = None) -> dict[str, object]:
    if family not in MODULES or mode not in {"verify", "rerun"}:
        raise ValueError(f"unsupported family/mode: {family}/{mode}")
    if mode == "rerun":
        if output is None:
            raise ValueError("--rerun requires --output pointing to a new directory")
        output_path = Path(output)
        reference_root = (release_root() / "reference_results").resolve()
        resolved_output = output_path.resolve()
        if output_path.exists() or resolved_output == reference_root or reference_root in resolved_output.parents:
            raise ValueError("--rerun requires --output pointing to a new directory outside reference_results")
    module = import_module(MODULES[family])
    if mode == "verify":
        report = module.verify(release_root() / "reference_results")
    else:
        report = module.run(output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
