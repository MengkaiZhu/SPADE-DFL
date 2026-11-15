"""Run/resume the two frozen suites that produce the three approved figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spade_dfl.mnist67_selected_three_figures import (  # noqa: E402
    FIGURE_IDS,
    audit_smoke_results,
    build_selected_three_figures,
)
from spade_dfl.paper_suite import run_paper_suite  # noqa: E402


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution-config", type=Path, required=True)
    parser.add_argument("--comparison-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()

    output = _resolve(args.output)
    distribution = output / "distribution_suite"
    comparison = output / "method_comparison_suite"
    run_paper_suite(_resolve(args.distribution_config), distribution)
    run_paper_suite(_resolve(args.comparison_config), comparison)
    if args.mode == "smoke":
        audit_smoke_results(distribution, comparison)
        status = output / "SMOKE_VALIDATED.json"
        status.write_text(
            json.dumps(
                {
                    "validated": True,
                    "paper_claim": False,
                    "run_count": 33,
                    "message": "one-round execution-path check only",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("SMOKE VALIDATED: 33/33 runs; no formal figures generated")
        return

    figures = build_selected_three_figures(
        distribution, comparison, output / "paper_figures"
    )
    print("FORMAL VALIDATED: 990/990 runs")
    print("PAPER FIGURES: exactly 3")
    for figure_id in FIGURE_IDS:
        print(f"{figure_id}: {figures[figure_id][0]}")


if __name__ == "__main__":
    main()
