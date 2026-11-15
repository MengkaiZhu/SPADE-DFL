from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spade_dfl.brain_tumor_data import load_brain_tumor_zip
from spade_dfl.brain_tumor_suite import (
    load_frozen_baselines,
    load_frozen_selection,
    run_brain_tumor_suite,
)
from spade_dfl.isolation import assert_write_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated seven-method Brain Tumor SPADE-DFL suite."
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "local", "server"),
        default="local",
        help="smoke=2 rounds/1 seed; local=50 rounds/3 seeds; server=50 rounds/30 seeds",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=PACKAGE_ROOT.parent / "1、Brain Tumor MRI Dataset.zip",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=PACKAGE_ROOT / "data" / "brain_tumor_pca10_28x28.npz",
    )
    parser.add_argument(
        "--frozen",
        type=Path,
        default=PACKAGE_ROOT / "frozen" / "selected_parameters_robust.json",
    )
    parser.add_argument(
        "--baseline-frozen",
        type=Path,
        default=PACKAGE_ROOT
        / "frozen"
        / "baseline_selected_parameters.json",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen = load_frozen_selection(args.frozen)
    frozen_baselines = load_frozen_baselines(args.baseline_frozen)
    profiles = {
        "smoke": (2, (0,)),
        "local": (int(frozen["n_rounds"]), (0, 1, 2)),
        "server": (50, tuple(range(30))),
    }
    rounds, seeds = profiles[args.profile]
    output = args.output or PACKAGE_ROOT / "results" / args.profile
    cache = assert_write_path(args.cache, PACKAGE_ROOT)
    dataset = load_brain_tumor_zip(args.zip, cache_path=cache)
    start = time.perf_counter()
    run_brain_tumor_suite(
        dataset,
        frozen_selection=frozen,
        frozen_baselines=frozen_baselines,
        output_root=output,
        seeds=seeds,
        n_rounds=rounds,
    )
    summary = {
        "status": "completed",
        "profile": args.profile,
        "rounds": rounds,
        "seeds": list(seeds),
        "elapsed_seconds": time.perf_counter() - start,
        "output": str(output.resolve()),
        "dataset_sha256": dataset.sha256,
        "official_test_used_for_selection": False,
        "official_test_role": "final_evaluation_only",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
