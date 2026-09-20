"""Build the reviewer-facing SPADE-DFL submission archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spade_dfl_release.release import build_release, verification_status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_source = ROOT
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--zip", type=Path, default=default_source.parent / "SPADE_DFL_submission_code.zip")
    parser.add_argument(
        "--report",
        type=Path,
        default=default_source.parent / "SPADE_DFL_submission_code_verification.json",
    )
    for name in ("source-tests", "clean-room-tests", "public-verifiers", "installation"):
        parser.add_argument("--" + name + "-status", choices=("passed", "failed", "not-run"), default="not-run")
        parser.add_argument("--" + name + "-summary")
    args = parser.parse_args()

    result = build_release(args.source, args.zip)
    statuses = {
        "source_tests": args.source_tests_status,
        "clean_room_tests": args.clean_room_tests_status,
        "public_verifiers": args.public_verifiers_status,
        "installation": args.installation_status,
    }
    report = {
        **result,
        "verification_status": verification_status(statuses),
        "results": {
            "source_tests": {"status": statuses["source_tests"], "summary": args.source_tests_summary},
            "clean_room_tests": {"status": statuses["clean_room_tests"], "summary": args.clean_room_tests_summary},
            "public_verifiers": {"status": statuses["public_verifiers"], "summary": args.public_verifiers_summary},
            "installation": {"status": statuses["installation"], "summary": args.installation_summary},
        },
        "commands": {
            "install": "python -m pip install --no-deps -e submission_code",
            "source_tests": "python -B -m pytest -q -p no:cacheprovider submission_code/tests",
            "main": "python -B submission_code/experiments/01_main_comparison.py --verify-existing",
            "sensitivity": "python -B submission_code/experiments/02_sensitivity.py --verify-existing",
            "heterogeneity": "python -B submission_code/experiments/03_heterogeneity.py --verify-existing",
            "communication": "python -B submission_code/experiments/04_communication.py --verify-existing",
            "paper_outputs": "python -B submission_code/experiments/make_paper_outputs.py --verify",
        },
    }
    args.report.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
