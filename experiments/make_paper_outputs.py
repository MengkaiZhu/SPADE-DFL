"""Generate the frozen manuscript tables and figures; never trains models."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spade_dfl_release.paper_outputs import write_all_paper_outputs, verify_paper_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=ROOT / "reference_results")
    parser.add_argument("--output", type=Path, default=ROOT / "paper_outputs")
    parser.add_argument("--verify", action="store_true", help="read-only verification of an existing generated directory")
    args = parser.parse_args()
    function = verify_paper_outputs if args.verify else write_all_paper_outputs
    print(json.dumps(function(args.reference_root, args.output), sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
