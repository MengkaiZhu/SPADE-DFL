"""Import frozen results; no experiments are executed."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spade_dfl_release.reference import import_reference_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-index', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    result = import_reference_results(args.source_index, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
