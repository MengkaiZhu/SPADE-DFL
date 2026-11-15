"""Recompute saved main-comparison runs, without any training or raw datasets."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spade_dfl_release.main_recompute import verify
from spade_dfl_release.paths import release_root

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--verify-existing',action='store_true')
    modes.add_argument('--output',type=Path,help='new directory outside this release for recomputed aggregates')
    args=parser.parse_args()
    print(json.dumps(verify(release_root(),args.output),indent=2))
