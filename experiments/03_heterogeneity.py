from pathlib import Path
import sys

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from spade_dfl_release.cli import main


if __name__ == "__main__":
    raise SystemExit(main(fixed_family="heterogeneity"))
