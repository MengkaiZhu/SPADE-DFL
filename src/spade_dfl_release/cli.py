from __future__ import annotations

import argparse
from pathlib import Path

from .paths import release_root

FAMILIES = ("main", "sensitivity", "heterogeneity", "communication")


def build_parser(fixed_family: str | None = None) -> argparse.ArgumentParser:
    description = None
    if fixed_family is not None:
        description = f"Fixed paper experiment family: {fixed_family}"
    parser = argparse.ArgumentParser(prog="spade-dfl-paper", description=description)
    if fixed_family is None:
        parser.add_argument("family", choices=FAMILIES)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-existing", action="store_true")
    mode.add_argument("--rerun", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None, fixed_family: str | None = None) -> int:
    parser = build_parser(fixed_family)
    args = parser.parse_args(argv)
    if args.rerun and args.output is None:
        parser.error("--rerun requires --output pointing to a new directory")
    if args.rerun:
        output = args.output.resolve()
        reference_root = (release_root() / "reference_results").resolve()
        if args.output.exists() or output == reference_root or reference_root in output.parents:
            parser.error("--rerun requires --output pointing to a new directory outside reference_results")
    from .runners import execute_family
    family = fixed_family if fixed_family is not None else args.family
    execute_family(family, "rerun" if args.rerun else "verify", args.output)
    return 0
