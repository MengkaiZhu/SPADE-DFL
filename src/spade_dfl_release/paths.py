from pathlib import Path


def release_root() -> Path:
    return Path(__file__).resolve().parents[2]
