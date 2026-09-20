from pathlib import Path

import pytest


@pytest.fixture
def release_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def reference_root(release_root: Path) -> Path:
    return release_root / "reference_results"
