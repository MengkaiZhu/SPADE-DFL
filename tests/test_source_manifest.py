from pathlib import Path
import json

import pytest

from spade_dfl_release.provenance import freeze_sources, sha256_file


def test_freeze_sources_rejects_a_missing_required_root(tmp_path: Path):
    specs = tmp_path / "source_specs.json"
    specs.write_text(
        json.dumps({"sources": [{"id": "mnist_s7", "root": "missing", "required": True}]}),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="mnist_s7"):
        freeze_sources(tmp_path, specs, tmp_path / "manifest.json")


def test_manifest_is_sorted_and_hashes_file_bytes(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "b.py").write_bytes(b"b\n")
    (root / "a.py").write_bytes(b"a\n")
    specs = tmp_path / "source_specs.json"
    specs.write_text(
        json.dumps({"sources": [{"id": "sample", "root": "source", "include": ["*.py"], "required": True}]}),
        encoding="utf-8",
    )
    result = freeze_sources(tmp_path, specs, tmp_path / "manifest.json")
    assert [row["path"] for row in result["files"]] == ["a.py", "b.py"]
    assert result["files"][0]["sha256"] == sha256_file(root / "a.py")
