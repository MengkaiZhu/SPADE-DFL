import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from spade_dfl_release import release
from spade_dfl_release.release import build_release, validate_archive_name, verification_status


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_is_sorted_portable_deterministic_and_contains_no_cache(tmp_path, release_root):
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    report = build_release(release_root, first)
    build_release(release_root, second)

    with ZipFile(first) as archive:
        names = archive.namelist()
    assert names == sorted(names)
    assert all(name.startswith("submission_code/") for name in names)
    assert all("\\" not in name for name in names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
    assert report["file_count"] == len(names)
    assert first.read_bytes() == second.read_bytes()
    assert report["archive_sha256"] == _sha256(first)


def test_archive_excludes_every_external_input_and_local_freezer(tmp_path, release_root):
    zip_path = tmp_path / "release.zip"
    build_release(release_root, zip_path)
    policy = json.loads((release_root / "data/checksums.json").read_text(encoding="utf-8"))
    with ZipFile(zip_path) as archive:
        names = set(archive.namelist())

    for row in policy["files"]:
        assert row["included_in_release"] is False
        assert "submission_code/" + row["placement"] not in names
    assert "submission_code/tools/freeze_sources.py" not in names
    assert "submission_code/MANIFEST.json" in names
    assert "submission_code/SHA256SUMS.txt" in names


def test_generated_manifests_match_archived_bytes(tmp_path, release_root):
    zip_path = tmp_path / "release.zip"
    report = build_release(release_root, zip_path)
    manifest = json.loads((release_root / "MANIFEST.json").read_text(encoding="utf-8"))
    checksum_lines = (release_root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    checksums = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in checksum_lines}

    with ZipFile(zip_path) as archive:
        for row in manifest["files"]:
            archived = archive.read("submission_code/" + row["path"])
            assert len(archived) == row["bytes"]
            assert hashlib.sha256(archived).hexdigest() == row["sha256"]
        for name, digest in checksums.items():
            assert hashlib.sha256(archive.read("submission_code/" + name)).hexdigest() == digest

    assert manifest["file_count"] == len(manifest["files"])
    assert checksums["MANIFEST.json"] == _sha256(release_root / "MANIFEST.json")
    assert report["manifest_sha256"] == checksums["MANIFEST.json"]
    assert report["paper_contract"] == {
        "status": "verified",
        "manuscript_sha256": "4eafcd1bfd5ebb609f37d056fe0705589495d83c5abefcbeec097737dab7a9ab",
        "difference_count": 0,
        "missing_cells": 0,
        "duplicate_cells": 0,
        "displayed_numeric_fields": 326,
    }


@pytest.mark.parametrize("name", ["/absolute.py", "../escape.py", "a/../../escape.py", "C" + ":/private.py", "a\\b.py"])
def test_archive_name_validation_rejects_absolute_traversal_and_backslashes(name):
    with pytest.raises(ValueError):
        validate_archive_name(name)


def test_builder_rejects_a_symlink_without_needing_symlink_privilege(tmp_path, monkeypatch):
    source = tmp_path / "submission_code"
    (source / "src").mkdir(parents=True)
    probe = source / "src/probe.py"
    probe.write_text("pass\n", encoding="utf-8")
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == probe or original(path))
    with pytest.raises(ValueError, match="symlink"):
        list(release._iter_files(source, set()))


def test_builder_rejects_undeclared_suffix_cache_and_oversize(tmp_path, monkeypatch):
    source = tmp_path / "submission_code"
    (source / "src").mkdir(parents=True)

    bad = source / "src/payload.exe"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="suffix"):
        list(release._iter_files(source, set()))
    bad.unlink()

    cache = source / "src/__pycache__/module.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"x")
    with pytest.raises(ValueError, match="cache"):
        list(release._iter_files(source, set()))
    cache.unlink()
    cache.parent.rmdir()

    metadata = source / "src/package.egg-info/PKG-INFO"
    metadata.parent.mkdir()
    metadata.write_text("generated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cache"):
        list(release._iter_files(source, set()))
    metadata.unlink()
    metadata.parent.rmdir()

    large = source / "src/large.py"
    large.write_bytes(b"xxxx")
    monkeypatch.setattr(release, "MAX_FILE_BYTES", 3)
    with pytest.raises(ValueError, match="large"):
        list(release._iter_files(source, set()))


def test_builder_fails_closed_without_the_external_data_policy(tmp_path):
    source = tmp_path / "submission_code"
    (source / "src").mkdir(parents=True)
    (source / "src/probe.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksums.json"):
        build_release(source, tmp_path / "release.zip")


@pytest.mark.parametrize("mutation", ["removed", "extra"])
def test_external_data_policy_must_be_the_exact_23_manifest_inputs(release_root, monkeypatch, mutation):
    policy_path = (release_root / "data/checksums.json").resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if mutation == "removed":
        policy["files"].pop()
    else:
        added = dict(policy["files"][-1])
        added["placement"] = "src/spade_dfl_release/vendor/unregistered/input.npz"
        policy["files"].append(added)
    encoded = json.dumps(policy)
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: encoded if path.resolve() == policy_path else original(path, *args, **kwargs),
    )
    with pytest.raises(ValueError, match="23|manifest inputs"):
        release._external_exclusions(release_root)


def test_verification_status_requires_four_structured_passes():
    passed = {name: "passed" for name in ("source_tests", "clean_room_tests", "public_verifiers", "installation")}
    assert verification_status(passed) == "complete"
    for name in passed:
        failed = dict(passed)
        failed[name] = "failed"
        assert verification_status(failed) == "incomplete"
    with pytest.raises(ValueError):
        verification_status(dict(passed, extra="passed"))
