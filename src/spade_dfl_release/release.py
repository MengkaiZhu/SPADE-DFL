"""Deterministic, allowlisted packaging for the reviewer-facing release."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Set
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ALLOWED_ROOTS = {
    "configs",
    "data",
    "docs",
    "experiments",
    "paper_outputs",
    "reference_results",
    "scripts",
    "src",
    "tests",
    "tools",
}
ALLOWED_ROOT_FILES = {
    "README.md",
    "IMPLEMENTATION_NOTE.md",
    "NOTICE.md",
    "requirements.txt",
    "requirements-resnet18-cpu.txt",
    "requirements-verified.txt",
    "pyproject.toml",
    "MANIFEST.json",
    "SHA256SUMS.txt",
}
ALLOWED_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".pdf",
    ".ps1",
    ".py",
    ".sh",
    ".svg",
    ".tex",
    ".toml",
    ".txt",
}
FORBIDDEN_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
    ".git",
    ".superpowers",
}
FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
    ".aux",
    ".out",
    ".toc",
    ".fls",
    ".fdb_latexmk",
    ".bbl",
    ".blg",
}
EXPLICIT_EXCLUDES = {"tools/freeze_sources.py"}
GENERATED_FILES = {"MANIFEST.json", "SHA256SUMS.txt"}
MAX_FILE_BYTES = 10_000_000
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_VERIFICATION_STEPS = {"source_tests", "clean_room_tests", "public_verifiers", "installation"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def validate_archive_name(name: str) -> str:
    """Return a portable relative archive name or reject it."""
    if not name or "\\" in name or name.startswith("/") or _WINDOWS_DRIVE.match(name):
        raise ValueError(f"unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive path: {name!r}")
    return path.as_posix()


def verification_status(statuses: Dict[str, str]) -> str:
    """Return complete only for the exact four structured passing gates."""
    if set(statuses) != _VERIFICATION_STEPS:
        raise ValueError("verification status must contain the exact four release gates")
    if any(value not in {"passed", "failed", "not-run"} for value in statuses.values()):
        raise ValueError("verification gate status must be passed, failed, or not-run")
    return "complete" if all(value == "passed" for value in statuses.values()) else "incomplete"


def _manifest_external_inputs(source: Path) -> Dict[str, tuple]:
    expected = {}
    main_path = source / "configs/main_comparison/manifest.json"
    sensitivity_path = source / "configs/sensitivity/manifest.json"
    heterogeneity_path = source / "configs/heterogeneity/manifest.json"
    for required in (main_path, sensitivity_path, heterogeneity_path):
        if not required.is_file():
            raise ValueError(f"required external-input manifest is missing: {required.relative_to(source).as_posix()}")
    main = json.loads(main_path.read_text(encoding="utf-8"))
    for config in main["datasets"].values():
        for row in config["required_inputs"]:
            path = (source / config["vendor_root"] / row["path"]).resolve()
            try:
                placement = path.relative_to(source).as_posix()
            except ValueError as error:
                raise ValueError("main external input escapes the release root") from error
            expected[placement] = (row["sha256"].lower(), "configs/main_comparison/manifest.json", row["path"])
    for family, manifest_path in (("sensitivity", sensitivity_path), ("heterogeneity", heterogeneity_path)):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["required_inputs"]:
            placement = validate_archive_name(f"src/spade_dfl_release/vendor/{family}/{row['path']}")
            expected[placement] = (row["sha256"].lower(), f"configs/{family}/manifest.json", row["path"])
    if len(expected) != 23:
        raise ValueError("external-input manifests must define exactly 23 unique inputs")
    return expected


def _external_exclusions(source: Path) -> Set[str]:
    policy_path = source / "data/checksums.json"
    if not policy_path.is_file():
        raise ValueError("required external data policy is missing: data/checksums.json")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    rows = policy.get("files", [])
    expected = _manifest_external_inputs(source)
    exclusions = set()
    actual = {}
    for row in rows:
        placement = validate_archive_name(row["placement"])
        if row.get("included_in_release") is not False:
            raise ValueError(f"external input is not explicitly excluded: {placement}")
        if row.get("release_builder_exclusion_required") is not True:
            raise ValueError(f"external input lacks the exclusion gate: {placement}")
        exclusions.add(placement)
        actual[placement] = (row["sha256"].lower(), row["manifest"], row["manifest_input_path"])
    if len(rows) != 23 or len(actual) != 23 or actual != expected:
        raise ValueError("data/checksums.json must match the exact 23 manifest inputs")
    return exclusions


def _iter_files(source: Path, exclusions: Set[str], *, include_generated: bool = False) -> Iterable[Path]:
    for path in sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()):
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed: {relative}")
        parts = path.relative_to(source).parts
        if any(part.lower() in FORBIDDEN_DIRS or part.lower().endswith(".egg-info") for part in parts):
            raise ValueError(f"cache/environment directory is not allowed: {relative}")
        if path.is_dir():
            if len(parts) == 1 and parts[0] not in ALLOWED_ROOTS:
                raise ValueError(f"undeclared release root: {relative}")
            continue
        if len(parts) == 1 and relative not in ALLOWED_ROOT_FILES:
            raise ValueError(f"undeclared release root file: {relative}")
        if (relative in GENERATED_FILES and not include_generated) or relative in EXPLICIT_EXCLUDES or relative in exclusions:
            continue
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES or path.name.endswith(".synctex.gz"):
            raise ValueError(f"cache/build file is not allowed: {relative}")
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f"undeclared suffix is not allowed: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"file is too large for the declared allowance: {relative}")
        validate_archive_name(relative)
        yield path


def _file_rows(source: Path, paths: Iterable[Path]) -> List[Dict[str, object]]:
    rows = []
    for path in paths:
        relative = path.relative_to(source).as_posix()
        data = path.read_bytes()
        rows.append({"path": relative, "bytes": len(data), "sha256": _sha256_bytes(data)})
    return rows


def _write_zip(source: Path, rows: List[Dict[str, object]], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for row in sorted(rows, key=lambda item: str(item["path"])):
            relative = validate_archive_name(str(row["path"]))
            archive_name = validate_archive_name("submission_code/" + relative)
            info = ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (source / relative).read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)


def build_release(source: Path, zip_path: Path) -> Dict[str, object]:
    """Build a deterministic ZIP and return independently checkable facts."""
    source = Path(source).resolve(strict=True)
    zip_path = Path(zip_path).resolve()
    try:
        zip_path.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("release ZIP must be outside the source tree")

    exclusions = _external_exclusions(source)
    payload_rows = _file_rows(source, _iter_files(source, exclusions))
    manifest = {
        "schema_version": 1,
        "archive_root": "submission_code",
        "scope": "Payload files only; MANIFEST.json and SHA256SUMS.txt are excluded to avoid self-reference.",
        "excluded_external_inputs": sorted(exclusions),
        "excluded_local_tools": sorted(EXPLICIT_EXCLUDES),
        "file_count": len(payload_rows),
        "files": payload_rows,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    (source / "MANIFEST.json").write_bytes(manifest_bytes)

    checksum_rows = payload_rows + [
        {"path": "MANIFEST.json", "bytes": len(manifest_bytes), "sha256": _sha256_bytes(manifest_bytes)}
    ]
    checksum_text = "".join(
        f"{row['sha256']}  {row['path']}\n" for row in sorted(checksum_rows, key=lambda item: str(item["path"]))
    )
    (source / "SHA256SUMS.txt").write_bytes(checksum_text.encode("utf-8"))

    final_paths = list(_iter_files(source, exclusions, include_generated=True))
    final_rows = _file_rows(source, final_paths)
    expected_paths = {str(row["path"]) for row in payload_rows} | GENERATED_FILES
    if {str(row["path"]) for row in final_rows} != expected_paths:
        raise ValueError("generated release inventory is inconsistent")
    _write_zip(source, final_rows, zip_path)
    paper_verification_path = source / "reference_results/paper_values/verification.json"
    paper_contract = None
    if paper_verification_path.is_file():
        paper = json.loads(paper_verification_path.read_text(encoding="utf-8"))
        paper_contract = {
            "status": paper.get("status"),
            "manuscript_sha256": paper.get("source_hashes", {}).get("paper_tex/spade_dfl_v1.tex"),
            "difference_count": paper.get("difference_count"),
            "missing_cells": paper.get("missing_cells"),
            "duplicate_cells": paper.get("duplicate_cells"),
            "displayed_numeric_fields": paper.get("displayed_numeric_fields"),
        }
    return {
        "schema_version": 1,
        "archive": zip_path.name,
        "archive_bytes": zip_path.stat().st_size,
        "archive_sha256": _sha256_file(zip_path),
        "file_count": len(final_rows),
        "payload_file_count": len(payload_rows),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "excluded_external_input_count": len(exclusions),
        "excluded_external_inputs": sorted(exclusions),
        "excluded_local_tools": sorted(EXPLICIT_EXCLUDES),
        "paper_contract": paper_contract,
    }
