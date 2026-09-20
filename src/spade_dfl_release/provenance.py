from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def freeze_sources(workspace_root: Path, specs_path: Path, output: Path) -> dict[str, object]:
    root = Path(workspace_root).resolve()
    specs = json.loads(Path(specs_path).read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for source in sorted(specs["sources"], key=lambda row: row["id"]):
        source_root = Path(source["root"])
        resolved = source_root if source_root.is_absolute() else root / source_root
        if source.get("required", False) and not resolved.exists():
            raise FileNotFoundError(f"missing required source {source['id']}: {resolved}")
        patterns = source.get("include", ["**/*"])
        paths = [resolved] if resolved.is_file() else [p for pattern in patterns for p in resolved.glob(pattern) if p.is_file()]
        for path in sorted(set(paths), key=lambda item: item.relative_to(resolved).as_posix() if resolved.is_dir() else item.name):
            rel = path.relative_to(resolved).as_posix() if resolved.is_dir() else path.name
            records.append({"source_id": source["id"], "path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {"schema_version": 1, "files": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
