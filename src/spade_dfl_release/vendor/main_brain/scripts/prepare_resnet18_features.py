from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from spade_dfl.brain_tumor_data import BRAIN_TUMOR_CLASS_NAMES
from spade_dfl.brain_tumor_resnet18_data import (
    ResNet18V1Extractor,
    extract_split_features,
    write_feature_cache,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen official ResNet18-V1 features from one declared split."
    )
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--split", choices=("training", "testing"), default="training")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    sidecar = output.with_suffix(output.suffix + ".metadata.json")
    existing = [path for path in (output, sidecar) if path.exists()]
    if existing and not args.force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing output: {joined}")
    extractor = ResNet18V1Extractor(
        device=args.device,
        batch_size=args.batch_size,
    )
    started = time.perf_counter()
    split = args.split.title()
    matrix = extract_split_features(args.zip, split, extractor)
    elapsed_seconds = time.perf_counter() - started
    class_counts = {
        class_name: int(np.sum(matrix.labels == label))
        for label, class_name in enumerate(BRAIN_TUMOR_CLASS_NAMES)
    }
    metadata = extractor.audit_metadata()
    metadata.update(
        {
            "split": split,
            "official_testing_read": split == "Testing",
            "feature_shape": list(matrix.features.shape),
            "finite": bool(np.isfinite(matrix.features).all()),
            "class_counts": class_counts,
            "dataset_sha256": matrix.dataset_sha256,
            "sample_order_sha256": matrix.sample_order_sha256,
            "source_path": matrix.source_path,
            "elapsed_seconds": elapsed_seconds,
        }
    )
    write_feature_cache(output, matrix, metadata)
    sidecar_payload = dict(metadata)
    sidecar_payload.update(
        {
            "feature_cache_path": str(output),
            "feature_cache_sha256": _sha256(output),
        }
    )
    sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sidecar_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
