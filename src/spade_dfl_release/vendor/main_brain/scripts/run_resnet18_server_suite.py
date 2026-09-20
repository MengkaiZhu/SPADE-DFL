from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spade_dfl.brain_tumor_formal_figures import write_formal_accuracy_figures
from spade_dfl.brain_tumor_resnet18_data import (
    FeatureMatrix,
    ResNet18V1Extractor,
    build_full_dataset_bundle,
    extract_split_features,
    load_feature_cache,
    write_feature_cache,
)
from spade_dfl.brain_tumor_suite import (
    BRAIN_TUMOR_PANELS,
    load_frozen_baselines,
    load_frozen_selection,
    run_brain_tumor_suite,
)
from spade_dfl.isolation import BRAIN_TUMOR_METHOD_IDS, assert_write_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_profile_config(path: Path, *, expected_profile: str) -> dict[str, Any]:
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("profile") != expected_profile:
        raise ValueError("configuration profile does not match the requested profile")
    if tuple(profile.get("panels", ())) != BRAIN_TUMOR_PANELS:
        raise ValueError("configuration does not contain the exact four-panel contract")
    if tuple(profile.get("methods", ())) != BRAIN_TUMOR_METHOD_IDS:
        raise ValueError("configuration does not contain the exact seven-method contract")
    seeds = profile.get("seeds")
    rounds = profile.get("rounds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("configuration seeds must be a non-empty unique list")
    if not isinstance(rounds, int) or rounds <= 0:
        raise ValueError("configuration rounds must be a positive integer")
    expected = len(BRAIN_TUMOR_PANELS) * len(BRAIN_TUMOR_METHOD_IDS) * len(seeds)
    if profile.get("expected_run_count") != expected:
        raise ValueError("configuration run count does not match panels x methods x seeds")
    if profile.get("official_testing_role") != "final_evaluation_only":
        raise ValueError("official Testing must be declared final_evaluation_only")
    return profile


def _write_cache_sidecar(
    cache_path: Path,
    matrix: FeatureMatrix,
    metadata: dict[str, object],
) -> None:
    sidecar = cache_path.with_suffix(cache_path.suffix + ".metadata.json")
    payload = dict(metadata)
    payload.update(
        {
            "feature_cache_path": str(cache_path.resolve()),
            "feature_cache_sha256": sha256(cache_path),
            "dataset_sha256": matrix.dataset_sha256,
            "sample_order_sha256": matrix.sample_order_sha256,
            "feature_shape": list(matrix.features.shape),
            "split": matrix.split,
        }
    )
    sidecar.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_or_extract(
    zip_path: Path,
    split: str,
    cache_path: Path,
    extractor: ResNet18V1Extractor,
    profile: dict[str, Any],
) -> tuple[FeatureMatrix, dict[str, object]]:
    if cache_path.is_file():
        matrix, metadata = load_feature_cache(cache_path)
    else:
        matrix = extract_split_features(zip_path, split, extractor)
        metadata = extractor.audit_metadata()
        metadata.update(
            {
                "split": split,
                "official_testing_read": split == "Testing",
                "representation": profile["representation"],
            }
        )
        write_feature_cache(cache_path, matrix, metadata)
        _write_cache_sidecar(cache_path, matrix, metadata)
    if matrix.split != split:
        raise ValueError(f"feature cache split mismatch: expected {split}")
    if matrix.dataset_sha256 != profile["dataset_sha256"]:
        raise ValueError(f"feature cache dataset hash mismatch for {split}")
    if matrix.extractor_name != extractor.name:
        raise ValueError(f"feature cache extractor mismatch for {split}")
    if metadata.get("weight_sha256") != profile["resnet18_weight_sha256"]:
        raise ValueError(f"feature cache weight hash mismatch for {split}")
    expected_count = int(profile["expected_split_counts"][split])
    if matrix.features.shape != (expected_count, 512):
        raise ValueError(f"feature cache shape mismatch for {split}")
    return matrix, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen ResNet18-V1/PCA10 seven-method Brain Tumor suite."
    )
    parser.add_argument("--profile", choices=("smoke", "server"), default="smoke")
    parser.add_argument(
        "--zip",
        type=Path,
        default=PACKAGE_ROOT.parent / "1、Brain Tumor MRI Dataset.zip",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_path = PACKAGE_ROOT / "configs" / f"resnet18_{args.profile}.json"
    profile = load_profile_config(profile_path, expected_profile=args.profile)
    zip_path = Path(args.zip).resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Brain Tumor MRI ZIP not found: {zip_path}")
    actual_zip_hash = sha256(zip_path)
    if actual_zip_hash != profile["dataset_sha256"]:
        raise ValueError("dataset ZIP SHA-256 does not match the frozen configuration")

    extractor = ResNet18V1Extractor(
        device=args.device,
        batch_size=args.feature_batch_size,
    )
    if extractor.weight_sha256 != profile["resnet18_weight_sha256"]:
        raise ValueError("ResNet18 IMAGENET1K_V1 weight SHA-256 mismatch")
    data_dir = assert_write_path(PACKAGE_ROOT / "data", PACKAGE_ROOT)
    training, training_metadata = _load_or_extract(
        zip_path,
        "Training",
        data_dir / "resnet18_v1_training_512d.npz",
        extractor,
        profile,
    )
    testing, testing_metadata = _load_or_extract(
        zip_path,
        "Testing",
        data_dir / "resnet18_v1_testing_512d.npz",
        extractor,
        profile,
    )
    dataset = build_full_dataset_bundle(training, testing, dimension=10)

    frozen = load_frozen_selection(
        PACKAGE_ROOT / str(profile["selected_parameter_source"])
    )
    frozen_baselines = load_frozen_baselines(
        PACKAGE_ROOT / str(profile["baseline_parameter_source"])
    )
    output = args.output or PACKAGE_ROOT / "results" / args.profile
    output = assert_write_path(Path(output), PACKAGE_ROOT)
    started = time.perf_counter()
    aggregates = run_brain_tumor_suite(
        dataset,
        frozen_selection=frozen,
        frozen_baselines=frozen_baselines,
        output_root=output,
        seeds=tuple(int(seed) for seed in profile["seeds"]),
        n_rounds=int(profile["rounds"]),
        panels=tuple(profile["panels"]),
        methods=tuple(profile["methods"]),
        resume=True,
        write_figures=False,
    )
    write_formal_accuracy_figures(
        aggregates,
        output / "figures_formal",
        n_rounds=int(profile["rounds"]),
    )
    completed_count = sum(1 for _ in (output / "raw").rglob("COMPLETED"))
    if completed_count != int(profile["expected_run_count"]):
        raise RuntimeError(
            f"completed run count {completed_count} does not match "
            f"expected {profile['expected_run_count']}"
        )
    elapsed = time.perf_counter() - started
    manifest = {
        "status": "completed",
        "protocol_id": profile["protocol_id"],
        "profile": args.profile,
        "rounds": int(profile["rounds"]),
        "seeds": profile["seeds"],
        "panels": profile["panels"],
        "methods": profile["methods"],
        "expected_run_count": int(profile["expected_run_count"]),
        "completed_run_count": completed_count,
        "elapsed_seconds": elapsed,
        "dataset_zip": str(zip_path),
        "dataset_sha256": actual_zip_hash,
        "training_feature_cache_sha256": sha256(
            data_dir / "resnet18_v1_training_512d.npz"
        ),
        "testing_feature_cache_sha256": sha256(
            data_dir / "resnet18_v1_testing_512d.npz"
        ),
        "training_feature_metadata": training_metadata,
        "testing_feature_metadata": testing_metadata,
        "resnet18_weight_sha256": extractor.weight_sha256,
        "official_testing_used_for_selection": False,
        "official_testing_role": profile["official_testing_role"],
        "formal_figure_contract": profile["formal_figure_contract"],
        "accuracy_statistics": "per-round population mean and population standard deviation across the declared seeds",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
