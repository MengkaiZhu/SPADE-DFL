"""Strict cache validation for Alzheimer tuning run evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

SCIENTIFIC_FIELDS = (
    "status",
    "finite",
    "validation_accuracy",
    "validation_loss",
    "stationarity_consensus",
    "query_count",
    "beta",
)
IDENTITY_FIELDS = (
    "fingerprint",
    "candidate_id",
    "partition",
    "seed",
    "artifact_path",
    "frozen_before",
)


def load_metrics_cache(path: Path) -> Optional[dict[str, object]]:  # noqa: UP045
    """Return a JSON-object cache or None for any corrupt/unreadable cache."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        return None
    return payload if isinstance(payload, dict) else None


def canonical_evidence(
    scientific: Mapping[str, object],
    identity: Mapping[str, object],
) -> dict[str, object]:
    """Compose evidence only from recomputed science and current identity."""
    return {
        **{field: scientific[field] for field in SCIENTIFIC_FIELDS},
        **{field: identity[field] for field in IDENTITY_FIELDS},
    }


def cache_matches_canonical(
    stored: Optional[Mapping[str, object]],  # noqa: UP045
    canonical: Mapping[str, object],
) -> bool:
    """Require exact equality for every scientific and identity cache field."""
    if stored is None:
        return False
    fields = (*SCIENTIFIC_FIELDS, *IDENTITY_FIELDS)
    return all(
        field in stored
        and stored[field] == canonical[field]
        for field in fields
    )


def valid_failed_cache(
    stored: Optional[Mapping[str, object]],  # noqa: UP045
    identity: Mapping[str, object],
) -> bool:
    """Validate the complete schema of a deterministic failed-run record."""
    if stored is None or stored.get("status") != "failed":
        return False
    if any(stored.get(field) != identity[field] for field in IDENTITY_FIELDS):
        return False
    required = {
        "finite": False,
        "validation_accuracy": None,
        "validation_loss": None,
        "stationarity_consensus": None,
        "query_count": 0,
        "beta": None,
    }
    return all(stored.get(field) == value for field, value in required.items()) and (
        isinstance(stored.get("failure_reason"), str)
        and isinstance(stored.get("failure_message"), str)
    )
