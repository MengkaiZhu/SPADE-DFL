"""Capability-narrowing view for training-only Alzheimer validation."""

from __future__ import annotations

from typing import Any

import numpy as np


class ValidationOnlySplitView:
    """Expose inner training/validation state while guarding outer-test fields."""

    __slots__ = (
        "_forbidden_access_fields",
        "bundle",
        "model_indices",
        "validation_indices",
    )
    FORBIDDEN_FIELDS = ("test_x", "test_y", "test_indices")
    EXPOSED_FIELDS = ("bundle", "model_indices", "validation_indices")

    def __init__(
        self,
        *,
        bundle: Any,
        model_indices: np.ndarray,
        validation_indices: np.ndarray,
    ) -> None:
        self.bundle = bundle
        self.model_indices = np.asarray(model_indices, dtype=np.int64).copy()
        self.validation_indices = np.asarray(
            validation_indices, dtype=np.int64
        ).copy()
        self._forbidden_access_fields: list[str] = []

    @classmethod
    def from_loader_result(cls, loader_result: Any) -> ValidationOnlySplitView:
        """Narrow the loader result without dereferencing any outer-test field."""
        return cls(
            bundle=loader_result.bundle,
            model_indices=loader_result.model_indices,
            validation_indices=loader_result.validation_indices,
        )

    def __getattr__(self, name: str) -> Any:
        if name in self.FORBIDDEN_FIELDS:
            self._forbidden_access_fields.append(name)
            raise RuntimeError(
                f"outer final-test field {name!r} is forbidden during tuning"
            )
        raise AttributeError(name)

    def firewall_evidence(self) -> dict[str, object]:
        """Return evidence derived from guarded state and structural exposure."""
        accesses = list(self._forbidden_access_fields)
        slots = set(type(self).__slots__)
        return {
            "forbidden_access_count": len(accesses),
            "forbidden_access_fields": accesses,
            "forbidden_fields": list(self.FORBIDDEN_FIELDS),
            "outer_fields_structurally_absent": all(
                field not in slots for field in self.FORBIDDEN_FIELDS
            ),
            "exposed_fields": list(self.EXPOSED_FIELDS),
        }
