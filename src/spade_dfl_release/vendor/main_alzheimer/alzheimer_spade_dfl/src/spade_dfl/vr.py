from __future__ import annotations

import numpy as np


class ZOMemory:
    """One-point ZO memory for SPADE-DFL equations (13)--(15)."""

    def __init__(
        self,
        entries: np.ndarray,
        mode: str,
        initial_queries: int = 0,
    ) -> None:
        values = np.asarray(entries, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("entries must have shape (n_samples, dimension)")
        if mode not in {"full_refresh", "persistent_saga"}:
            raise ValueError("mode must be full_refresh or persistent_saga")
        self.entries = values.copy()
        self.mean = self.entries.mean(axis=0)
        self.mode = mode
        self.refresh_queries = int(initial_queries)
        self.updated_queries = 0

    def control_variate(
        self, batch_ids: np.ndarray, current_queries: np.ndarray
    ) -> np.ndarray:
        ids = np.asarray(batch_ids, dtype=np.int64)
        current = np.asarray(current_queries, dtype=np.float64)
        if current.shape != (ids.size, self.entries.shape[1]):
            raise ValueError("current_queries shape does not match batch and dimension")
        return np.mean(current - self.entries[ids], axis=0) + self.mean

    def update(self, batch_ids: np.ndarray, current_queries: np.ndarray) -> None:
        ids = np.asarray(batch_ids, dtype=np.int64)
        current = np.asarray(current_queries, dtype=np.float64)
        if current.shape != (ids.size, self.entries.shape[1]):
            raise ValueError("current_queries shape does not match batch and dimension")
        n_samples = self.entries.shape[0]
        _, reverse_positions = np.unique(ids[::-1], return_index=True)
        last_positions = ids.size - 1 - reverse_positions
        changed_ids = ids[last_positions]
        new_entries = current[last_positions]
        old_entries = self.entries[changed_ids].copy()
        self.entries[changed_ids] = new_entries
        self.mean += np.sum(new_entries - old_entries, axis=0) / n_samples
        self.updated_queries += ids.size

    def refresh(self, entries: np.ndarray) -> None:
        if self.mode != "full_refresh":
            raise ValueError("round refresh is only valid in full_refresh mode")
        values = np.asarray(entries, dtype=np.float64)
        if values.shape != self.entries.shape:
            raise ValueError("refresh entries must preserve memory shape")
        self.entries[...] = values
        self.mean = self.entries.mean(axis=0)
        self.refresh_queries += values.shape[0]

    @property
    def query_count(self) -> int:
        return self.refresh_queries + self.updated_queries
