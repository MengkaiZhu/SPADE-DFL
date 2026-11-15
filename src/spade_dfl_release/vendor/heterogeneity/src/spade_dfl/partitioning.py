from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PartitionResult:
    indices: tuple[np.ndarray, ...]
    classes: np.ndarray
    label_histograms: np.ndarray
    mode: str
    alpha: float


def _result(
    partitions: list[np.ndarray], labels: np.ndarray, mode: str, alpha: float
) -> PartitionResult:
    classes = np.unique(labels)
    histograms = np.zeros((len(partitions), len(classes)), dtype=np.int64)
    for client, indices in enumerate(partitions):
        for class_index, label in enumerate(classes):
            histograms[client, class_index] = int(np.sum(labels[indices] == label))
    return PartitionResult(
        indices=tuple(np.asarray(indices, dtype=np.int64) for indices in partitions),
        classes=classes,
        label_histograms=histograms,
        mode=mode,
        alpha=float(alpha),
    )


def partition_labels(
    labels: np.ndarray,
    n_clients: int,
    mode: str,
    rng: np.random.Generator,
    alpha: float = 0.3,
    max_retries: int = 100,
    pathological_minor_fraction: float = 0.1,
) -> PartitionResult:
    labels = np.asarray(labels)
    if labels.ndim != 1 or n_clients <= 0 or n_clients > labels.size:
        raise ValueError("labels must be one-dimensional and n_clients must be valid")
    if mode == "iid":
        shuffled = rng.permutation(labels.size)
        partitions = [part.copy() for part in np.array_split(shuffled, n_clients)]
        return _result(partitions, labels, mode, alpha)
    if mode == "dirichlet":
        if alpha <= 0:
            raise ValueError("alpha must be positive for dirichlet partitioning")
        classes = np.unique(labels)
        for _ in range(max_retries):
            buckets: list[list[int]] = [[] for _ in range(n_clients)]
            for label in classes:
                class_indices = rng.permutation(np.flatnonzero(labels == label))
                proportions = rng.dirichlet(np.full(n_clients, alpha))
                counts = rng.multinomial(class_indices.size, proportions)
                offsets = np.cumsum(counts)[:-1]
                for client, group in enumerate(np.split(class_indices, offsets)):
                    buckets[client].extend(int(index) for index in group)
            if all(buckets):
                partitions = [rng.permutation(bucket) for bucket in buckets]
                return _result(partitions, labels, mode, alpha)
        raise ValueError("dirichlet partition failed to produce nonempty clients")
    if mode == "dirichlet_balanced":
        if alpha <= 0:
            raise ValueError("alpha must be positive for dirichlet partitioning")
        classes = np.unique(labels)
        target_sizes = np.array(
            [part.size for part in np.array_split(np.arange(labels.size), n_clients)],
            dtype=np.int64,
        )
        remaining = target_sizes.copy()
        buckets: list[list[int]] = [[] for _ in range(n_clients)]
        class_preferences = rng.dirichlet(
            np.full(n_clients, alpha), size=classes.size
        )
        for class_index, label in enumerate(classes):
            for sample_index in rng.permutation(np.flatnonzero(labels == label)):
                probabilities = class_preferences[class_index] * (remaining > 0)
                if probabilities.sum() == 0:
                    probabilities = remaining.astype(np.float64)
                probabilities /= probabilities.sum()
                client = int(rng.choice(n_clients, p=probabilities))
                buckets[client].append(int(sample_index))
                remaining[client] -= 1
        if np.any(remaining != 0):
            raise RuntimeError("balanced dirichlet allocation did not fill all capacities")
        partitions = [rng.permutation(bucket) for bucket in buckets]
        return _result(partitions, labels, mode, alpha)
    if mode == "pathological":
        classes = np.unique(labels)
        if n_clients < classes.size:
            raise ValueError(
                "pure pathological partition requires at least one client per class"
            )
        assignments = np.arange(n_clients) % classes.size
        partitions: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n_clients)]
        for class_index, label in enumerate(classes):
            clients = np.flatnonzero(assignments == class_index)
            class_indices = rng.permutation(np.flatnonzero(labels == label))
            for client, group in zip(clients, np.array_split(class_indices, clients.size)):
                partitions[int(client)] = group.copy()
        if any(partition.size == 0 for partition in partitions):
            raise ValueError("pathological partition produced an empty client")
        return _result(partitions, labels, mode, alpha)
    if mode == "pathological_mixed":
        classes = np.unique(labels)
        if classes.size != 2:
            raise ValueError("binary-adapted pathological partition requires two labels")
        if not 0.0 < pathological_minor_fraction < 0.5:
            raise ValueError("pathological_minor_fraction must lie in (0, 0.5)")
        target_sizes = np.array(
            [part.size for part in np.array_split(np.arange(labels.size), n_clients)],
            dtype=np.int64,
        )
        if np.any(target_sizes < 2):
            raise ValueError("each client needs at least two samples")
        majority_first = np.arange(n_clients) % 2 == 0
        desired_first_fraction = np.where(
            majority_first,
            1.0 - pathological_minor_fraction,
            pathological_minor_fraction,
        )
        desired_first = target_sizes * desired_first_fraction
        first_counts = np.floor(desired_first).astype(np.int64)
        first_counts = np.clip(first_counts, 1, target_sizes - 1)
        required_first = int(np.sum(labels == classes[0]))
        while int(first_counts.sum()) < required_first:
            candidates = np.flatnonzero(first_counts < target_sizes - 1)
            if candidates.size == 0:
                raise ValueError("cannot allocate both labels to every client")
            scores = desired_first[candidates] - first_counts[candidates]
            first_counts[int(candidates[np.argmax(scores)])] += 1
        while int(first_counts.sum()) > required_first:
            candidates = np.flatnonzero(first_counts > 1)
            if candidates.size == 0:
                raise ValueError("cannot allocate both labels to every client")
            scores = first_counts[candidates] - desired_first[candidates]
            first_counts[int(candidates[np.argmax(scores)])] -= 1
        class_groups = []
        for class_index, label in enumerate(classes):
            class_indices = rng.permutation(np.flatnonzero(labels == label))
            counts = first_counts if class_index == 0 else target_sizes - first_counts
            offsets = np.cumsum(counts)[:-1]
            class_groups.append(np.split(class_indices, offsets))
        partitions = [
            rng.permutation(np.concatenate([class_groups[0][client], class_groups[1][client]]))
            for client in range(n_clients)
        ]
        return _result(partitions, labels, mode, alpha)
    if mode == "pathological_shards":
        n_shards = 2 * n_clients
        if labels.size < n_shards:
            raise ValueError("pathological shards require at least two samples per client")
        # Stable label sorting forms low-entropy shards; random shard assignment
        # then gives each client exactly two shards, as in the common FedAvg
        # pathological non-IID construction.
        sorted_indices = np.argsort(labels, kind="stable")
        shards = [part.copy() for part in np.array_split(sorted_indices, n_shards)]
        shard_order = rng.permutation(n_shards)
        partitions = []
        for client in range(n_clients):
            selected = [
                shards[int(shard_order[2 * client])],
                shards[int(shard_order[2 * client + 1])],
            ]
            partitions.append(rng.permutation(np.concatenate(selected)))
        return _result(partitions, labels, mode, alpha)
    raise ValueError(
        "mode must be iid, dirichlet, dirichlet_balanced, pathological, "
        "pathological_mixed, or pathological_shards"
    )
