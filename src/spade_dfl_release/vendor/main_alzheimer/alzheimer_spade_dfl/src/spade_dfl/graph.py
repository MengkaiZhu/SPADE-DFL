from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Edge = tuple[int, int]


@dataclass(frozen=True)
class Graph:
    adjacency: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.adjacency, dtype=np.int64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("adjacency must be square")
        if not np.array_equal(matrix, matrix.T) or np.any(np.diag(matrix)):
            raise ValueError("graph must be undirected with a zero diagonal")
        if np.any((matrix != 0) & (matrix != 1)):
            raise ValueError("adjacency entries must be zero or one")
        if not _is_connected(matrix):
            raise ValueError("graph must be connected")
        object.__setattr__(self, "adjacency", matrix)

    @classmethod
    def single(cls) -> Graph:
        return cls(np.zeros((1, 1), dtype=np.int64))

    @classmethod
    def ring(cls, n_clients: int) -> Graph:
        if n_clients <= 0:
            raise ValueError("n_clients must be positive")
        if n_clients == 1:
            return cls.single()
        adjacency = np.zeros((n_clients, n_clients), dtype=np.int64)
        for i in range(n_clients):
            adjacency[i, (i - 1) % n_clients] = 1
            adjacency[i, (i + 1) % n_clients] = 1
        return cls(adjacency)

    @property
    def n_clients(self) -> int:
        return self.adjacency.shape[0]

    def neighbors(self, client: int) -> np.ndarray:
        return np.flatnonzero(self.adjacency[client])

    @property
    def laplacian(self) -> np.ndarray:
        return np.diag(self.adjacency.sum(axis=1)) - self.adjacency

    @property
    def laplacian_eigenvalues(self) -> np.ndarray:
        values = np.linalg.eigvalsh(self.laplacian.astype(np.float64))
        values[np.abs(values) < 1e-12] = 0.0
        return values

    @property
    def metropolis_matrix(self) -> np.ndarray:
        degrees = self.adjacency.sum(axis=1)
        matrix = np.zeros_like(self.adjacency, dtype=np.float64)
        for i in range(self.n_clients):
            for j_raw in self.neighbors(i):
                j = int(j_raw)
                matrix[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
            matrix[i, i] = 1.0 - matrix[i].sum()
        return matrix


def _is_connected(adjacency: np.ndarray) -> bool:
    n = adjacency.shape[0]
    if n == 1:
        return True
    visited = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in np.flatnonzero(adjacency[node]):
            neighbor = int(neighbor)
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return len(visited) == n


def build_graph(
    n_clients: int,
    topology: str = "ring",
    probability: float = 0.3,
    seed: int = 0,
    max_attempts: int = 10_000,
) -> Graph:
    """Build the configured connected graph without sharing optimizer RNG state."""
    if n_clients <= 0:
        raise ValueError("n_clients must be positive")
    if n_clients == 1:
        return Graph.single()
    if topology == "ring":
        return Graph.ring(n_clients)
    if topology != "erdos_renyi":
        raise ValueError("topology must be ring or erdos_renyi")
    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must lie in (0, 1]")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        upper = rng.random((n_clients, n_clients)) < probability
        upper = np.triu(upper, k=1)
        adjacency = (upper | upper.T).astype(np.int64)
        if _is_connected(adjacency):
            return Graph(adjacency)
    raise RuntimeError(
        "failed to sample a connected Erdos-Renyi graph within "
        f"{max_attempts} attempts"
    )


def initialize_auxiliary(x: np.ndarray, graph: Graph) -> dict[Edge, np.ndarray]:
    states = np.asarray(x, dtype=np.float64)
    return {
        (i, int(j)): states[i].copy()
        for i in range(graph.n_clients)
        for j in graph.neighbors(i)
    }


def admm_penalties(
    x: np.ndarray,
    z_edges: dict[Edge, np.ndarray],
    graph: Graph,
    rho: float,
) -> np.ndarray:
    states = np.asarray(x, dtype=np.float64)
    penalties = np.zeros_like(states)
    for i in range(graph.n_clients):
        neighbors = graph.neighbors(i)
        if neighbors.size:
            summed = np.sum([z_edges[(i, int(j))] for j in neighbors], axis=0)
            penalties[i] = rho * neighbors.size * states[i] - summed
    return penalties


def exchange_messages(
    released_x: np.ndarray,
    z_edges: dict[Edge, np.ndarray],
    graph: Graph,
    rho: float,
) -> tuple[dict[Edge, np.ndarray], dict[Edge, np.ndarray]]:
    states = np.asarray(released_x, dtype=np.float64)
    messages: dict[Edge, np.ndarray] = {}
    for edge, auxiliary in z_edges.items():
        source, _ = edge
        messages[edge] = auxiliary - 2.0 * rho * states[source]
    updated: dict[Edge, np.ndarray] = {}
    for i in range(graph.n_clients):
        for j_raw in graph.neighbors(i):
            j = int(j_raw)
            updated[(i, j)] = 0.5 * (z_edges[(i, j)] - messages[(j, i)])
    return messages, updated


def derive_midpoint_beta(graph: Graph, tau: int, rho: float) -> float:
    if graph.n_clients < 2:
        raise ValueError("beta interval requires a multi-client graph")
    lambda_u = float(graph.laplacian_eigenvalues[-1])
    if tau <= 0 or rho <= 0 or lambda_u <= 0:
        raise ValueError("tau, rho, and lambda_u must be positive")
    return 1.5 / (tau * lambda_u * rho)
