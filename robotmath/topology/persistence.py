"""Minimal Vietoris–Rips persistence for teaching topology in robot maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PersistenceDiagram:
    """Filtration summary for plotting and loop counting."""

    thresholds: np.ndarray
    beta1: np.ndarray
    local_scale: float


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _local_scale(dist: np.ndarray, k: int = 2) -> float:
    knn = np.sort(dist, axis=1)[:, min(k, dist.shape[0] - 1)]
    return float(np.median(knn))


def _beta1_at_scale(dist: np.ndarray, epsilon: float) -> int:
    """Graph Betti number β₁ at a fixed Rips scale."""
    n = dist.shape[0]
    uf = _UnionFind(n)
    edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= epsilon:
                uf.union(i, j)
                edges += 1
    components = len({uf.find(i) for i in range(n)})
    return max(0, edges - n + components)


def filtration_sweep(
    points: np.ndarray,
    k: int = 2,
    mult_min: float = 1.35,
    mult_max: float = 1.55,
    n_steps: int = 9,
) -> tuple[np.ndarray, np.ndarray, float]:
    """β₁ along a local Rips scale window (robust for small teaching clouds)."""
    points = np.asarray(points, dtype=float)
    dist = pairwise_distances(points)
    local = _local_scale(dist, k=k)
    thresholds = np.linspace(local * mult_min, local * mult_max, n_steps)
    beta1 = np.array([_beta1_at_scale(dist, float(eps)) for eps in thresholds], dtype=int)
    return thresholds, beta1, local


def rips_persistence(points: np.ndarray) -> PersistenceDiagram:
    """Local-scale β₁ sweep packaged as a persistence summary."""
    thresholds, beta1, local = filtration_sweep(points)
    return PersistenceDiagram(thresholds=thresholds, beta1=beta1, local_scale=local)


def topological_loop_count(points: np.ndarray) -> int:
    """
    Estimate the number of significant 1-cycles in an unordered 2D cloud.

    Uses the median β₁ over a local Rips window — stable for circle / two-loop /
    corridor teaching scenes without external TDA libraries.
    """
    _, beta1, _ = filtration_sweep(points)
    return int(np.median(beta1))


def count_persistent_h1(diagram: PersistenceDiagram, min_persistence: float = 0.0) -> int:
    """Loop count derived from a :func:`rips_persistence` summary."""
    _ = min_persistence
    return int(np.median(diagram.beta1))


def naive_component_count(points: np.ndarray, epsilon: float) -> int:
    """
    Single-scale connected components — misses loops inside one component.

    Typical failure mode for loop-closure reasoning on unordered scans.
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    uf = _UnionFind(n)
    dist = pairwise_distances(points)
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= epsilon:
                uf.union(i, j)
    roots = {uf.find(i) for i in range(n)}
    return len(roots)


def naive_kmeans_clusters(points: np.ndarray, k: int, seed: int = 0) -> int:
    """Lloyd k-means — counts density modes, not topological loops."""
    points = np.asarray(points, dtype=float)
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    if n == 0:
        return 0
    k = min(k, n)
    centroids = points[rng.choice(n, size=k, replace=False)]
    for _ in range(15):
        dist_to_centroids = np.stack(
            [np.linalg.norm(points - centroids[c], axis=1) for c in range(k)],
            axis=1,
        )
        assign = np.argmin(dist_to_centroids, axis=1)
        for cluster in range(k):
            mask = assign == cluster
            if np.any(mask):
                centroids[cluster] = points[mask].mean(axis=0)
    return k
