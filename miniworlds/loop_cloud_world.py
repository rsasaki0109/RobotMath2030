"""Synthetic loop-shaped scans for topological data analysis demos."""

from __future__ import annotations

import numpy as np


def circle_cloud(n: int = 36, radius: float = 1.0, noise: float = 0.02, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    if noise > 0:
        pts = pts + rng.normal(0.0, noise, size=pts.shape)
    return pts


def two_loop_cloud(
    n_per_loop: int = 20,
    radius: float = 0.8,
    separation: float = 2.5,
    noise: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """Two separated loops — clearer β₁ = 2 teaching scene than a pinched figure-eight."""
    rng = np.random.default_rng(seed)
    left = circle_cloud(n_per_loop, radius, noise=0.0, seed=seed)
    left[:, 0] -= separation
    right = circle_cloud(n_per_loop, radius, noise=0.0, seed=seed + 1)
    right[:, 0] += separation
    pts = np.vstack([left, right])
    if noise > 0:
        pts = pts + rng.normal(0.0, noise, size=pts.shape)
    return pts


def corridor_cloud(length: float = 3.0, width: float = 0.05, n: int = 40, seed: int = 0) -> np.ndarray:
    """Open path — no 1-cycles when width stays small at local Rips scale."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, length, n)
    offsets = rng.uniform(-width, width, size=n)
    return np.column_stack([t, offsets])


def shuffled_scan(points: np.ndarray, seed: int = 0) -> np.ndarray:
    """Remove time/order from a trajectory sample — OT/TDA sees an unordered set."""
    rng = np.random.default_rng(seed)
    return points[rng.permutation(points.shape[0])]
