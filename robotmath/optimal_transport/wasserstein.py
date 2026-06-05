"""Wasserstein distances via entropic regularization."""

from __future__ import annotations

import numpy as np

from robotmath.optimal_transport.sinkhorn import sinkhorn2


def wasserstein2_sinkhorn(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float = 0.05,
    **kwargs,
) -> tuple[float, np.ndarray, list[float]]:
    """
    Approximate squared Wasserstein-2 distance between empirical distributions.

    Returns (W2^2 estimate, transport plan, convergence gaps).
    """
    cost, P, gaps = sinkhorn2(X, Y, epsilon=epsilon, **kwargs)
    return cost, P, gaps


def wasserstein_between_maps(
    map_a: np.ndarray,
    map_b: np.ndarray,
    map_to_points,
    epsilon: float = 0.01,
    **kwargs,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare occupancy maps by treating occupied cells as point distributions.

    Returns W2^2 estimate, transport plan, and point sets for visualization.
    """
    pts_a = map_to_points(map_a)
    pts_b = map_to_points(map_b)
    if pts_a.size == 0 or pts_b.size == 0:
        raise ValueError("maps must contain occupied cells")
    cost, plan, _ = wasserstein2_sinkhorn(pts_a, pts_b, epsilon=epsilon, **kwargs)
    return cost, plan, pts_a, pts_b


def compare_map_metrics(
    map_a: np.ndarray,
    map_b: np.ndarray,
    map_to_points,
    l2_fn,
    epsilon: float = 0.01,
) -> dict[str, float]:
    """Return L2 grid MSE and Wasserstein^2 for a map pair."""
    w2, _, _, _ = wasserstein_between_maps(
        map_a, map_b, map_to_points, epsilon=epsilon, max_iter=500,
    )
    return {
        "l2_grid_mse": l2_fn(map_a, map_b),
        "wasserstein2": w2,
    }
