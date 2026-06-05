"""Synthetic point clouds for optimal transport demos."""

from __future__ import annotations

import numpy as np


def _transform(X: np.ndarray, theta: float, tx: float, ty: float, noise: float, rng) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    Y = (X @ R.T) + np.array([tx, ty])
    if noise > 0:
        Y = Y + rng.normal(0.0, noise, size=Y.shape)
    return Y


def l_shape_cloud(n_per_edge: int = 20, scale: float = 1.0) -> np.ndarray:
    """L-shaped 2D point cloud resembling a partial room scan."""
    e1 = np.linspace(0, scale, n_per_edge)
    e2 = np.linspace(0, scale, n_per_edge)
    leg_a = np.column_stack([e1, np.zeros(n_per_edge)])
    leg_b = np.column_stack([np.zeros(n_per_edge), e2])
    return np.vstack([leg_a, leg_b[1:]])


def misaligned_pair(
    n_per_edge: int = 18,
    theta: float = 0.35,
    tx: float = 0.25,
    ty: float = -0.15,
    noise: float = 0.02,
    outlier_fraction: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Source and target point clouds with rigid misalignment + noise.

    Useful for showing why L2 on sorted indices fails but OT finds soft correspondence.
    """
    rng = np.random.default_rng(seed)
    source = l_shape_cloud(n_per_edge=n_per_edge)
    target = _transform(source, theta, tx, ty, noise, rng)

    if outlier_fraction > 0:
        n_out = max(1, int(outlier_fraction * source.shape[0]))
        outliers = rng.uniform(-0.2, 1.2, size=(n_out, 2))
        source = np.vstack([source, outliers])
        target = np.vstack([target, rng.uniform(-0.2, 1.2, size=(n_out, 2))])

    return source, target


def occupancy_samples(
    width: int = 16,
    height: int = 16,
    occupied_fraction: float = 0.35,
    seed: int = 0,
) -> np.ndarray:
    """Sample occupied grid cells as 2D points in [0,1]^2."""
    rng = np.random.default_rng(seed)
    xs = np.arange(width)
    ys = np.arange(height)
    grid = np.array([[x / width, y / height] for x in xs for y in ys])
    mask = rng.random(grid.shape[0]) < occupied_fraction
    return grid[mask]
