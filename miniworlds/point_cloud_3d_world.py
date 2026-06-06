"""3D point cloud toy data for equivariance demos."""

from __future__ import annotations

import numpy as np

from robotmath.equivariance.se3_point_readout import random_rotation, rotate_points


def l_shape_cloud_3d(n_per_leg: int = 12, noise: float = 0.02, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    leg_a = np.column_stack([
        np.linspace(0.0, 0.6, n_per_leg),
        np.zeros(n_per_leg),
        np.zeros(n_per_leg),
    ])
    leg_b = np.column_stack([
        np.zeros(n_per_leg),
        np.linspace(0.0, 0.5, n_per_leg),
        np.zeros(n_per_leg),
    ])
    pts = np.vstack([leg_a, leg_b])
    pts += rng.normal(0.0, noise, pts.shape)
    return pts.astype(float)


def grasp_direction_target(points: np.ndarray) -> np.ndarray:
    """Target 3-vector: from centroid toward a fixed body-frame landmark."""
    landmark = np.array([0.35, 0.25, 0.05], dtype=float)
    centroid = points.mean(axis=0)
    v = landmark - centroid
    return v.astype(float)


def make_dataset(
    n_clouds: int = 256,
    n_per_leg: int = 12,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (points, targets) with random SE(3) rotations applied per sample."""
    rng = np.random.default_rng(seed)
    base = l_shape_cloud_3d(n_per_leg=n_per_leg, seed=seed)
    target_base = grasp_direction_target(base)
    clouds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for i in range(n_clouds):
        R = random_rotation(seed + i)
        t = rng.normal(0.0, 0.05, size=3)
        cloud = rotate_points(base, R) + t
        tgt = rotate_points(target_base[None, :], R)[0]
        clouds.append(cloud)
        targets.append(tgt)
    return np.stack(clouds), np.stack(targets)
