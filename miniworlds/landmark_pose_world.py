"""Landmark-based SE(2) pose estimation toy problem."""

from __future__ import annotations

import numpy as np

from robotmath.lie import se2


def landmark_pose_problem(
    seed: int = 0,
    noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (T_true, T_init, landmarks_body, landmarks_world).

    T_init has a large heading error near the pi boundary to expose projection failure.
    """
    rng = np.random.default_rng(seed)
    landmarks_body = np.array(
        [
            [0.55, 0.05],
            [0.05, 0.45],
            [-0.50, -0.05],
            [0.10, -0.40],
            [-0.15, 0.25],
        ],
        dtype=float,
    )
    T_true = se2.from_xytheta(1.1, 0.7, np.deg2rad(28.0))
    landmarks_world = np.array([se2.action(T_true, p) for p in landmarks_body])
    if noise > 0.0:
        landmarks_world += rng.normal(0.0, noise, landmarks_world.shape)

    T_init = se2.from_xytheta(0.15, 0.35, np.deg2rad(152.0))
    return T_true, T_init, landmarks_body, landmarks_world
