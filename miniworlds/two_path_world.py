"""Two-path 2D navigation world for multimodal trajectory demos."""

from __future__ import annotations

import numpy as np

START = np.array([0.5, 0.08], dtype=float)
GOAL = np.array([0.5, 0.92], dtype=float)
OBSTACLE_CENTER = np.array([0.5, 0.55], dtype=float)
OBSTACLE_RADIUS = 0.12
LEFT_VIA_X = 0.22
RIGHT_VIA_X = 0.78


def make_bezier_path(
    start: np.ndarray,
    via: np.ndarray,
    goal: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Quadratic Bezier through via point, shape (horizon, 2)."""
    t = np.linspace(0.0, 1.0, horizon)
    path = (
        (1 - t)[:, None] ** 2 * start
        + 2 * (1 - t)[:, None] * t[:, None] * via
        + t[:, None] ** 2 * goal
    )
    return path.astype(float)


def left_path(horizon: int = 24) -> np.ndarray:
    via = np.array([LEFT_VIA_X, 0.55])
    return make_bezier_path(START, via, GOAL, horizon)


def right_path(horizon: int = 24) -> np.ndarray:
    via = np.array([RIGHT_VIA_X, 0.55])
    return make_bezier_path(START, via, GOAL, horizon)


def mean_regression_path(horizon: int = 24) -> np.ndarray:
    """Straight-line interpolation — collides with the obstacle."""
    t = np.linspace(0.0, 1.0, horizon)
    return START + t[:, None] * (GOAL - START)


def generate_demonstrations(
    n_per_mode: int = 64,
    horizon: int = 24,
    noise: float = 0.012,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return trajectories (N, horizon, 2) and conditions (N, 4) = start + goal.
    Half left, half right demonstrations.
    """
    rng = np.random.default_rng(seed)
    trajs: list[np.ndarray] = []
    for _ in range(n_per_mode):
        trajs.append(left_path(horizon) + rng.normal(0.0, noise, size=(horizon, 2)))
    for _ in range(n_per_mode):
        trajs.append(right_path(horizon) + rng.normal(0.0, noise, size=(horizon, 2)))
    data = np.stack(trajs, axis=0)
    cond = np.tile(np.concatenate([START, GOAL])[None, :], (data.shape[0], 1))
    return data, cond


def hits_obstacle(
    traj: np.ndarray,
    center: np.ndarray = OBSTACLE_CENTER,
    radius: float = OBSTACLE_RADIUS,
) -> bool:
    traj = np.asarray(traj, dtype=float)
    dist = np.linalg.norm(traj - center, axis=-1)
    return bool(np.any(dist < radius))


def collision_rate(trajs: np.ndarray) -> float:
    trajs = np.asarray(trajs)
    if trajs.ndim == 2:
        return float(hits_obstacle(trajs))
    return float(np.mean([hits_obstacle(t) for t in trajs]))
