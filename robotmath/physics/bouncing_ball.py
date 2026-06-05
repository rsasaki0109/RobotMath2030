"""Bouncing ball with contact — gradients break at hard impacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass
class BounceParams:
    gravity: float = 9.81
    restitution: float = 0.85
    dt: float = 0.02


def simulate_bounce_hard(
    params: BounceParams,
    y0: float,
    vy0: float,
    steps: int,
) -> Array:
    """Hard floor at y=0. Returns (steps+1, 2) array [height, vy]."""
    traj = np.zeros((steps + 1, 2), dtype=float)
    y, vy = y0, vy0
    traj[0] = [y, vy]
    g, e, dt = params.gravity, params.restitution, params.dt
    for t in range(steps):
        vy -= g * dt
        y += vy * dt
        if y < 0.0:
            y = 0.0
            vy = -e * vy
        traj[t + 1] = [y, vy]
    return traj


def simulate_bounce_soft(
    params: BounceParams,
    y0: float,
    vy0: float,
    steps: int,
    contact_stiffness: float = 800.0,
) -> Array:
    """Soft penalty contact — differentiable but wrong physics if too soft."""
    traj = np.zeros((steps + 1, 2), dtype=float)
    y, vy = y0, vy0
    traj[0] = [y, vy]
    g, dt = params.gravity, params.dt
    for t in range(steps):
        penetration = min(0.0, y)
        fy = -g - contact_stiffness * penetration
        vy += fy * dt
        y += vy * dt
        traj[t + 1] = [y, vy]
    return traj


def bounce_height_mse(params: BounceParams, target_y: Array, y0: float, vy0: float) -> float:
    pred = simulate_bounce_hard(params, y0, vy0, target_y.shape[0] - 1)
    return float(np.mean((pred[:, 0] - target_y) ** 2))
