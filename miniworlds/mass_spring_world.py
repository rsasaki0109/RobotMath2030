"""Synthetic trajectories for system identification demos."""

from __future__ import annotations

import numpy as np

from robotmath.physics.bouncing_ball import BounceParams, simulate_bounce_hard
from robotmath.physics.mass_spring import MassSpringParams, simulate


def noisy_mass_spring_rollout(
    true_k: float = 8.0,
    true_c: float = 0.5,
    x0: float = 1.0,
    v0: float = 0.0,
    steps: int = 80,
    noise: float = 0.01,
    seed: int = 0,
) -> tuple[np.ndarray, MassSpringParams]:
    params = MassSpringParams(spring_k=true_k, damping=true_c)
    traj = simulate(params, x0, v0, steps)
    rng = np.random.default_rng(seed)
    return traj + rng.normal(0.0, noise, size=traj.shape), params


def bouncing_ball_rollout(
    gravity: float = 9.81,
    y0: float = 1.5,
    vy0: float = 0.0,
    steps: int = 120,
    noise: float = 0.002,
    seed: int = 0,
) -> tuple[np.ndarray, BounceParams]:
    params = BounceParams(gravity=gravity)
    traj = simulate_bounce_hard(params, y0, vy0, steps)
    rng = np.random.default_rng(seed)
    noisy = traj.copy()
    noisy[:, 0] += rng.normal(0.0, noise, size=traj.shape[0])
    return noisy, params
