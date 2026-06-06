"""Trajectory visualization for 2D policy demos."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from miniworlds.two_path_world import (
    GOAL,
    OBSTACLE_CENTER,
    OBSTACLE_RADIUS,
    START,
)


def draw_two_path_world(ax, alpha_obstacle: float = 0.5):
    ax.add_patch(plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color="gray", alpha=alpha_obstacle))
    ax.scatter(*START, c="k", s=40, zorder=4, label="start")
    ax.scatter(*GOAL, c="gold", s=50, edgecolors="k", zorder=4, label="goal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)


def draw_trajectories(
    ax,
    trajs: np.ndarray,
    color: str = "C0",
    alpha: float = 0.6,
    label: str | None = None,
    linewidth: float = 1.5,
):
    trajs = np.asarray(trajs, dtype=float)
    if trajs.ndim == 2:
        trajs = trajs[None, ...]
    for i, traj in enumerate(trajs):
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=label if i == 0 else None,
        )
