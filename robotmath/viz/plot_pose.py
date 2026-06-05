"""Pose visualization helpers."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from robotmath.lie import se2


def draw_pose(ax, T: np.ndarray, color: str = "C0", scale: float = 0.4, label: str | None = None):
    """Draw a 2D pose as an arrow at (x,y) pointing along heading."""
    x, y, theta = se2.to_xytheta(T)
    dx = scale * np.cos(theta)
    dy = scale * np.sin(theta)
    ax.arrow(
        x, y, dx, dy,
        head_width=0.12 * scale,
        head_length=0.12 * scale,
        fc=color,
        ec=color,
        length_includes_head=True,
    )
    if label:
        ax.text(x, y + 0.15, label, ha="center", fontsize=9, color=color)


def draw_pose_graph(
    ax,
    poses: list[np.ndarray],
    edges: list[tuple[int, int]] | None = None,
    colors: list[str] | None = None,
):
    """Draw pose graph nodes and optional edges."""
    n = len(poses)
    colors = colors or [f"C{i % 10}" for i in range(n)]
    xs, ys = [], []
    for i, T in enumerate(poses):
        x, y, _ = se2.to_xytheta(T)
        xs.append(x)
        ys.append(y)
        draw_pose(ax, T, color=colors[i], label=str(i))
    if edges:
        for i, j in edges:
            xi, yi, _ = se2.to_xytheta(poses[i])
            xj, yj, _ = se2.to_xytheta(poses[j])
            ax.plot([xi, xj], [yi, yj], "k--", alpha=0.35, linewidth=1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def trajectory_from_poses(poses: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    xy = np.array([se2.to_xytheta(T)[:2] for T in poses])
    return xy[:, 0], xy[:, 1]
