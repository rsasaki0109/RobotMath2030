"""Optimal transport visualization."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def draw_point_clouds(
    ax,
    X: np.ndarray,
    Y: np.ndarray,
    label_x: str = "source",
    label_y: str = "target",
):
    ax.scatter(X[:, 0], X[:, 1], s=30, c="C0", alpha=0.85, label=label_x, zorder=3)
    ax.scatter(Y[:, 0], Y[:, 1], s=30, c="C3", alpha=0.85, label=label_y, zorder=3)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def draw_transport_plan(
    ax,
    X: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    threshold: float = 0.002,
    max_lines: int = 80,
    line_alpha: float = 0.45,
):
    """Draw soft correspondences from transport plan P."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    P = np.asarray(P, dtype=float)
    pairs = []
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] >= threshold:
                pairs.append((P[i, j], i, j))
    pairs.sort(reverse=True)
    pairs = pairs[:max_lines]
    for mass, i, j in pairs:
        ax.plot(
            [X[i, 0], Y[j, 0]],
            [X[i, 1], Y[j, 1]],
            "k-",
            alpha=line_alpha * min(1.0, mass / threshold),
            linewidth=0.8,
            zorder=1,
        )


def plot_sinkhorn_panel(
    ax,
    X: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    title: str,
    threshold: float = 0.002,
):
    draw_transport_plan(ax, X, Y, P, threshold=threshold)
    draw_point_clouds(ax, X, Y)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
