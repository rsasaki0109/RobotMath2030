"""Occupancy map visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def draw_occupancy_map(ax, grid: np.ndarray, title: str = "", cmap: str = "Greys"):
    ax.imshow(grid, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_map_pair(ax, map_a: np.ndarray, map_b: np.ndarray, title: str = ""):
    diff = np.abs(map_a - map_b)
    panel = np.concatenate([map_a, np.ones((map_a.shape[0], 2)), map_b, np.ones((map_a.shape[0], 2)), diff], axis=1)
    ax.imshow(panel, origin="lower", cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_map_transport(
    ax,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    plan: np.ndarray,
    threshold: float = 0.0008,
    max_lines: int = 50,
):
    from robotmath.viz.plot_ot import draw_point_clouds, draw_transport_plan

    draw_transport_plan(ax, pts_a, pts_b, plan, threshold=threshold, max_lines=max_lines)
    draw_point_clouds(ax, pts_a, pts_b, label_x="map A", label_y="map B")
    ax.set_title("Wasserstein transport between occupied cells")
    ax.legend(loc="upper right", fontsize=8)
