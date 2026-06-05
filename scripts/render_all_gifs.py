#!/usr/bin/env python3
"""Render README GIFs for RobotMath2030 demos."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniworlds.point_cloud_world import misaligned_pair
from miniworlds.pose_graph_world import square_loop_graph
from robotmath.lie import se2
from robotmath.optimal_transport.sinkhorn import pairwise_sq_distances, sinkhorn
from robotmath.viz import draw_pose_graph, draw_transport_plan, trajectory_from_poses

OUT_DIR = ROOT / "assets" / "animations"


def _save_animation(anim: FuncAnimation, path: Path, fps: int = 10):
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    print(f"wrote {path}")


def render_pose_graph_gif(path: Path):
    graph, gt = square_loop_graph(seed=42, loop_noise_theta=0.4)
    edges_idx = [(e.i, e.j) for e in graph.edges]
    _, _, history = graph.optimize(use_lie=True, max_iters=25, record_poses=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    (line_gt,) = ax.plot([], [], "g-", alpha=0.45, linewidth=2, label="ground truth")
    title = ax.set_title("Pose graph loop closure")

    def init():
        draw_pose_graph(ax, history[0], edges=edges_idx)
        gx, gy = trajectory_from_poses(gt)
        line_gt.set_data(gx, gy)
        ax.legend(loc="upper right")
        return line_gt,

    def update(frame):
        ax.cla()
        draw_pose_graph(ax, history[frame], edges=edges_idx)
        gx, gy = trajectory_from_poses(gt)
        ax.plot(gx, gy, "g-", alpha=0.45, linewidth=2, label="ground truth")
        ax.set_title(f"Lie pose graph optimization — iter {frame}")
        ax.legend(loc="upper right")
        return line_gt,

    anim = FuncAnimation(fig, update, init_func=init, frames=len(history), blit=False)
    _save_animation(anim, path, fps=8)
    plt.close(fig)


def _sinkhorn_snapshots(X, Y, epsilon=0.03, max_iter=80, every=4):
    n, m = X.shape[0], Y.shape[0]
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    C = pairwise_sq_distances(X, Y)
    K = np.exp(-C / epsilon)
    u = np.ones(n)
    v = np.ones(m)
    snaps: list[np.ndarray] = []
    for it in range(max_iter + 1):
        P = u[:, None] * K * v[None, :]
        if it % every == 0 or it == max_iter:
            snaps.append(P.copy())
        if it == max_iter:
            break
        u = a / (K @ v + 1e-16)
        v = b / (K.T @ u + 1e-16)
    return snaps


def render_sinkhorn_gif(path: Path):
    source, target = misaligned_pair(seed=7, theta=0.4, tx=0.2, ty=-0.1, noise=0.025)
    snaps = _sinkhorn_snapshots(source, target, epsilon=0.03, max_iter=100, every=5)

    fig, ax = plt.subplots(figsize=(5, 5))

    def update(frame):
        ax.cla()
        P = snaps[frame]
        draw_transport_plan(ax, source, target, P, threshold=0.0012, max_lines=70)
        ax.scatter(source[:, 0], source[:, 1], s=28, c="C0", label="source", zorder=3)
        ax.scatter(target[:, 0], target[:, 1], s=28, c="C3", label="target", zorder=3)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Sinkhorn OT — step {frame * 5}")
        ax.legend(loc="upper right", fontsize=8)

    anim = FuncAnimation(fig, update, frames=len(snaps), blit=False)
    _save_animation(anim, path, fps=6)
    plt.close(fig)


def render_diffusion_concept_gif(path: Path):
    """Concept GIF: unimodal mean vs multimodal paths (Ch.07 preview, NumPy only)."""
    from PIL import Image

    start = np.array([0.5, 0.08])
    goal = np.array([0.5, 0.92])
    left_path = np.column_stack([
        np.linspace(start[0], 0.22, 40),
        np.linspace(start[1], goal[1], 40),
    ])
    right_path = np.column_stack([
        np.linspace(start[0], 0.78, 40),
        np.linspace(start[1], goal[1], 40),
    ])
    mean_path = np.column_stack([
        np.full(40, 0.5),
        np.linspace(start[1], goal[1], 40),
    ])
    rng = np.random.default_rng(0)
    frames: list[Image.Image] = []

    for frame in range(40):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.add_patch(plt.Circle((0.5, 0.55), 0.12, color="gray", alpha=0.5))
        ax.scatter(*start, c="k", s=40)
        ax.scatter(*goal, c="gold", s=50, edgecolors="k")
        ax.plot(mean_path[:, 0], mean_path[:, 1], "C3--", linewidth=2, label="mean regression")
        ax.plot(left_path[:, 0], left_path[:, 1], "C0-", alpha=0.35, linewidth=1.5)
        ax.plot(right_path[:, 0], right_path[:, 1], "C0-", alpha=0.35, linewidth=1.5)

        idx = min(frame, 39)
        for _ in range(12):
            lp = left_path[idx] + rng.normal(0, 0.015, size=2)
            rp = right_path[idx] + rng.normal(0, 0.015, size=2)
            ax.scatter(lp[0], lp[1], s=18, c="C0", alpha=0.8)
            ax.scatter(rp[0], rp[1], s=18, c="C0", alpha=0.8)
        ax.scatter(mean_path[idx, 0], mean_path[idx, 1], s=60, c="C3", marker="x")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title("Multimodal actions need diffusion, not one mean")
        ax.legend(loc="upper left", fontsize=8)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba))
        plt.close(fig)

    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    print(f"wrote {path}")


def main():
    plt.switch_backend("Agg")
    render_pose_graph_gif(OUT_DIR / "pose_graph_loop_closure.gif")
    render_sinkhorn_gif(OUT_DIR / "sinkhorn_point_clouds.gif")
    render_diffusion_concept_gif(OUT_DIR / "diffusion_multimodal_concept.gif")
    print("Done.")


if __name__ == "__main__":
    main()
