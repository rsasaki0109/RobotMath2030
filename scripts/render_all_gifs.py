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
from miniworlds.two_path_world import GOAL, START, generate_demonstrations
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


def render_diffusion_policy_gif(path: Path):
    """Train tiny diffusion policy and animate denoising (Ch.07)."""
    from PIL import Image

    from robotmath.diffusion import DiffusionPolicy2D, TrainConfig, predict_mean_regression, train_mean_regression
    from robotmath.viz.plot_trajectory import draw_trajectories, draw_two_path_world

    horizon = 24
    demos, cond = generate_demonstrations(n_per_mode=48, horizon=horizon, seed=0)
    test_cond = np.concatenate([START, GOAL])[None, :]

    mean_model = train_mean_regression(demos, cond, horizon=horizon, epochs=80, seed=0)
    mean_pred = predict_mean_regression(mean_model, test_cond, horizon=horizon)[0]

    cfg = TrainConfig(horizon=horizon, timesteps=20, epochs=100, seed=0)
    policy = DiffusionPolicy2D(cfg)
    policy.fit(demos, cond)
    _, history = policy.sample(test_cond, n_samples=1, return_history=True)
    samples = policy.sample(test_cond, n_samples=8)

    frames: list[Image.Image] = []
    n_hist = len(history)
    for frame in range(n_hist + 8):
        fig, ax = plt.subplots(figsize=(5, 5))
        draw_two_path_world(ax)
        draw_trajectories(ax, samples, color="C0", alpha=0.35, linewidth=1.2, label="samples")
        draw_trajectories(ax, mean_pred[None, ...], color="C3", linewidth=2.0, label="mean regression")
        if frame < n_hist:
            draw_trajectories(ax, history[frame][None, ...], color="C1", linewidth=2.5, label="denoising")
            ax.set_title(f"Diffusion policy — step {frame}/{n_hist - 1}")
        else:
            ax.set_title("Diffusion policy — multimodal samples")
        ax.legend(loc="upper left", fontsize=7)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())))
        plt.close(fig)

    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0)
    print(f"wrote {path}")


def main():
    plt.switch_backend("Agg")
    render_pose_graph_gif(OUT_DIR / "pose_graph_loop_closure.gif")
    render_sinkhorn_gif(OUT_DIR / "sinkhorn_point_clouds.gif")
    render_diffusion_policy_gif(OUT_DIR / "diffusion_policy_2d.gif")
    print("Done.")


if __name__ == "__main__":
    main()
