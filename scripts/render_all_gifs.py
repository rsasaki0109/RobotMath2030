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


def render_flow_matching_gif(path: Path):
    """Animate Heun integration for flow matching (Ch.08)."""
    from PIL import Image

    from robotmath.diffusion import FlowMatchingPolicy2D, FlowTrainConfig
    from robotmath.viz.plot_trajectory import draw_trajectories, draw_two_path_world

    horizon = 24
    demos, cond = generate_demonstrations(n_per_mode=40, horizon=horizon, seed=0)
    test_cond = np.concatenate([START, GOAL])[None, :]

    flow = FlowMatchingPolicy2D(FlowTrainConfig(horizon=horizon, epochs=100, sample_steps=16, seed=0))
    flow.fit(demos, cond, verbose=False)
    _, history = flow.sample(test_cond, n_samples=6, steps=16, integrator="heun", return_history=True)
    final = flow.sample(test_cond, n_samples=12, steps=16, integrator="heun")

    frames: list[Image.Image] = []
    for frame in range(len(history) + 6):
        fig, ax = plt.subplots(figsize=(5, 5))
        draw_two_path_world(ax)
        draw_trajectories(ax, final, color="C2", alpha=0.35, linewidth=1.2, label="samples")
        if frame < len(history):
            draw_trajectories(ax, history[frame], color="C1", linewidth=2.2, label="integrating")
            ax.set_title(f"Flow matching — Heun step {frame + 1}/{len(history)}")
        else:
            ax.set_title("Flow matching — multimodal samples")
        ax.legend(loc="upper left", fontsize=7)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())))
        plt.close(fig)

    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=130, loop=0)
    print(f"wrote {path}")


def render_world_model_gif(out_path: Path):
    """Closed-loop replanning on grid world (Ch.10)."""
    from miniworlds.grid_world import GOAL, WALL, GridWorld
    from robotmath.world_models import WorldModelConfig, rollout_env, train_world_model

    env = GridWorld(layout="easy")
    model, _ = train_world_model(
        env.collect_random_dataset(n_transitions=900, seed=0),
        WorldModelConfig(epochs=45, hidden=64, seed=0),
    )
    env.reset()
    env.agent = (1, 1)
    goal_obs = env.observation()
    agent_path, _ = rollout_env(
        env, model, goal_obs, replan=True, horizon=6, n_candidates=48, seed=1,
    )

    fig, ax = plt.subplots(figsize=(5, 5))

    def _draw_grid(prefix_len: int, title: str):
        ax.cla()
        grid = np.zeros_like(env.grid, dtype=float)
        grid[env.grid == WALL] = 1.0
        grid[env.grid == GOAL] = 0.6
        ax.imshow(grid, origin="upper", cmap="Greys", vmin=0, vmax=1)
        sub = agent_path[:prefix_len]
        if sub:
            ys = [p[0] for p in sub]
            xs = [p[1] for p in sub]
            ax.plot(xs, ys, "-o", color="C0", markersize=5, linewidth=2)
        ax.scatter([env.goal[1]], [env.goal[0]], c="gold", s=90, edgecolors="k", zorder=5)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    def update(frame):
        n = min(frame + 1, len(agent_path))
        _draw_grid(n, f"World model MPC — step {n}/{len(agent_path)}")
        return ax,

    anim = FuncAnimation(fig, update, frames=len(agent_path) + 4, blit=False)
    _save_animation(anim, out_path, fps=4)
    plt.close(fig)


def render_tda_loop_gif(path: Path):
    """β₁ sweep on a loop cloud vs corridor (Ch.14)."""
    from miniworlds.loop_cloud_world import circle_cloud, corridor_cloud, shuffled_scan
    from robotmath.topology.persistence import filtration_sweep

    circle = shuffled_scan(circle_cloud(n=36, noise=0.015, seed=2), seed=9)
    corridor = shuffled_scan(corridor_cloud(n=36, seed=4), seed=11)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    def update(frame):
        for ax, pts, name in zip(
            axes,
            (circle, corridor),
            ("circle scan", "corridor scan"),
        ):
            ax.cla()
            ax.scatter(pts[:, 0], pts[:, 1], s=16, c="C0", alpha=0.85)
            th, beta1, _ = filtration_sweep(pts)
            idx = min(frame, len(th) - 1)
            loops = int(beta1[idx])
            ax.set_aspect("equal")
            ax.set_title(f"{name}: β₁={loops} @ ε={th[idx]:.2f}")
            ax.grid(True, alpha=0.25)
        fig.suptitle("TDA — loops appear on circle, not on open corridor", fontsize=11)
        return axes

    anim = FuncAnimation(fig, update, frames=9, blit=False)
    _save_animation(anim, path, fps=2)
    plt.close(fig)


def main():
    plt.switch_backend("Agg")
    render_pose_graph_gif(OUT_DIR / "pose_graph_loop_closure.gif")
    render_sinkhorn_gif(OUT_DIR / "sinkhorn_point_clouds.gif")
    render_diffusion_policy_gif(OUT_DIR / "diffusion_policy_2d.gif")
    render_flow_matching_gif(OUT_DIR / "flow_matching_2d.gif")
    render_world_model_gif(OUT_DIR / "world_model_mpc.gif")
    render_tda_loop_gif(OUT_DIR / "tda_loop_detection.gif")
    print("Done.")


if __name__ == "__main__":
    main()
