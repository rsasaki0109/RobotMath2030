#!/usr/bin/env python3
"""Chapter 10: Tiny latent world model with imagination planning."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from miniworlds.grid_world import WALL, GOAL, GridWorld
from robotmath.world_models import WorldModelConfig, rollout_env, train_world_model


def _draw_path(ax, env: GridWorld, path: list[tuple[int, int]], title: str):
    grid = np.zeros_like(env.grid, dtype=float)
    grid[env.grid == WALL] = 1.0
    grid[env.grid == GOAL] = 0.6
    ax.imshow(grid, origin="upper", cmap="Greys", vmin=0, vmax=1)
    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    ax.plot(xs, ys, "-o", color="C0", markersize=4, linewidth=1.5)
    ax.scatter([env.goal[1]], [env.goal[0]], c="gold", s=80, edgecolors="k", zorder=5)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    print("Chapter 10 — Tiny latent world model + planning")
    env = GridWorld(layout="easy")
    data = env.collect_random_dataset(n_transitions=2000, seed=0)
    cfg = WorldModelConfig(epochs=70, hidden=64, seed=0)
    print("Training tiny world model...")
    model, losses = train_world_model(data, cfg)
    print(f"Final training loss: {losses[-1]:.5f}")

    goal_obs = env.reset()
    gy, gx = env.goal
    env.agent = (1, 1)
    goal_obs = env.observation()

    closed_path, closed_ok = rollout_env(
        env, model, goal_obs, replan=True, horizon=6, n_candidates=64, seed=1,
    )
    open_path, open_ok = rollout_env(
        env, model, goal_obs, replan=False, horizon=5, n_candidates=128, seed=1,
    )

    print(f"Closed-loop replanning success: {closed_ok}")
    print(f"Open-loop imagination success:  {open_ok}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    _draw_path(axes[0], env, closed_path, "Closed-loop MPC (replan each step)")
    _draw_path(axes[1], env, open_path, "Open-loop plan (failure case)")
    axes[2].semilogy(losses, "C0-o", markersize=3)
    axes[2].set_title("World model training loss")
    axes[2].set_xlabel("epoch")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("Imagine futures in latent space, then act — replan or drift", fontsize=13)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
