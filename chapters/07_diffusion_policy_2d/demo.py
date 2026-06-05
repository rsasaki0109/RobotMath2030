#!/usr/bin/env python3
"""Chapter 07: Diffusion policy for 2D multimodal trajectories."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from miniworlds.two_path_world import (
    GOAL,
    START,
    collision_rate,
    generate_demonstrations,
    left_path,
    right_path,
)
from robotmath.diffusion import DiffusionPolicy2D, TrainConfig, predict_mean_regression, train_mean_regression
from robotmath.viz.plot_trajectory import draw_trajectories, draw_two_path_world


def main():
    print("Chapter 07 — Diffusion policy for 2D trajectories")
    horizon = 24
    demos, cond = generate_demonstrations(n_per_mode=48, horizon=horizon, seed=0)
    test_cond = np.concatenate([START, GOAL])[None, :]

    print("Training mean regression baseline...")
    mean_model = train_mean_regression(demos, cond, horizon=horizon, epochs=80, seed=0)
    mean_pred = predict_mean_regression(mean_model, test_cond, horizon=horizon)

    print("Training tiny diffusion policy...")
    cfg = TrainConfig(horizon=horizon, timesteps=20, epochs=120, seed=0)
    policy = DiffusionPolicy2D(cfg)
    losses = policy.fit(demos, cond, verbose=True)
    samples = policy.sample(test_cond, n_samples=24)

    mean_coll = collision_rate(mean_pred)
    sample_coll = collision_rate(samples)
    left_hits = sum(
        1 for s in samples if np.mean(s[:, 0] < 0.4) > 0.5 and not collision_rate(s)
    )
    right_hits = sum(
        1 for s in samples if np.mean(s[:, 0] > 0.6) > 0.5 and not collision_rate(s)
    )

    print(f"Mean regression collision: {mean_coll:.0%}")
    print(f"Diffusion sample collision: {sample_coll:.0%}")
    print(f"Diffusion modes (approx): left={left_hits}, right={right_hits}, total={len(samples)}")
    print(f"Final diffusion loss: {losses[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    draw_two_path_world(ax)
    draw_trajectories(ax, left_path(horizon), color="C0", alpha=0.25, label="left demos")
    draw_trajectories(ax, right_path(horizon), color="C0", alpha=0.25)
    draw_trajectories(ax, mean_pred, color="C3", linewidth=2.5, label="mean regression")
    ax.set_title("Unimodal regression averages modes → collision")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    draw_two_path_world(ax)
    draw_trajectories(ax, samples, color="C0", alpha=0.55, label="diffusion samples")
    ax.set_title("Diffusion policy preserves multimodality")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Why diffusion for robot trajectories?", fontsize=13)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
