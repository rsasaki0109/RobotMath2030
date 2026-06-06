#!/usr/bin/env python3
"""Chapter 08: Flow matching vs diffusion on the same 2D multimodal task."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from miniworlds.two_path_world import GOAL, START, collision_rate, generate_demonstrations
from robotmath.diffusion import DiffusionPolicy2D, FlowMatchingPolicy2D, FlowTrainConfig, TrainConfig
from robotmath.viz.plot_trajectory import draw_trajectories, draw_two_path_world


def _mode_counts(samples: np.ndarray) -> tuple[int, int]:
    left = right = 0
    for s in samples:
        if collision_rate(s):
            continue
        if np.mean(s[:, 0] < 0.4) > 0.5:
            left += 1
        elif np.mean(s[:, 0] > 0.6) > 0.5:
            right += 1
    return left, right


def main():
    print("Chapter 08 — Flow matching vs diffusion")
    horizon = 24
    demos, cond = generate_demonstrations(n_per_mode=64, horizon=horizon, seed=0)
    test_cond = np.concatenate([START, GOAL])[None, :]
    n_samples = 24

    print("Training diffusion policy...")
    diff_cfg = TrainConfig(horizon=horizon, timesteps=20, epochs=120, seed=0)
    diffusion = DiffusionPolicy2D(diff_cfg)
    diff_losses = diffusion.fit(demos, cond, verbose=True)

    print("Training flow matching policy...")
    flow_cfg = FlowTrainConfig(horizon=horizon, epochs=200, sample_steps=20, seed=0)
    flow = FlowMatchingPolicy2D(flow_cfg)
    flow_losses = flow.fit(demos, cond, verbose=True)

    t0 = time.perf_counter()
    diff_samples = diffusion.sample(test_cond, n_samples=n_samples)
    diff_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    flow_samples = flow.sample(test_cond, n_samples=n_samples, steps=20, integrator="heun")
    flow_time = time.perf_counter() - t0

    diff_coll = collision_rate(diff_samples)
    flow_coll = collision_rate(flow_samples)
    diff_left, diff_right = _mode_counts(diff_samples)
    flow_left, flow_right = _mode_counts(flow_samples)

    print(f"Diffusion  collision: {diff_coll:.0%}  modes left/right: {diff_left}/{diff_right}")
    print(f"Flow match collision: {flow_coll:.0%}  modes left/right: {flow_left}/{flow_right}")
    print(f"Final diffusion loss: {diff_losses[-1]:.4f}")
    print(f"Final flow loss:      {flow_losses[-1]:.4f}")
    print(f"Sample time — diffusion ({diff_cfg.timesteps} steps): {diff_time:.3f}s")
    print(f"Sample time — flow match ({flow_cfg.sample_steps} steps): {flow_time:.3f}s")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    draw_two_path_world(ax)
    draw_trajectories(ax, diff_samples, color="C0", alpha=0.55, label="diffusion")
    ax.set_title(f"Diffusion ({diff_cfg.timesteps} denoise steps)")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    draw_two_path_world(ax)
    draw_trajectories(ax, flow_samples, color="C2", alpha=0.55, label="flow match")
    ax.set_title(f"Flow matching ({flow_cfg.sample_steps} Heun steps)")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    ax.plot(diff_losses, label="diffusion (ε loss)", color="C0")
    ax.plot(flow_losses, label="flow (velocity loss)", color="C2")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("Same data, different objective")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Flow matching vs diffusion on multimodal trajectories", fontsize=13)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
