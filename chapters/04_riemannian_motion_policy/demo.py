#!/usr/bin/env python3
"""Chapter 04: Riemannian motion policies in 2D."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from miniworlds.rmp_world import GOAL, RMP_START
from robotmath.motion import RMPConfig, rollout
from robotmath.viz.plot_trajectory import draw_trajectories, draw_two_path_world


def main():
    print("Chapter 04 — Riemannian motion policy (2D)")
    cfg = RMPConfig()

    traj_rmp, reached_rmp, coll_rmp = rollout(RMP_START, GOAL, policy="rmp", cfg=cfg)
    traj_naive, reached_naive, coll_naive = rollout(RMP_START, GOAL, policy="naive", cfg=cfg)

    print(f"RMP rollout:   reached={reached_rmp}, collision={coll_rmp}, steps={len(traj_rmp) - 1}")
    print(f"Naive rollout: reached={reached_naive}, collision={coll_naive}, steps={len(traj_naive) - 1}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    draw_two_path_world(ax)
    draw_trajectories(ax, traj_naive, color="C3", linewidth=2.0, label="naive sum of forces")
    ax.set_title("Naive APF-style sum → collision / stall")

    ax = axes[1]
    draw_two_path_world(ax)
    draw_trajectories(ax, traj_rmp, color="C0", linewidth=2.0, label="RMP metric fusion")
    ax.set_title("RMP: a = (ΣM)⁻¹ Σf → smooth detour")

    fig.suptitle("Goal + obstacle tasks combined on a 2D point robot", fontsize=13)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
