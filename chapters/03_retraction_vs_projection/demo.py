#!/usr/bin/env python3
"""Chapter 03: Retraction vs projection on SE(2)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from miniworlds.landmark_pose_world import landmark_pose_problem
from robotmath.lie import se2
from robotmath.optimization.manifold_update import optimize_pose_landmarks, pose_error
from robotmath.viz import draw_pose


def main():
    print("Chapter 03 — Retraction vs projection")
    T_true, T_init, body, world = landmark_pose_problem(seed=0)

    T_ret, cost_ret, hist_ret = optimize_pose_landmarks(
        T_init, body, world, method="retraction", max_iters=120, step=0.1
    )
    T_proj, cost_proj, hist_proj = optimize_pose_landmarks(
        T_init, body, world, method="projection", max_iters=120, step=0.1
    )

    err_init = pose_error(T_init, T_true)
    err_ret = pose_error(T_ret, T_true)
    err_proj = pose_error(T_proj, T_true)

    print(f"Pose error (sum sq log-residual): init={err_init:.4f}")
    print(f"  retraction={err_ret:.4f}, projection={err_proj:.4f}")
    print(f"Final landmark cost: retraction={cost_ret[-1]:.6f}, projection={cost_proj[-1]:.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, T, title in [
        (axes[0], T_init, "Initial guess (large heading error)"),
        (axes[1], T_ret, "Retraction: T ← T exp(−αξ)"),
        (axes[2], T_proj, "Projection: add to (x, y, θ)"),
    ]:
        ax.scatter(world[:, 0], world[:, 1], s=60, c="C0", label="landmarks")
        draw_pose(ax, T_true, color="green", scale=0.45, label="true")
        draw_pose(ax, T, color="C3", scale=0.45, label="estimate")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Landmark pose estimation on SE(2)", fontsize=13)
    fig.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.semilogy(cost_ret, "C0-o", markersize=4, label="retraction")
    ax2.semilogy(cost_proj, "C3--s", markersize=4, label="projection")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("landmark cost")
    ax2.set_title("Gradient descent convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    thetas_ret = [np.rad2deg(se2.to_xytheta(T)[2]) for T in hist_ret]
    thetas_proj = [np.rad2deg(se2.to_xytheta(T)[2]) for T in hist_proj]
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(thetas_ret, "C0-", label="retraction heading")
    ax3.plot(thetas_proj, "C3--", label="projection heading")
    ax3.axhline(np.rad2deg(se2.to_xytheta(T_true)[2]), color="green", linestyle=":", label="true heading")
    ax3.set_xlabel("iteration")
    ax3.set_ylabel("heading [deg]")
    ax3.set_title("Why projection struggles: tangent step ≠ Euler increment")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
