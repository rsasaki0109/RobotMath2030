#!/usr/bin/env python3
"""Chapter 02: Tiny Lie graph optimizer — pose graph SLAM demo."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from miniworlds.pose_graph_world import square_loop_graph
from robotmath.lie import se2
from robotmath.viz import draw_pose_graph, trajectory_from_poses


def pose_error(poses: list, gt: list) -> float:
    err = 0.0
    for T, G in zip(poses, gt):
        r = se2.log(se2.between(T, G))
        err += float(r @ r)
    return err


def main():
    print("Chapter 02 — Tiny Lie graph optimizer")
    graph, gt = square_loop_graph(seed=42)
    edges_idx = [(e.i, e.j) for e in graph.edges]

    graph_lie = graph.copy()
    graph_euc = graph.copy()

    opt_lie, cost_lie = graph_lie.optimize(use_lie=True, max_iters=40)
    opt_euc, cost_euc = graph_euc.optimize(use_lie=False, max_iters=40)

    err_init = pose_error(graph.poses, gt)
    err_lie = pose_error(opt_lie.poses, gt)
    err_euc = pose_error(opt_euc.poses, gt)

    print(f"Pose error (sum sq log-residual): init={err_init:.4f}, lie={err_lie:.4f}, euclidean={err_euc:.4f}")
    print(f"Final cost: lie={cost_lie[-1]:.6f}, euclidean={cost_euc[-1]:.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, poses, title in [
        (axes[0], graph.poses, "Before optimization"),
        (axes[1], opt_lie.poses, "Lie residual"),
        (axes[2], opt_euc.poses, "Euclidean residual"),
    ]:
        draw_pose_graph(ax, poses, edges=edges_idx)
        gx, gy = trajectory_from_poses(gt)
        ax.plot(gx, gy, "g-", alpha=0.4, linewidth=2, label="ground truth")
        ax.set_title(title)
        ax.legend(loc="upper right")

    fig.suptitle("Pose graph loop closure: Lie vs Euclidean residuals", fontsize=13)
    fig.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.semilogy(cost_lie, "C0-o", markersize=4, label="Lie")
    ax2.semilogy(cost_euc, "C3--s", markersize=4, label="Euclidean")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("cost")
    ax2.set_title("Gauss-Newton convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
