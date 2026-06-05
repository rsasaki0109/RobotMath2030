#!/usr/bin/env python3
"""Chapter 05: Sinkhorn optimal transport for point cloud correspondence."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from miniworlds.point_cloud_world import misaligned_pair
from robotmath.optimal_transport import pairwise_sq_distances, sinkhorn, sinkhorn2
from robotmath.viz import draw_point_clouds, draw_transport_plan, plot_sinkhorn_panel


def naive_index_matching_cost(source: np.ndarray, target: np.ndarray) -> float:
    """Wrong when clouds are reordered — kept as a deliberate failure case."""
    n = min(source.shape[0], target.shape[0])
    return float(np.mean(np.sum((source[:n] - target[:n]) ** 2, axis=1)))


def nearest_neighbor_cost(source: np.ndarray, target: np.ndarray) -> float:
    C = pairwise_sq_distances(source, target)
    return float(np.mean(np.min(C, axis=1)))


def main():
    print("Chapter 05 — Sinkhorn for point clouds and maps")
    source, target = misaligned_pair(seed=7, theta=0.4, tx=0.2, ty=-0.1, noise=0.025)
    # Shuffle target: same set of points, unknown correspondence (typical for scans)
    rng = np.random.default_rng(11)
    target = target[rng.permutation(target.shape[0])]

    w2_cost, plan, gaps = sinkhorn2(source, target, epsilon=0.03, max_iter=300)
    idx_cost = naive_index_matching_cost(source, target)
    nn_cost = nearest_neighbor_cost(source, target)

    print(f"Sinkhorn transport cost (W2^2 approx): {w2_cost:.5f}")
    print(f"Naive index L2 (wrong):                {idx_cost:.5f}")
    print(f"Nearest-neighbor mean sq dist:         {nn_cost:.5f}")
    print(f"Sinkhorn marginal gap (final):         {gaps[-1]:.2e}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    draw_point_clouds(axes[0], source, target)
    axes[0].set_title("Misaligned L-scan pair")
    axes[0].legend(loc="upper right")

    draw_transport_plan(axes[1], source, target, plan, threshold=0.0015, max_lines=60)
    draw_point_clouds(axes[1], source, target)
    axes[1].set_title("Sinkhorn soft correspondence")
    axes[1].legend(loc="upper right")

    axes[2].semilogy(gaps, "C0-o", markersize=3)
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("marginal error")
    axes[2].set_title("Sinkhorn convergence")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Why OT for robot maps: unordered points need soft matching", fontsize=13)
    fig.tight_layout()

    fig2, ax = plt.subplots(figsize=(5, 5))
    plot_sinkhorn_panel(ax, source, target, plan, title="Entropic transport plan", threshold=0.0015)
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
