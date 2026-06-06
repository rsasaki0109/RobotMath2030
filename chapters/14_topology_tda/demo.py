#!/usr/bin/env python3
"""Chapter 14: Topology / TDA — count loops in lidar-like clouds."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from miniworlds.loop_cloud_world import circle_cloud, corridor_cloud, shuffled_scan, two_loop_cloud
from robotmath.topology import naive_component_count, rips_persistence, topological_loop_count


def _plot_cloud(ax, points: np.ndarray, title: str) -> None:
    ax.scatter(points[:, 0], points[:, 1], s=18, c="C0", alpha=0.85)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def _plot_sweep(ax, diagram, title: str) -> None:
    ax.step(diagram.thresholds, diagram.beta1, where="post", color="C0")
    ax.axhline(int(np.median(diagram.beta1)), color="C3", linestyle="--", label=f"median β₁={int(np.median(diagram.beta1))}")
    ax.set_xlabel("Rips scale ε (local window)")
    ax.set_ylabel("β₁ (loop count proxy)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def main():
    print("Chapter 14 — Topology / TDA for loop detection")

    scenes = {
        "circle (1 loop)": shuffled_scan(circle_cloud(n=40, noise=0.015, seed=2), seed=9),
        "two loops (β₁=2)": shuffled_scan(two_loop_cloud(n_per_loop=20, noise=0.015, seed=3), seed=10),
        "corridor (0 loops)": shuffled_scan(corridor_cloud(n=45, seed=4), seed=11),
    }

    print(f"{'scene':<22} {'loops (TDA)':>12} {'components@ε':>14}")
    print("-" * 52)

    diagrams = {}
    for name, pts in scenes.items():
        diagram = rips_persistence(pts)
        diagrams[name] = diagram
        loops = topological_loop_count(pts)
        eps = float(diagram.local_scale * 1.5)
        components = naive_component_count(pts, epsilon=eps)
        print(f"{name:<22} {loops:>12} {components:>14}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for col, (name, pts) in enumerate(scenes.items()):
        _plot_cloud(axes[0, col], pts, name)
        _plot_sweep(axes[1, col], diagrams[name], "Local Rips β₁ sweep")

    fig.suptitle(
        "TDA: β₁ over a local Rips window counts loops; single-scale components do not",
        fontsize=13,
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
