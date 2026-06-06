#!/usr/bin/env python3
"""Chapter 06: Wasserstein distance for occupancy map evaluation."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from miniworlds.occupancy_map_world import (
    l2_grid_mse,
    map_pair_drift,
    map_pair_ghost,
    map_pair_identical,
    map_to_points,
)
from robotmath.optimal_transport import compare_map_metrics, wasserstein_between_maps
from robotmath.viz.plot_map import draw_map_pair, draw_map_transport, draw_occupancy_map


def evaluate_pair(map_a, map_b, label: str) -> dict[str, float]:
    metrics = compare_map_metrics(map_a, map_b, map_to_points, l2_grid_mse, epsilon=0.01)
    print(
        f"{label:22s}  L2 grid MSE={metrics['l2_grid_mse']:.5f}"
        f"  W2 approx={metrics['wasserstein2']:.5f}"
    )
    return metrics


def main():
    print("Chapter 06 — Wasserstein map evaluation")
    pairs = [map_pair_identical(), map_pair_drift(), map_pair_ghost()]
    results = [evaluate_pair(a, b, label) for a, b, label in pairs]

    drift_a, drift_b, _ = map_pair_drift()
    w2, plan, pts_a, pts_b = wasserstein_between_maps(
        drift_a, drift_b, map_to_points, epsilon=0.01, max_iter=500,
    )
    print(f"\nDrift pair W2={w2:.5f} with {pts_a.shape[0]} vs {pts_b.shape[0]} occupied cells")

    # Failure case highlight: drift W2 >> L2 MSE (sparse grid dilution)
    drift_metrics = results[1]
    ident_metrics = results[0]
    ghost_metrics = results[2]
    print(
        "\nKey insight:"
        f" ghost L2 ({ghost_metrics['l2_grid_mse']:.5f})"
        f" < drift L2 ({drift_metrics['l2_grid_mse']:.5f})"
        " — sparse grid makes the ghost map look *more* similar in L2."
    )
    print(
        "W2 above identical baseline:"
        f" drift +{drift_metrics['wasserstein2'] - ident_metrics['wasserstein2']:.5f},"
        f" ghost +{ghost_metrics['wasserstein2'] - ident_metrics['wasserstein2']:.5f}"
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ref, shifted, drift_label = map_pair_drift()
    draw_map_pair(axes[0, 0], ref, shifted, title=f"Drift pair — {drift_label}")
    draw_map_transport(axes[0, 1], pts_a, pts_b, plan)

    ref_g, ghost, ghost_label = map_pair_ghost()
    draw_occupancy_map(axes[1, 0], ref_g, title="Reference map")
    draw_occupancy_map(axes[1, 1], ghost, title=f"With ghost obstacle — {ghost_label}")

    fig.suptitle("Map evaluation: L2 dilutes sparse errors, Wasserstein tracks mass", fontsize=13)
    fig.tight_layout()

    fig2, ax = plt.subplots(figsize=(7, 4))
    labels = ["identical", "drift", "ghost"]
    l2_vals = [r["l2_grid_mse"] for r in results]
    w2_vals = [r["wasserstein2"] for r in results]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, l2_vals, width, label="L2 grid MSE", color="C3")
    ax.bar(x + width / 2, w2_vals, width, label="W2 approx", color="C0")
    ax.set_xticks(x, labels)
    ax.set_ylabel("distance")
    ax.set_title("Metric comparison across map pairs")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
