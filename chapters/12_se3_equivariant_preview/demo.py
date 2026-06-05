#!/usr/bin/env python3
"""Chapter 12: SE(3)-equivariant point cloud readout preview."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from miniworlds.point_cloud_3d_world import make_dataset
from robotmath.equivariance import (
    EquivariantTrainConfig,
    EquivariantVectorReadout,
    NaivePointMLP,
    eval_canonical_error,
    eval_rotated_error,
    random_rotation,
    train_vector_predictor,
)


def main():
    print("Chapter 12 — SE(3)-equivariant policy preview")
    points, targets = make_dataset(n_clouds=256, seed=0)
    n_pts = points.shape[1]
    cfg = EquivariantTrainConfig(n_points=n_pts, epochs=120, seed=0)
    R_holdout = random_rotation(999)

    naive = NaivePointMLP(n_pts, hidden=cfg.hidden)
    equi = EquivariantVectorReadout(hidden=32)

    print("Training naive MLP on canonical clouds...")
    naive_losses = train_vector_predictor(naive, points, targets, cfg)
    print("Training equivariant vector readout...")
    equi_losses = train_vector_predictor(equi, points, targets, cfg)

    test_pts = points[:64]
    test_tgt = targets[:64]

    naive_can = eval_canonical_error(naive, test_pts, test_tgt)
    naive_rot = eval_rotated_error(naive, test_pts, test_tgt, R_holdout)
    equi_can = eval_canonical_error(equi, test_pts, test_tgt)
    equi_rot = eval_rotated_error(equi, test_pts, test_tgt, R_holdout)

    print(f"Naive MSE — canonical frame: {naive_can:.5f}")
    print(f"Naive MSE — rotated frame:    {naive_rot:.5f}  (×{naive_rot/max(naive_can,1e-8):.1f})")
    print(f"Equiv MSE — canonical frame:  {equi_can:.5f}")
    print(f"Equiv MSE — rotated frame:    {equi_rot:.5f}  (×{equi_rot/max(equi_can,1e-8):.1f})")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.semilogy(naive_losses, "C3--", label="naive MLP")
    ax.semilogy(equi_losses, "C0-", label="equivariant readout")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train MSE")
    ax.set_title("Same data, different inductive bias")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels = ["naive\ncanonical", "naive\nrotated", "equiv\ncanonical", "equiv\nrotated"]
    vals = [naive_can, naive_rot, equi_can, equi_rot]
    colors = ["C3", "C3", "C0", "C0"]
    ax.bar(labels, vals, color=colors, alpha=0.75)
    ax.set_ylabel("holdout MSE")
    ax.set_title("Rotation should not change an equivariant map")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Why manipulation papers use SE(3)-equivariant networks", fontsize=13)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
