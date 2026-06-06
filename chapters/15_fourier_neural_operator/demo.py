#!/usr/bin/env python3
"""Chapter 15: Fourier Neural Operator vs DeepONet on mass-spring rollouts."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from robotmath.neural_operators import OperatorTrainConfig, make_operator_dataset, operator_trajectory_mse, train_deeponet
from robotmath.neural_operators.fno_mass_spring import fno_trajectory_mse, train_fno
from robotmath.physics.mass_spring import MassSpringParams, simulate


def main():
    print("Chapter 15 — Fourier Neural Operator vs DeepONet")
    train_u, times, train_y = make_operator_dataset(n_samples=512, seed=0)
    test_u, _, test_y = make_operator_dataset(n_samples=64, seed=1)
    cfg = OperatorTrainConfig(epochs=120, seed=0)

    print("Training DeepONet...")
    deeponet, deeponet_losses = train_deeponet(train_u, times, train_y, cfg)
    print("Training FNO...")
    fno, fno_losses = train_fno(train_u, times, train_y, cfg)

    deeponet_mse = operator_trajectory_mse(deeponet, test_u, times, test_y)
    fno_mse = fno_trajectory_mse(fno, test_u, times, test_y)
    print(f"Test MSE — DeepONet: {deeponet_mse:.6f}, FNO: {fno_mse:.6f}")

    params = MassSpringParams()
    steps = len(times) - 1
    n_query = 400

    t0 = time.perf_counter()
    deeponet.predict_trajectory(test_u[:n_query], times)
    deeponet_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    fno.predict_trajectory(test_u[:n_query], times)
    fno_t = time.perf_counter() - t0

    u_demo = test_u[0]
    y_true = test_y[0]
    y_deep = deeponet.predict_trajectory(u_demo[None, :], times)[0]
    y_fno = fno.predict_trajectory(u_demo[None, :], times)[0]
    y_sim = simulate(params, float(u_demo[0]), float(u_demo[1]), steps)[:, 0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.semilogy(deeponet_losses, "C0-", label="DeepONet")
    ax.semilogy(fno_losses, "C2-", label="FNO")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train MSE")
    ax.set_title("Operator training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times, y_true, "k-", label="ground truth")
    ax.plot(times, y_sim, "C3--", label="integrator", alpha=0.7)
    ax.plot(times, y_deep, "C0-o", markersize=3, label="DeepONet")
    ax.plot(times, y_fno, "C2-s", markersize=3, label="FNO")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position")
    ax.set_title(f"Held-out IC x0={u_demo[0]:.2f}, v0={u_demo[1]:.2f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.bar(["DeepONet", "FNO"], [deeponet_t, fno_t], color=["C0", "C2"], alpha=0.85)
    ax.set_ylabel("seconds")
    ax.set_title(f"Batch inference ({n_query} trajectories)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("FNO: spectral mixing on time grid vs DeepONet branch–trunk", fontsize=13)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
