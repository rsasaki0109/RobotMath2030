#!/usr/bin/env python3
"""Chapter 13: Neural operators — learn a mass-spring rollout map."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from robotmath.neural_operators import (
    OperatorTrainConfig,
    make_operator_dataset,
    operator_trajectory_mse,
    train_deeponet,
)
from robotmath.physics.mass_spring import MassSpringParams, simulate


def main():
    print("Chapter 13 — Neural operators (DeepONet preview)")
    train_u, times, train_y = make_operator_dataset(n_samples=512, seed=0)
    test_u, _, test_y = make_operator_dataset(n_samples=64, seed=1)

    cfg = OperatorTrainConfig(epochs=150, seed=0)
    print("Training tiny DeepONet operator...")
    model, losses = train_deeponet(train_u, times, train_y, cfg)

    train_mse = operator_trajectory_mse(model, train_u[:32], times, train_y[:32])
    test_mse = operator_trajectory_mse(model, test_u, times, test_y)
    print(f"Operator MSE — train subset: {train_mse:.6f}, test: {test_mse:.6f}")

    params = MassSpringParams()
    steps = len(times) - 1
    n_query = 500

    t0 = time.perf_counter()
    for u in test_u[:n_query]:
        simulate(params, float(u[0]), float(u[1]), steps)
    sim_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.predict_trajectory(test_u[:n_query], times)
    op_time = time.perf_counter() - t0

    print(f"Rollout {n_query} trajectories — simulator: {sim_time:.3f}s, operator: {op_time:.3f}s")

    u_demo = test_u[0]
    y_demo = test_y[0]
    y_pred = model.predict_trajectory(u_demo[None, :], times)[0]
    y_sim = simulate(params, float(u_demo[0]), float(u_demo[1]), steps)[:, 0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.semilogy(losses, "C0-")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train MSE")
    ax.set_title("DeepONet operator training")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times, y_demo, "k-", label="ground truth")
    ax.plot(times, y_sim, "C3--", label="integrator")
    ax.plot(times, y_pred, "C0-o", markersize=3, label="operator")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position")
    ax.set_title(f"Held-out IC x0={u_demo[0]:.2f}, v0={u_demo[1]:.2f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.bar(["simulator", "operator"], [sim_time, op_time], color=["C3", "C0"], alpha=0.8)
    ax.set_ylabel("seconds")
    ax.set_title(f"Batch inference ({n_query} trajectories)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Neural operators: amortized rollout vs numerical integration", fontsize=13)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
