#!/usr/bin/env python3
"""Chapter 09: Differentiable physics — system ID and contact failure."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from miniworlds.mass_spring_world import bouncing_ball_rollout, noisy_mass_spring_rollout
from robotmath.physics import (
    BounceParams,
    MassSpringParams,
    identify_autograd,
    identify_finite_diff,
    simulate,
    simulate_bounce_hard,
    simulate_bounce_soft,
)


def _contact_gradient_probe(true_g: float = 9.81, y0: float = 1.5, steps: int = 120):
    """Autograd gradient of bounce loss w.r.t. gravity — noisy at hard contact."""
    import torch

    obs, _ = bouncing_ball_rollout(gravity=true_g, y0=y0, steps=steps, noise=0.0, seed=0)
    obs_y = torch.tensor(obs[:, 0], dtype=torch.float32)
    guesses = np.linspace(5.0, 14.0, 19)
    grads = []
    for g in guesses:
        gravity = torch.tensor([g], dtype=torch.float32, requires_grad=True)
        y = torch.tensor([y0], dtype=torch.float32)
        vy = torch.tensor([0.0], dtype=torch.float32)
        dt = 0.02
        e = 0.85
        heights = []
        for _ in range(obs_y.shape[0] - 1):
            vy = vy - gravity * dt
            y = y + vy * dt
            y = torch.clamp(y, min=0.0)
            vy = torch.where(y <= 0.0, -e * vy, vy)
            heights.append(y)
        pred = torch.stack(heights)
        loss = torch.mean((pred - obs_y[1:]) ** 2)
        loss.backward()
        grads.append(float(gravity.grad.item()))
    return guesses, np.array(grads)


def main():
    print("Chapter 09 — Differentiable physics")
    obs, true_params = noisy_mass_spring_rollout(true_k=8.0, true_c=0.5, seed=0)
    x0, v0 = obs[0, 0], obs[0, 1]

    fd_params, fd_losses = identify_finite_diff(obs, x0, v0, init_k=2.0, init_c=0.1, steps=80)
    ag_params, ag_losses = identify_autograd(obs, x0, v0, init_k=2.0, init_c=0.1, steps=120)

    print("True mass-spring:", f"k={true_params.spring_k:.2f}", f"c={true_params.damping:.2f}")
    print("Finite diff:     ", f"k={fd_params.spring_k:.2f}", f"c={fd_params.damping:.2f}")
    print("Autograd:        ", f"k={ag_params.spring_k:.2f}", f"c={ag_params.damping:.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    t = np.arange(obs.shape[0]) * true_params.dt
    axes[0, 0].plot(t, obs[:, 0], "k.", markersize=3, label="observed x")
    for label, params, color in [
        ("finite diff", fd_params, "C0"),
        ("autograd", ag_params, "C2"),
        ("true", true_params, "C1"),
    ]:
        pred = simulate(params, x0, v0, obs.shape[0] - 1)
        axes[0, 0].plot(t, pred[:, 0], color=color, linewidth=1.8, label=label)
    axes[0, 0].set_title("Mass-spring system ID")
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(fd_losses, label="finite diff")
    axes[0, 1].semilogy(ag_losses, label="autograd")
    axes[0, 1].set_title("Identification loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    ball_obs, ball_true = bouncing_ball_rollout(gravity=9.81, steps=120, seed=1)
    tb = np.arange(ball_obs.shape[0]) * ball_true.dt
    hard = simulate_bounce_hard(ball_true, ball_obs[0, 0], ball_obs[0, 1], ball_obs.shape[0] - 1)
    soft = simulate_bounce_soft(ball_true, ball_obs[0, 0], ball_obs[0, 1], ball_obs.shape[0] - 1)
    axes[1, 0].plot(tb, ball_obs[:, 0], "k.", markersize=3, label="obs height")
    axes[1, 0].plot(tb, hard[:, 0], "C3-", label="hard contact")
    axes[1, 0].plot(tb, soft[:, 0], "C0--", label="soft contact")
    axes[1, 0].set_title("Bouncing ball: hard vs soft contact model")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    guesses, grads = _contact_gradient_probe()
    axes[1, 1].plot(guesses, grads, "C3-o", markersize=4)
    axes[1, 1].axvline(9.81, color="gray", linestyle=":", label="true g")
    axes[1, 1].set_title("Autograd ∂loss/∂g — jagged under hard contact")
    axes[1, 1].set_xlabel("gravity guess")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Smooth dynamics are easy to differentiate; contact is not", fontsize=13)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
