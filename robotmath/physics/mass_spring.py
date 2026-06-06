"""Damped mass-spring — smooth differentiable dynamics (mostly)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass
class MassSpringParams:
    mass: float = 1.0
    spring_k: float = 8.0
    damping: float = 0.5
    dt: float = 0.05


def simulate(
    params: MassSpringParams,
    x0: float,
    v0: float,
    steps: int,
) -> Array:
    """Semi-implicit Euler. Returns (steps+1, 2) array of [position, velocity]."""
    traj = np.zeros((steps + 1, 2), dtype=float)
    traj[0] = [x0, v0]
    m, k, c, dt = params.mass, params.spring_k, params.damping, params.dt
    for t in range(steps):
        x, v = traj[t]
        a = -(k / m) * x - (c / m) * v
        v_next = v + dt * a
        x_next = x + dt * v_next
        traj[t + 1] = [x_next, v_next]
    return traj


def trajectory_mse(
    params: MassSpringParams,
    target: Array,
    x0: float,
    v0: float,
) -> float:
    steps = target.shape[0] - 1
    pred = simulate(params, x0, v0, steps)
    return float(np.mean((pred - target) ** 2))


def identify_finite_diff(
    target: Array,
    x0: float,
    v0: float,
    init_k: float = 2.0,
    init_c: float = 0.1,
    lr: float = 0.5,
    steps: int = 80,
    eps: float = 1e-4,
) -> tuple[MassSpringParams, list[float]]:
    """Gradient descent with central-difference gradients."""
    k, c = float(init_k), float(init_c)
    dt = 0.05
    losses: list[float] = []
    for _ in range(steps):
        base = MassSpringParams(spring_k=k, damping=c, dt=dt)
        loss = trajectory_mse(base, target, x0, v0)
        losses.append(loss)
        lk = trajectory_mse(MassSpringParams(spring_k=k + eps, damping=c, dt=dt), target, x0, v0)
        lm = trajectory_mse(MassSpringParams(spring_k=k - eps, damping=c, dt=dt), target, x0, v0)
        lc = trajectory_mse(MassSpringParams(spring_k=k, damping=c + eps, dt=dt), target, x0, v0)
        lcm = trajectory_mse(MassSpringParams(spring_k=k, damping=c - eps, dt=dt), target, x0, v0)
        dk = (lk - lm) / (2 * eps)
        dc = (lc - lcm) / (2 * eps)
        k -= lr * dk
        c = max(0.01, c - lr * dc)
    return MassSpringParams(spring_k=k, damping=c, dt=dt), losses


def identify_autograd(
    target: Array,
    x0: float,
    v0: float,
    init_k: float = 2.0,
    init_c: float = 0.1,
    lr: float = 0.15,
    steps: int = 120,
) -> tuple[MassSpringParams, list[float]]:
    """Gradient descent through a differentiable torch rollout."""
    import torch

    target_t = torch.tensor(target, dtype=torch.float32)
    log_k = torch.tensor([np.log(init_k)], dtype=torch.float32, requires_grad=True)
    log_c = torch.tensor([np.log(init_c)], dtype=torch.float32, requires_grad=True)
    dt = 0.05
    opt = torch.optim.Adam([log_k, log_c], lr=lr)
    losses: list[float] = []

    for _ in range(steps):
        opt.zero_grad()
        k = torch.exp(log_k)
        c = torch.exp(log_c)
        x = torch.tensor([x0], dtype=torch.float32)
        v = torch.tensor([v0], dtype=torch.float32)
        preds = [torch.stack([x.squeeze(), v.squeeze()])]
        for _ in range(target.shape[0] - 1):
            a = -k * x - c * v
            v = v + dt * a
            x = x + dt * v
            preds.append(torch.stack([x.squeeze(), v.squeeze()]))
        pred = torch.stack(preds)
        loss = torch.mean((pred - target_t) ** 2)
        losses.append(float(loss.item()))
        loss.backward()
        opt.step()

    k = float(torch.exp(log_k).item())
    c = float(torch.exp(log_c).item())
    return MassSpringParams(spring_k=k, damping=c, dt=dt), losses
