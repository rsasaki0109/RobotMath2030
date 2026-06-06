"""Tiny DeepONet-style operator for damped mass-spring trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotmath.physics.mass_spring import MassSpringParams, simulate

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required. Install with: pip install -e '.[torch]'")


if torch is not None:

    class DeepONet1D(nn.Module):
        """
        Maps (initial state, time) -> position.

        G[u](t) ≈ branch(u) · trunk(t)
        """

        def __init__(self, branch_dim: int = 2, hidden: int = 32, width: int = 16):
            super().__init__()
            self.branch = nn.Sequential(
                nn.Linear(branch_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, width),
            )
            self.trunk = nn.Sequential(
                nn.Linear(1, hidden),
                nn.Tanh(),
                nn.Linear(hidden, width),
            )

        def forward(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            b = self.branch(u)
            phi = self.trunk(t)
            return torch.sum(b * phi, dim=-1, keepdim=True)

        def predict_trajectory(
            self,
            u: np.ndarray,
            times: np.ndarray,
        ) -> np.ndarray:
            self.eval()
            u_t = torch.tensor(u, dtype=torch.float32)
            if u_t.ndim == 1:
                u_t = u_t.unsqueeze(0)
            outs: list[np.ndarray] = []
            with torch.no_grad():
                for i in range(u_t.shape[0]):
                    t_t = torch.tensor(times[:, None], dtype=torch.float32)
                    u_rep = u_t[i : i + 1].repeat(t_t.shape[0], 1)
                    y = self.forward(u_rep, t_t).squeeze(-1).numpy()
                    outs.append(y)
            return np.stack(outs, axis=0)

else:  # pragma: no cover
    DeepONet1D = object  # type: ignore[misc, assignment]


@dataclass
class OperatorTrainConfig:
    hidden: int = 32
    width: int = 16
    lr: float = 2e-3
    epochs: int = 150
    batch_size: int = 64
    seed: int = 0


def rollout_positions(
    u: np.ndarray,
    params: MassSpringParams | None = None,
    steps: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Return times and position trajectory for initial state u = [x0, v0]."""
    params = params or MassSpringParams()
    traj = simulate(params, float(u[0]), float(u[1]), steps)
    times = np.arange(steps + 1, dtype=float) * params.dt
    return times, traj[:, 0]


def make_operator_dataset(
    n_samples: int = 512,
    steps: int = 40,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (initial_states, times, positions).

    initial_states: (N, 2), times: (T,), positions: (N, T)
    """
    rng = np.random.default_rng(seed)
    params = MassSpringParams()
    u_list: list[np.ndarray] = []
    pos_list: list[np.ndarray] = []
    times, _ = rollout_positions(np.array([0.0, 0.0]), params, steps)
    for _ in range(n_samples):
        u = np.array([rng.uniform(-1.5, 1.5), rng.uniform(-1.0, 1.0)])
        _, pos = rollout_positions(u, params, steps)
        u_list.append(u)
        pos_list.append(pos)
    return np.stack(u_list), times, np.stack(pos_list)


def train_deeponet(
    initial_states: np.ndarray,
    times: np.ndarray,
    positions: np.ndarray,
    cfg: OperatorTrainConfig | None = None,
) -> tuple["DeepONet1D", list[float]]:
    _require_torch()
    cfg = cfg or OperatorTrainConfig()
    torch.manual_seed(cfg.seed)
    model = DeepONet1D(hidden=cfg.hidden, width=cfg.width)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    u_all = torch.tensor(initial_states, dtype=torch.float32)
    t_grid = torch.tensor(times, dtype=torch.float32)
    y_all = torch.tensor(positions, dtype=torch.float32)
    n, t_len = y_all.shape
    losses: list[float] = []

    model.train()
    for _ in range(cfg.epochs):
        idx = torch.randperm(n)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, n, cfg.batch_size):
            b = idx[start : start + cfg.batch_size]
            u_b = u_all[b]
            y_b = y_all[b]
            batch_loss = 0.0
            count = 0
            for j in range(t_len):
                t_j = t_grid[j].view(1, 1).expand(u_b.shape[0], 1)
                pred = model(u_b, t_j).squeeze(-1)
                batch_loss = batch_loss + nn.functional.mse_loss(pred, y_b[:, j])
                count += 1
            batch_loss = batch_loss / count
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            epoch_loss += float(batch_loss.item())
            steps += 1
        losses.append(epoch_loss / max(steps, 1))
    model.eval()
    return model, losses


def operator_trajectory_mse(
    model: "DeepONet1D",
    initial_states: np.ndarray,
    times: np.ndarray,
    positions: np.ndarray,
) -> float:
    pred = model.predict_trajectory(initial_states, times)
    return float(np.mean((pred - positions) ** 2))


def simulate_batch_mse(
    initial_states: np.ndarray,
    times: np.ndarray,
    positions: np.ndarray,
    params: MassSpringParams | None = None,
) -> float:
    params = params or MassSpringParams()
    steps = len(times) - 1
    errs = []
    for u, target in zip(initial_states, positions):
        pred = simulate(params, float(u[0]), float(u[1]), steps)[:, 0]
        errs.append(float(np.mean((pred - target) ** 2)))
    return float(np.mean(errs))
