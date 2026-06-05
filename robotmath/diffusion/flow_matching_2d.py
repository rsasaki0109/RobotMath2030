"""Tiny 2D conditional flow matching — compare with diffusion on the same world."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

    class VelocityPredictor(nn.Module):
        """Predicts flow velocity v(x_t, t, cond) along a straight noise→data path."""

        def __init__(self, traj_dim: int, cond_dim: int, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(traj_dim + cond_dim + 1, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, traj_dim),
            )

        def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            t_norm = t.float().unsqueeze(-1)
            return self.net(torch.cat([x_t, cond, t_norm], dim=-1))

else:  # pragma: no cover
    VelocityPredictor = object  # type: ignore[misc, assignment]


@dataclass
class FlowTrainConfig:
    horizon: int = 24
    hidden: int = 64
    lr: float = 2e-3
    epochs: int = 120
    batch_size: int = 64
    sample_steps: int = 10
    seed: int = 0


class FlowMatchingPolicy2D:
    """Conditional flow matching with a linear Gaussian probability path."""

    def __init__(self, cfg: FlowTrainConfig | None = None):
        _require_torch()
        self.cfg = cfg or FlowTrainConfig()
        self.traj_dim = self.cfg.horizon * 2
        self.cond_dim = 4
        self.model = VelocityPredictor(
            self.traj_dim, self.cond_dim, hidden=self.cfg.hidden
        )

    @staticmethod
    def flatten(traj: torch.Tensor) -> torch.Tensor:
        return traj.reshape(traj.shape[0], -1)

    @staticmethod
    def unflatten(x: torch.Tensor, horizon: int) -> torch.Tensor:
        return x.reshape(x.shape[0], horizon, 2)

    def loss(self, traj: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x1 = self.flatten(traj)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        t_view = t.unsqueeze(-1)
        x_t = (1.0 - t_view) * x0 + t_view * x1
        target = x1 - x0
        pred = self.model(x_t, t, cond)
        return nn.functional.mse_loss(pred, target)

    def fit(
        self,
        traj: np.ndarray,
        cond: np.ndarray,
        epochs: int | None = None,
        verbose: bool = False,
    ) -> list[float]:
        torch.manual_seed(self.cfg.seed)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        cond_t = torch.tensor(cond, dtype=torch.float32)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        n = traj_t.shape[0]
        losses: list[float] = []
        n_epochs = epochs or self.cfg.epochs

        self.model.train()
        for epoch in range(n_epochs):
            idx = torch.randperm(n)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, n, self.cfg.batch_size):
                batch = idx[start : start + self.cfg.batch_size]
                opt.zero_grad()
                loss = self.loss(traj_t[batch], cond_t[batch])
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item())
                steps += 1
            losses.append(epoch_loss / max(steps, 1))
            if verbose and (epoch + 1) % 50 == 0:
                print(f"  epoch {epoch + 1}/{n_epochs}  loss={losses[-1]:.4f}")
        return losses

    @torch.no_grad()
    def sample(
        self,
        cond: np.ndarray,
        n_samples: int | None = None,
        steps: int | None = None,
        integrator: str = "heun",
        return_history: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        self.model.eval()
        cond_np = np.asarray(cond, dtype=float)
        if cond_np.ndim == 1:
            cond_np = cond_np[None, :]
        if n_samples is None:
            n_samples = cond_np.shape[0]
        if cond_np.shape[0] == 1 and n_samples > 1:
            cond_np = np.repeat(cond_np, n_samples, axis=0)
        elif cond_np.shape[0] != n_samples:
            raise ValueError("cond batch size must be 1 or equal to n_samples")

        n_steps = steps or self.cfg.sample_steps
        cond_t = torch.tensor(cond_np, dtype=torch.float32)
        x = torch.randn(n_samples, self.traj_dim)
        dt = 1.0 / n_steps
        history: list[np.ndarray] = []

        for step in range(n_steps):
            t_val = step * dt
            t = torch.full((n_samples,), t_val, dtype=torch.float32)
            if integrator == "euler":
                v = self.model(x, t, cond_t)
                x = x + dt * v
            else:
                v1 = self.model(x, t, cond_t)
                t_next = torch.full((n_samples,), t_val + dt, dtype=torch.float32)
                x_pred = x + dt * v1
                v2 = self.model(x_pred, t_next, cond_t)
                x = x + dt * 0.5 * (v1 + v2)
            if return_history:
                history.append(self.unflatten(x, self.cfg.horizon).numpy())

        out = self.unflatten(x, self.cfg.horizon).numpy()
        if return_history:
            return out, history
        return out
