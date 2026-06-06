"""Tiny 2D trajectory diffusion policy — readable reference, not production IL."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required. Install with: pip install -e '.[torch]'")


if torch is not None:

    class MeanRegressionPolicy(nn.Module):
        """Unimodal baseline: one deterministic trajectory per condition."""

        def __init__(self, traj_dim: int, cond_dim: int, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(cond_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, traj_dim),
            )

        def forward(self, cond: torch.Tensor) -> torch.Tensor:
            return self.net(cond)

    class EpsilonPredictor(nn.Module):
        """Predicts diffusion noise epsilon given noisy trajectory, timestep, condition."""

        def __init__(self, traj_dim: int, cond_dim: int, timesteps: int, hidden: int = 128):
            super().__init__()
            self.timesteps = timesteps
            self.net = nn.Sequential(
                nn.Linear(traj_dim + cond_dim + 1, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, traj_dim),
            )

        def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            t_norm = t.float().unsqueeze(-1) / self.timesteps
            return self.net(torch.cat([x_t, cond, t_norm], dim=-1))

else:  # pragma: no cover
    MeanRegressionPolicy = object  # type: ignore[misc, assignment]
    EpsilonPredictor = object  # type: ignore[misc, assignment]


@dataclass
class TrainConfig:
    horizon: int = 24
    timesteps: int = 20
    hidden: int = 64
    lr: float = 2e-3
    epochs: int = 120
    batch_size: int = 64
    seed: int = 0


class DiffusionPolicy2D:
    """DDPM-style conditional diffusion on flattened 2D trajectories."""

    def __init__(self, cfg: TrainConfig | None = None):
        _require_torch()
        self.cfg = cfg or TrainConfig()
        self.traj_dim = self.cfg.horizon * 2
        self.cond_dim = 4
        self.model = EpsilonPredictor(
            self.traj_dim, self.cond_dim, self.cfg.timesteps, hidden=self.cfg.hidden
        )
        self._setup_schedule()

    def _setup_schedule(self):
        T = self.cfg.timesteps
        beta = torch.linspace(1e-4, 2e-2, T)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.beta = beta
        self.alpha = alpha
        self.alpha_bar = alpha_bar

    @staticmethod
    def flatten(traj: torch.Tensor) -> torch.Tensor:
        return traj.reshape(traj.shape[0], -1)

    @staticmethod
    def unflatten(x: torch.Tensor, horizon: int) -> torch.Tensor:
        return x.reshape(x.shape[0], horizon, 2)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        ab = self.alpha_bar[t].unsqueeze(-1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

    def loss(self, traj: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x0 = self.flatten(traj)
        b = x0.shape[0]
        t = torch.randint(0, self.cfg.timesteps, (b,))
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.model(x_t, t, cond)
        return nn.functional.mse_loss(pred, noise)

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
                x = traj_t[batch]
                c = cond_t[batch]
                opt.zero_grad()
                loss = self.loss(x, c)
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

        cond_t = torch.tensor(cond_np, dtype=torch.float32)
        x = torch.randn(n_samples, self.traj_dim)
        history: list[np.ndarray] = []

        for step in reversed(range(self.cfg.timesteps)):
            t = torch.full((n_samples,), step, dtype=torch.long)
            eps = self.model(x, t, cond_t)
            ab = self.alpha_bar[step]
            a = self.alpha[step]
            b = self.beta[step]
            mean = (x - (1.0 - a) / torch.sqrt(1.0 - ab) * eps) / torch.sqrt(a)
            if step > 0:
                x = mean + torch.sqrt(b) * torch.randn_like(x)
            else:
                x = mean
            if return_history:
                history.append(self.unflatten(x, self.cfg.horizon).numpy())

        out = self.unflatten(x, self.cfg.horizon).numpy()
        if return_history:
            history.reverse()
            return out, history
        return out


def train_mean_regression(
    traj: np.ndarray,
    cond: np.ndarray,
    horizon: int = 24,
    epochs: int = 300,
    lr: float = 1e-3,
    seed: int = 0,
) -> MeanRegressionPolicy:
    _require_torch()
    torch.manual_seed(seed)
    model = MeanRegressionPolicy(horizon * 2, cond.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    traj_t = torch.tensor(traj, dtype=torch.float32)
    cond_t = torch.tensor(cond, dtype=torch.float32)
    target = traj_t.reshape(traj_t.shape[0], -1)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(cond_t)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_mean_regression(
    model: MeanRegressionPolicy,
    cond: np.ndarray,
    horizon: int = 24,
) -> np.ndarray:
    cond_t = torch.tensor(np.asarray(cond, dtype=float), dtype=torch.float32)
    if cond_t.ndim == 1:
        cond_t = cond_t.unsqueeze(0)
    flat = model(cond_t).numpy()
    return flat.reshape(flat.shape[0], horizon, 2)
