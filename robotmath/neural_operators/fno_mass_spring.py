"""Tiny Fourier Neural Operator for damped mass-spring trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotmath.neural_operators.deeponet_mass_spring import (
    OperatorTrainConfig,
    make_operator_dataset,
    operator_trajectory_mse,
)
from robotmath.physics.mass_spring import MassSpringParams, simulate

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required. Install with: pip install -e '.[torch]'")


if torch is not None:

    class SpectralConv1d(nn.Module):
        """1D spectral convolution (FNO layer core)."""

        def __init__(self, in_channels: int, out_channels: int, modes: int):
            super().__init__()
            self.modes = modes
            scale = 1.0 / max(in_channels * out_channels, 1)
            self.weights = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, _, n = x.shape
            x_ft = torch.fft.rfft(x)
            out_ft = torch.zeros(
                batch,
                self.weights.shape[1],
                x_ft.shape[-1],
                device=x.device,
                dtype=torch.cfloat,
            )
            modes = min(self.modes, x_ft.shape[-1])
            out_ft[:, :, :modes] = torch.einsum(
                "bix,iox->box",
                x_ft[:, :, :modes],
                self.weights[:, :, :modes],
            )
            return torch.fft.irfft(out_ft, n=n)

    class FNOBlock1d(nn.Module):
        def __init__(self, width: int, modes: int):
            super().__init__()
            self.spectral = SpectralConv1d(width, width, modes)
            self.pointwise = nn.Conv1d(width, width, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.gelu(self.spectral(x) + self.pointwise(x))

    class FNO1D(nn.Module):
        """
    Map initial state u = (x0, v0) to a full position trajectory on a fixed time grid.

    Input channels: broadcast ICs plus normalized time coordinate (FNO grid input).
        """

        def __init__(self, width: int = 32, modes: int = 12, depth: int = 3):
            super().__init__()
            self.lift = nn.Conv1d(3, width, kernel_size=1)
            self.blocks = nn.ModuleList(FNOBlock1d(width, modes) for _ in range(depth))
            self.project = nn.Sequential(
                nn.Conv1d(width, width, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(width, 1, kernel_size=1),
            )

        def forward(self, u: torch.Tensor, t_len: int) -> torch.Tensor:
            device, dtype = u.device, u.dtype
            t_grid = torch.linspace(0.0, 1.0, t_len, device=device, dtype=dtype)
            t_grid = t_grid.view(1, 1, t_len).expand(u.shape[0], 1, t_len)
            x0 = u[:, 0:1].unsqueeze(-1).expand(-1, 1, t_len)
            v0 = u[:, 1:2].unsqueeze(-1).expand(-1, 1, t_len)
            x = torch.cat([x0, v0, t_grid], dim=1)
            x = self.lift(x)
            for block in self.blocks:
                x = x + block(x)
            return self.project(x).squeeze(1)

        def predict_trajectory(
            self,
            u: np.ndarray,
            times: np.ndarray,
        ) -> np.ndarray:
            self.eval()
            u_t = torch.tensor(u, dtype=torch.float32)
            if u_t.ndim == 1:
                u_t = u_t.unsqueeze(0)
            with torch.no_grad():
                pred = self.forward(u_t, len(times))
            return pred.numpy()

else:  # pragma: no cover
    FNO1D = object  # type: ignore[misc, assignment]


def train_fno(
    initial_states: np.ndarray,
    times: np.ndarray,
    positions: np.ndarray,
    cfg: OperatorTrainConfig | None = None,
    width: int = 32,
    modes: int = 12,
    depth: int = 3,
) -> tuple["FNO1D", list[float]]:
    _require_torch()
    cfg = cfg or OperatorTrainConfig()
    torch.manual_seed(cfg.seed)
    model = FNO1D(width=width, modes=modes, depth=depth)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    u_all = torch.tensor(initial_states, dtype=torch.float32)
    y_all = torch.tensor(positions, dtype=torch.float32)
    t_len = y_all.shape[1]
    losses: list[float] = []

    model.train()
    for _ in range(cfg.epochs):
        idx = torch.randperm(u_all.shape[0])
        epoch_loss = 0.0
        steps = 0
        for start in range(0, u_all.shape[0], cfg.batch_size):
            b = idx[start : start + cfg.batch_size]
            pred = model(u_all[b], t_len)
            loss = F.mse_loss(pred, y_all[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            steps += 1
        losses.append(epoch_loss / max(steps, 1))
    model.eval()
    return model, losses


def fno_trajectory_mse(
    model: "FNO1D",
    initial_states: np.ndarray,
    times: np.ndarray,
    positions: np.ndarray,
) -> float:
    pred = model.predict_trajectory(initial_states, times)
    return float(np.mean((pred - positions) ** 2))
