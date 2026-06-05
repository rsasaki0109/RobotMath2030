"""Minimal SE(3)-equivariant vector readout vs naive MLP on point clouds."""

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


def random_rotation(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q.astype(float)


def rotate_points(points: np.ndarray, R: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    R = np.asarray(R, dtype=float)
    if points.ndim == 1:
        return points @ R.T
    if points.ndim == 2:
        return points @ R.T
    return points @ R.T


if torch is not None:

    class NaivePointMLP(nn.Module):
        """Flattened coordinates — not SE(3)-equivariant."""

        def __init__(self, n_points: int, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_points * 3, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 3),
            )

        def forward(self, points: torch.Tensor) -> torch.Tensor:
            b = points.shape[0]
            return self.net(points.reshape(b, -1))

    class EquivariantVectorReadout(nn.Module):
        """
        Scalar weights from invariant distances, linear combo of centered vectors.

        v = Σ_i w_i (p_i - c)  transforms equivariantly with the point cloud.
        """

        def __init__(self, hidden: int = 32):
            super().__init__()
            self.scorer = nn.Sequential(
                nn.Linear(1, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, points: torch.Tensor) -> torch.Tensor:
            centroid = points.mean(dim=1, keepdim=True)
            centered = points - centroid
            radii = torch.linalg.norm(centered, dim=-1, keepdim=True)
            weights = self.scorer(radii)
            return (weights * centered).sum(dim=1)

else:  # pragma: no cover
    NaivePointMLP = object  # type: ignore[misc, assignment]
    EquivariantVectorReadout = object  # type: ignore[misc, assignment]


@dataclass
class EquivariantTrainConfig:
    n_points: int = 24
    hidden: int = 64
    lr: float = 2e-3
    epochs: int = 120
    batch_size: int = 32
    seed: int = 0


def train_vector_predictor(
    model,
    points: np.ndarray,
    targets: np.ndarray,
    cfg: EquivariantTrainConfig,
) -> list[float]:
    _require_torch()
    torch.manual_seed(cfg.seed)
    pts = torch.tensor(points, dtype=torch.float32)
    tgt = torch.tensor(targets, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    losses: list[float] = []
    n = pts.shape[0]

    model.train()
    for _ in range(cfg.epochs):
        idx = torch.randperm(n)
        epoch = 0.0
        steps = 0
        for start in range(0, n, cfg.batch_size):
            b = idx[start : start + cfg.batch_size]
            pred = model(pts[b])
            loss = nn.functional.mse_loss(pred, tgt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch += float(loss.item())
            steps += 1
        losses.append(epoch / max(steps, 1))
    model.eval()
    return losses


@torch.no_grad()
def eval_rotated_error(
    model,
    points: np.ndarray,
    targets: np.ndarray,
    R: np.ndarray,
) -> float:
    _require_torch()
    pts_r = rotate_points(points, R)
    tgt_r = rotate_points(targets, R)
    pred = model(torch.tensor(pts_r, dtype=torch.float32)).numpy()
    return float(np.mean(np.sum((pred - tgt_r) ** 2, axis=1)))


@torch.no_grad()
def eval_canonical_error(
    model,
    points: np.ndarray,
    targets: np.ndarray,
) -> float:
    _require_torch()
    pred = model(torch.tensor(points, dtype=torch.float32)).numpy()
    return float(np.mean(np.sum((pred - targets) ** 2, axis=1)))
