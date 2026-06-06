"""Tiny world model — learns next agent cell from (y, x, action)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from miniworlds.grid_world import ACTIONS

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


def agent_position(obs: np.ndarray) -> tuple[int, int]:
    obs = np.asarray(obs)
    if obs.ndim == 3:
        plane = obs[1]
        side = obs.shape[1]
    else:
        side = int(round(np.sqrt(obs.size // 4)))
        plane = obs.reshape(4, side, side)[1]
    idx = int(np.argmax(plane))
    return divmod(idx, side)


def goal_position(obs: np.ndarray) -> tuple[int, int]:
    obs = np.asarray(obs)
    if obs.ndim == 3:
        plane = obs[2]
        side = obs.shape[1]
    else:
        side = int(round(np.sqrt(obs.size // 4)))
        plane = obs.reshape(4, side, side)[2]
    idx = int(np.argmax(plane))
    return divmod(idx, side)


def rebuild_observation(base_obs: np.ndarray, agent: tuple[int, int]) -> np.ndarray:
    base_obs = np.asarray(base_obs)
    if base_obs.ndim == 3:
        side = base_obs.shape[1]
        obs = base_obs.copy()
        obs[1] = 0.0
        obs[1, agent[0], agent[1]] = 1.0
        return obs.reshape(-1)
    side = int(round(np.sqrt(base_obs.size // 4)))
    obs = base_obs.copy().reshape(4, side, side)
    obs[1] = 0.0
    obs[1, agent[0], agent[1]] = 1.0
    return obs.reshape(-1)


def apply_grid_action(
    agent: tuple[int, int],
    action: int,
    wall_plane: np.ndarray,
) -> tuple[int, int]:
    dy, dx = ACTIONS[int(action) % 4]
    ny, nx = agent[0] + dy, agent[1] + dx
    if wall_plane[ny, nx] > 0.5:
        return agent
    return ny, nx


if torch is not None:

    class TinyWorldModel(nn.Module):
        def __init__(self, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2 + 4, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2),
            )

        def predict_position(self, pos: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat([pos, action_onehot], dim=-1))

        def predict_next(
            self,
            obs: torch.Tensor,
            action_onehot: torch.Tensor,
            wall_plane: torch.Tensor | None = None,
        ) -> torch.Tensor:
            b = obs.shape[0]
            side = int(round(np.sqrt(obs.shape[1] // 4)))
            pos_list = []
            for i in range(b):
                ay, ax = agent_position(obs[i].detach().cpu().numpy())
                pos_list.append([ay, ax])
            pos = torch.tensor(pos_list, dtype=torch.float32, device=obs.device)
            npos = self.predict_position(pos, action_onehot)
            next_obs = obs.clone().view(b, 4, side, side)
            next_obs[:, 1] = 0.0
            for i in range(b):
                ny = int(round(float(npos[i, 0])))
                nx = int(round(float(npos[i, 1])))
                ny = max(0, min(side - 1, ny))
                nx = max(0, min(side - 1, nx))
                if wall_plane is not None and wall_plane[i, ny, nx] > 0.5:
                    ny, nx = int(pos[i, 0]), int(pos[i, 1])
                next_obs[i, 1, ny, nx] = 1.0
            return next_obs.view(b, -1)

        def forward(self, pos: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
            return self.predict_position(pos, action_onehot)

else:  # pragma: no cover
    TinyWorldModel = object  # type: ignore[misc, assignment]


@dataclass
class WorldModelConfig:
    hidden: int = 64
    lr: float = 2e-3
    epochs: int = 60
    batch_size: int = 64
    seed: int = 0


def _onehot_actions(actions: np.ndarray, n: int = 4) -> np.ndarray:
    out = np.zeros((actions.shape[0], n), dtype=np.float32)
    out[np.arange(actions.shape[0]), actions] = 1.0
    return out


def _extract_positions(obs_batch: np.ndarray) -> np.ndarray:
    return np.array([agent_position(o) for o in obs_batch], dtype=np.float32)


def train_world_model(
    dataset: dict[str, np.ndarray],
    cfg: WorldModelConfig | None = None,
    grid_size: int = 8,
) -> tuple["TinyWorldModel", list[float]]:
    _require_torch()
    cfg = cfg or WorldModelConfig()
    torch.manual_seed(cfg.seed)
    obs = dataset["obs"]
    nxt = dataset["next_obs"]
    acts = _onehot_actions(dataset["action"])
    pos = _extract_positions(obs)
    npos = _extract_positions(nxt)
    pos_t = torch.tensor(pos, dtype=torch.float32)
    npos_t = torch.tensor(npos, dtype=torch.float32)
    act_t = torch.tensor(acts, dtype=torch.float32)
    obs_t = torch.tensor(obs.reshape(obs.shape[0], -1), dtype=torch.float32)
    model = TinyWorldModel(hidden=cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    losses: list[float] = []
    n = pos.shape[0]

    model.train()
    for _ in range(cfg.epochs):
        idx = torch.randperm(n)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, n, cfg.batch_size):
            b = idx[start : start + cfg.batch_size]
            pred = model(pos_t[b], act_t[b])
            loss = F.mse_loss(pred, npos_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            steps += 1
        losses.append(epoch_loss / max(steps, 1))
    model.eval()
    model._obs_template = obs_t[0:1]  # type: ignore[attr-defined]
    return model, losses
