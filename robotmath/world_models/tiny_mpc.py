"""Short-horizon imagination planning with a tiny world model."""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from robotmath.world_models.latent_dynamics import agent_position, goal_position


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required. Install with: pip install -e '.[torch]'")


def _onehot(action: int, n: int = 4) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[action] = 1.0
    return v


def imagination_cost(pred_obs: np.ndarray, goal_obs: np.ndarray) -> float:
    """Lower is better — Manhattan distance from predicted agent to goal cell."""
    ay, ax = agent_position(pred_obs)
    gy, gx = goal_position(goal_obs)
    return float(abs(ay - gy) + abs(ax - gx))


def _wall_plane(obs: np.ndarray) -> torch.Tensor:
    side = int(round(np.sqrt(obs.size // 4)))
    wall = obs.reshape(1, 4, side, side)[0, 0:1]
    return torch.tensor(wall, dtype=torch.float32)


def plan_greedy_action(model, obs: np.ndarray, goal_obs: np.ndarray) -> int:
    """One-step lookahead in imagination."""
    _require_torch()
    obs_flat = obs.reshape(1, -1)
    obs_t = torch.tensor(obs_flat, dtype=torch.float32)
    wall = _wall_plane(obs)
    best_a, best_c = 0, float("inf")
    with torch.no_grad():
        for a in range(4):
            act = torch.tensor(_onehot(a)[None, :], dtype=torch.float32)
            nxt = model.predict_next(obs_t, act, wall_plane=wall)
            cost = imagination_cost(nxt.numpy()[0], goal_obs)
            if cost < best_c:
                best_c = cost
                best_a = a
    return best_a


def plan_action_sequence(
    model,
    obs: np.ndarray,
    goal_obs: np.ndarray,
    horizon: int = 5,
    n_candidates: int = 64,
    seed: int = 0,
) -> list[int]:
    """Multi-step random shooting in imagination (used for open-loop plans)."""
    _require_torch()
    rng = np.random.default_rng(seed)
    obs_flat = obs.reshape(-1)
    obs_t = torch.tensor(obs_flat[None, :], dtype=torch.float32)
    wall = _wall_plane(obs)
    best_cost = float("inf")
    best_seq: list[int] = [0] * horizon
    with torch.no_grad():
        for _ in range(n_candidates):
            seq = [int(rng.integers(0, 4)) for _ in range(horizon)]
            cur = obs_t
            total = 0.0
            for a in seq:
                act = torch.tensor(_onehot(a)[None, :], dtype=torch.float32)
                cur = model.predict_next(cur, act, wall_plane=wall)
                total += imagination_cost(cur.numpy()[0], goal_obs)
            if total < best_cost:
                best_cost = total
                best_seq = seq
    return best_seq


def rollout_env(
    env,
    model,
    goal_obs: np.ndarray,
    replan: bool = True,
    horizon: int = 5,
    n_candidates: int = 48,
    seed: int = 0,
) -> tuple[list[tuple[int, int]], bool]:
    """Closed-loop greedy replanning vs fixed open-loop action sequence."""
    obs = env.reset()
    path = [env.agent]
    done = False

    if not replan:
        plan = plan_action_sequence(
            model, obs, goal_obs, horizon=horizon, n_candidates=n_candidates, seed=seed,
        )

    while not done:
        if replan:
            action = plan_greedy_action(model, obs, goal_obs)
        else:
            step_i = len(path) - 1
            action = plan[step_i] if step_i < len(plan) else 0
        obs, _, done = env.step(action)
        path.append(env.agent)
        if not replan and len(path) - 1 >= len(plan):
            break

    return path, env.agent == env.goal
