"""2D grid world for tiny world-model demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray

EMPTY, WALL, AGENT, GOAL = 0, 1, 2, 3
ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}


@dataclass
class GridWorldConfig:
    size: int = 8
    max_steps: int = 40


class GridWorld:
    """Minimal grid navigation: reach the goal without hitting walls."""

    def __init__(self, cfg: GridWorldConfig | None = None, layout: str = "default"):
        self.cfg = cfg or GridWorldConfig()
        self.layout = layout
        self.grid = np.zeros((self.cfg.size, self.cfg.size), dtype=int)
        self.agent = (1, 1)
        self.goal = (self.cfg.size - 2, self.cfg.size - 2)
        self.steps = 0
        self._build_layout()

    def _build_layout(self):
        n = self.cfg.size
        self.grid.fill(EMPTY)
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL
        if self.layout == "default":
            self.grid[2:6, 4] = WALL
            self.grid[4, 2:5] = WALL
        elif self.layout == "easy":
            pass
        self.grid[self.agent] = EMPTY
        self.grid[self.goal] = GOAL

    def reset(self) -> Array:
        self.agent = (1, 1)
        self.steps = 0
        self._build_layout()
        return self.observation()

    def observation(self) -> Array:
        """Return (4, H, W) channels: wall, agent, goal, empty."""
        n = self.cfg.size
        obs = np.zeros((4, n, n), dtype=np.float32)
        for r in range(n):
            for c in range(n):
                cell = self.grid[r, c]
                if (r, c) == self.agent:
                    obs[1, r, c] = 1.0
                elif cell == WALL:
                    obs[0, r, c] = 1.0
                elif cell == GOAL:
                    obs[2, r, c] = 1.0
                else:
                    obs[3, r, c] = 1.0
        return obs

    def step(self, action: int) -> tuple[Array, float, bool]:
        dy, dx = ACTIONS[int(action) % 4]
        ay, ax = self.agent
        ny, nx = ay + dy, ax + dx
        if self.grid[ny, nx] == WALL:
            ny, nx = ay, ax
        self.agent = (ny, nx)
        self.steps += 1
        reward = 1.0 if self.agent == self.goal else -0.01
        done = self.agent == self.goal or self.steps >= self.cfg.max_steps
        return self.observation(), reward, done

    def agent_goal_distance(self) -> float:
        ay, ax = self.agent
        gy, gx = self.goal
        return float(abs(ay - gy) + abs(ax - gx))

    def collect_random_dataset(self, n_transitions: int = 3000, seed: int = 0) -> dict[str, Array]:
        rng = np.random.default_rng(seed)
        obs_list, act_list, next_list = [], [], []
        obs = self.reset()
        for _ in range(n_transitions):
            action = int(rng.integers(0, 4))
            next_obs, _, done = self.step(action)
            obs_list.append(obs)
            act_list.append(action)
            next_list.append(next_obs)
            obs = next_obs if not done else self.reset()
        return {
            "obs": np.stack(obs_list),
            "action": np.array(act_list, dtype=np.int64),
            "next_obs": np.stack(next_list),
        }
