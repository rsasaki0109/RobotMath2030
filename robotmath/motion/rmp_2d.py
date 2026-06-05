"""Tiny 2D Riemannian motion policies — readable reference, not RMPflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from miniworlds.two_path_world import GOAL, OBSTACLE_CENTER, OBSTACLE_RADIUS, START, hits_obstacle


@dataclass
class RMPConfig:
    dt: float = 0.02
    max_steps: int = 600
    goal_kp: float = 6.0
    goal_kd: float = 2.5
    goal_weight: float = 1.0
    obs_k: float = 30.0
    obs_weight: float = 5.0
    obs_influence: float = 0.22
    damping: float = 1e-3
    goal_tol: float = 0.03
    speed_tol: float = 0.05


def goal_task(
    q: np.ndarray,
    qd: np.ndarray,
    goal: np.ndarray,
    cfg: RMPConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Attract to goal in task space x = q."""
    j = np.eye(2)
    u = -cfg.goal_kp * (q - goal) - cfg.goal_kd * qd
    m_t = cfg.goal_weight * np.eye(2)
    m_q = j.T @ m_t @ j
    f_q = j.T @ m_t @ u
    return m_q, f_q


def obstacle_task(
    q: np.ndarray,
    center: np.ndarray,
    radius: float,
    cfg: RMPConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Repel when clearance drops below influence radius."""
    diff = q - center
    dist = float(np.linalg.norm(diff))
    if dist < 1e-8:
        direction = np.array([1.0, 0.0])
    else:
        direction = diff / dist

    clearance = dist - radius
    if clearance >= cfg.obs_influence:
        return np.zeros((2, 2)), np.zeros(2)

    j = direction.reshape(1, 2)
    u = np.array([cfg.obs_k * (cfg.obs_influence - clearance)])
    metric_scale = cfg.obs_weight / max(clearance, 0.02) ** 2
    m_t = np.array([[metric_scale]])
    m_q = j.T @ m_t @ j
    f_q = (j.T @ m_t @ u).reshape(2)
    return m_q, f_q


def fuse_rmp(tasks: list[tuple[np.ndarray, np.ndarray]], damping: float) -> np.ndarray:
    """Combine task metrics and forces: a = (ΣM + λI)⁻¹ Σf."""
    m_total = damping * np.eye(2)
    f_total = np.zeros(2)
    for m_q, f_q in tasks:
        m_total += m_q
        f_total += f_q
    return np.linalg.solve(m_total, f_total)


def rmp_acceleration(q: np.ndarray, qd: np.ndarray, goal: np.ndarray, cfg: RMPConfig) -> np.ndarray:
    tasks = [
        goal_task(q, qd, goal, cfg),
        obstacle_task(q, OBSTACLE_CENTER, OBSTACLE_RADIUS, cfg),
    ]
    return fuse_rmp(tasks, cfg.damping)


def naive_acceleration(q: np.ndarray, qd: np.ndarray, goal: np.ndarray, cfg: RMPConfig) -> np.ndarray:
    """Sum task-space pushes without Riemannian metric fusion."""
    a_goal = -cfg.goal_kp * (q - goal) - cfg.goal_kd * qd
    diff = q - OBSTACLE_CENTER
    dist = float(np.linalg.norm(diff))
    if dist < 1e-8:
        direction = np.array([1.0, 0.0])
    else:
        direction = diff / dist
    clearance = dist - OBSTACLE_RADIUS
    if clearance < cfg.obs_influence:
        a_obs = cfg.obs_k * (cfg.obs_influence - clearance) * direction
    else:
        a_obs = np.zeros(2)
    return a_goal + a_obs


def rollout(
    start: np.ndarray,
    goal: np.ndarray,
    policy: str = "rmp",
    cfg: RMPConfig | None = None,
) -> tuple[np.ndarray, bool, bool]:
    """
    Simulate second-order point-mass motion.

    Returns (trajectory, reached_goal, collided).
    """
    cfg = cfg or RMPConfig()
    accel_fn = rmp_acceleration if policy == "rmp" else naive_acceleration

    q = np.asarray(start, dtype=float).copy()
    qd = np.zeros(2, dtype=float)
    traj = [q.copy()]

    for _ in range(cfg.max_steps):
        if hits_obstacle(q):
            return np.stack(traj), False, True

        dist_goal = float(np.linalg.norm(q - goal))
        if dist_goal < cfg.goal_tol and np.linalg.norm(qd) < cfg.speed_tol:
            return np.stack(traj), True, False

        a = accel_fn(q, qd, goal, cfg)
        qd = qd + cfg.dt * a
        q = q + cfg.dt * qd
        traj.append(q.copy())

    return np.stack(traj), False, False


def reached_goal(traj: np.ndarray, goal: np.ndarray, tol: float = 0.04) -> bool:
    return float(np.linalg.norm(traj[-1] - goal)) < tol
