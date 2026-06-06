"""Tests for 2D Riemannian motion policies."""

import numpy as np

from miniworlds.rmp_world import GOAL, RMP_START
from robotmath.motion import RMPConfig, fuse_rmp, rollout
from robotmath.motion.rmp_2d import goal_task, obstacle_task, rmp_acceleration


def test_fused_metric_is_spd():
    cfg = RMPConfig()
    q = np.array([0.48, 0.35])
    qd = np.array([0.0, 0.2])
    tasks = [
        goal_task(q, qd, GOAL, cfg),
        obstacle_task(q, np.array([0.5, 0.55]), 0.12, cfg),
    ]
    m_total = cfg.damping * np.eye(2) + sum(t[0] for t in tasks)
    eigvals = np.linalg.eigvalsh(m_total)
    assert np.all(eigvals > 0)


def test_rmp_reaches_goal_without_collision():
    cfg = RMPConfig()
    traj, reached, collided = rollout(RMP_START, GOAL, policy="rmp", cfg=cfg)
    assert reached
    assert not collided
    assert traj.shape[0] > 10


def test_naive_collides_on_same_setup():
    cfg = RMPConfig()
    _, reached, collided = rollout(RMP_START, GOAL, policy="naive", cfg=cfg)
    assert not reached
    assert collided


def test_rmp_acceleration_is_finite():
    cfg = RMPConfig()
    q = np.array([0.48, 0.40])
    qd = np.array([0.1, 0.5])
    a = rmp_acceleration(q, qd, GOAL, cfg)
    assert np.all(np.isfinite(a))
