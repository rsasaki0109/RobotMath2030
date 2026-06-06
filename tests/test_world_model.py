"""Tests for tiny latent world model."""

import pytest

torch = pytest.importorskip("torch")

from miniworlds.grid_world import GridWorld
from robotmath.world_models import WorldModelConfig, rollout_env, train_world_model


def test_grid_world_reaches_goal_with_manual_moves():
    env = GridWorld()
    env.reset()
    for _ in range(30):
        if env.agent == env.goal:
            break
        if env.agent[0] < env.goal[0]:
            env.step(1)
        elif env.agent[1] < env.goal[1]:
            env.step(3)
    assert env.agent == env.goal


def test_world_model_loss_decreases():
    env = GridWorld()
    data = env.collect_random_dataset(n_transitions=600, seed=0)
    cfg = WorldModelConfig(epochs=30, hidden=32, batch_size=64)
    _, losses = train_world_model(data, cfg)
    assert losses[-1] < losses[0]


def test_closed_loop_beats_open_loop():
    env = GridWorld(layout="easy")
    data = env.collect_random_dataset(n_transitions=1000, seed=1)
    model, _ = train_world_model(data, WorldModelConfig(epochs=60, hidden=48, seed=1))
    goal_obs = env.reset()
    goal_obs = env.observation()
    _, closed = rollout_env(env, model, goal_obs, replan=True, horizon=5, n_candidates=32, seed=2)
    _, open_loop = rollout_env(env, model, goal_obs, replan=False, horizon=4, n_candidates=64, seed=2)
    assert closed >= open_loop
    assert closed is True
