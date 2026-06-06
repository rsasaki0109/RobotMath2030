"""Tests for 2D diffusion policy (requires PyTorch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from miniworlds.two_path_world import START, GOAL, collision_rate, generate_demonstrations
from robotmath.diffusion import DiffusionPolicy2D, TrainConfig, predict_mean_regression, train_mean_regression


@pytest.fixture
def demo_data():
    demos, cond = generate_demonstrations(n_per_mode=12, horizon=16, seed=0)
    test_cond = np.concatenate([START, GOAL])[None, :]
    return demos, cond, test_cond


def test_mean_regression_collides(demo_data):
    demos, cond, test_cond = demo_data
    model = train_mean_regression(demos, cond, horizon=16, epochs=60, seed=0)
    pred = predict_mean_regression(model, test_cond, horizon=16)
    assert collision_rate(pred) == 1.0


def test_diffusion_samples_avoid_obstacle(demo_data):
    demos, cond, test_cond = demo_data
    cfg = TrainConfig(horizon=16, timesteps=12, epochs=80, hidden=64, seed=0)
    policy = DiffusionPolicy2D(cfg)
    policy.fit(demos, cond)
    samples = policy.sample(test_cond, n_samples=12)
    assert samples.shape == (12, 16, 2)
    assert collision_rate(samples) <= 0.75


def test_diffusion_loss_decreases(demo_data):
    demos, cond, _ = demo_data
    cfg = TrainConfig(horizon=16, timesteps=12, epochs=50, hidden=64, seed=0)
    policy = DiffusionPolicy2D(cfg)
    losses = policy.fit(demos, cond)
    assert losses[-1] < losses[0]
