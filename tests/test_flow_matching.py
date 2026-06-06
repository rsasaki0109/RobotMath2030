"""Tests for 2D flow matching policy (requires PyTorch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from miniworlds.two_path_world import START, GOAL, collision_rate, generate_demonstrations
from robotmath.diffusion import FlowMatchingPolicy2D, FlowTrainConfig


@pytest.fixture
def demo_data():
    demos, cond = generate_demonstrations(n_per_mode=12, horizon=16, seed=0)
    test_cond = np.concatenate([START, GOAL])[None, :]
    return demos, cond, test_cond


def test_flow_matching_samples_avoid_obstacle(demo_data):
    demos, cond, test_cond = demo_data
    cfg = FlowTrainConfig(horizon=16, epochs=80, hidden=64, sample_steps=10, seed=0)
    policy = FlowMatchingPolicy2D(cfg)
    policy.fit(demos, cond)
    samples = policy.sample(test_cond, n_samples=12)
    assert samples.shape == (12, 16, 2)
    assert collision_rate(samples) <= 0.75


def test_flow_matching_loss_decreases(demo_data):
    demos, cond, _ = demo_data
    cfg = FlowTrainConfig(horizon=16, epochs=50, hidden=64, seed=0)
    policy = FlowMatchingPolicy2D(cfg)
    losses = policy.fit(demos, cond)
    assert losses[-1] < losses[0]


def test_flow_matching_fewer_steps_still_works(demo_data):
    demos, cond, test_cond = demo_data
    cfg = FlowTrainConfig(horizon=16, epochs=80, hidden=64, sample_steps=8, seed=0)
    policy = FlowMatchingPolicy2D(cfg)
    policy.fit(demos, cond)
    samples = policy.sample(test_cond, n_samples=8, steps=8)
    assert collision_rate(samples) <= 0.875
