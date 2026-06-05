"""Tests for differentiable physics demos."""

import numpy as np
import pytest

from miniworlds.mass_spring_world import bouncing_ball_rollout, noisy_mass_spring_rollout
from robotmath.physics import (
    MassSpringParams,
    identify_autograd,
    identify_finite_diff,
    simulate,
    simulate_bounce_hard,
)


def test_mass_spring_rollout_shape():
    obs, params = noisy_mass_spring_rollout(steps=50)
    assert obs.shape == (51, 2)
    assert params.spring_k == 8.0


def test_finite_diff_recovers_parameters():
    torch = pytest.importorskip("torch")
    del torch  # only checking availability not needed here
    obs, true = noisy_mass_spring_rollout(steps=60, noise=0.005, seed=1)
    est, losses = identify_finite_diff(obs, obs[0, 0], obs[0, 1], steps=100)
    assert losses[-1] < losses[0]
    assert abs(est.spring_k - true.spring_k) < 1.5
    assert abs(est.damping - true.damping) < 0.5


def test_autograd_recovers_parameters():
    torch = pytest.importorskip("torch")
    del torch
    obs, true = noisy_mass_spring_rollout(steps=60, noise=0.005, seed=2)
    est, losses = identify_autograd(obs, obs[0, 0], obs[0, 1], steps=120)
    assert losses[-1] < losses[0]
    assert abs(est.spring_k - true.spring_k) < 1.0
    assert abs(est.damping - true.damping) < 0.4


def test_bounce_stays_nonnegative():
    _, params = bouncing_ball_rollout(steps=100)
    traj = simulate_bounce_hard(params, y0=1.2, vy0=0.5, steps=100)
    assert np.all(traj[:, 0] >= -1e-10)
