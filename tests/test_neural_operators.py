"""Tests for neural operator mass-spring DeepONet (requires PyTorch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robotmath.neural_operators import (
    OperatorTrainConfig,
    make_operator_dataset,
    operator_trajectory_mse,
    train_deeponet,
)


def test_deeponet_learns_mass_spring_family():
    train_u, times, train_y = make_operator_dataset(n_samples=128, seed=0)
    test_u, _, test_y = make_operator_dataset(n_samples=32, seed=1)
    cfg = OperatorTrainConfig(epochs=100, batch_size=32, seed=0)
    model, losses = train_deeponet(train_u, times, train_y, cfg)
    test_mse = operator_trajectory_mse(model, test_u, times, test_y)
    assert test_mse < 0.05
    assert losses[-1] < losses[0]
