"""Tests for Ch.15 FNO mass-spring operator."""

from __future__ import annotations

import pytest

from robotmath.neural_operators import OperatorTrainConfig, make_operator_dataset, operator_trajectory_mse, train_deeponet
from robotmath.neural_operators.fno_mass_spring import fno_trajectory_mse, train_fno

torch = pytest.importorskip("torch")


def test_fno_reaches_target_mse():
    train_u, times, train_y = make_operator_dataset(n_samples=128, seed=0)
    test_u, _, test_y = make_operator_dataset(n_samples=32, seed=1)
    cfg = OperatorTrainConfig(epochs=60, seed=0)
    model, losses = train_fno(train_u, times, train_y, cfg)
    assert losses[-1] < losses[0]
    mse = fno_trajectory_mse(model, test_u, times, test_y)
    assert mse < 0.12


def test_fno_beats_or_matches_deeponet_on_tiny_set():
    train_u, times, train_y = make_operator_dataset(n_samples=128, seed=2)
    test_u, _, test_y = make_operator_dataset(n_samples=32, seed=3)
    cfg = OperatorTrainConfig(epochs=60, seed=2)
    fno, _ = train_fno(train_u, times, train_y, cfg)
    deep, _ = train_deeponet(train_u, times, train_y, cfg)
    fno_mse = fno_trajectory_mse(fno, test_u, times, test_y)
    deep_mse = operator_trajectory_mse(deep, test_u, times, test_y)
    assert fno_mse < 0.12
    assert deep_mse < 0.12
