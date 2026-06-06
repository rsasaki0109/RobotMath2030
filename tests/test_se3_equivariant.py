"""Tests for SE(3)-equivariant point readout (requires PyTorch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from miniworlds.point_cloud_3d_world import make_dataset
from robotmath.equivariance import (
    EquivariantTrainConfig,
    EquivariantVectorReadout,
    NaivePointMLP,
    eval_canonical_error,
    eval_rotated_error,
    random_rotation,
    rotate_points,
    train_vector_predictor,
)


@pytest.fixture
def dataset():
    pts, tgt = make_dataset(n_clouds=128, seed=0)
    return pts, tgt


def test_rotate_points_batch():
    R = random_rotation(1)
    pts = np.random.randn(4, 10, 3)
    out = rotate_points(pts, R)
    assert out.shape == pts.shape


def test_equivariant_rotation_consistent(dataset):
    pts, tgt = make_dataset(n_clouds=64, seed=1)
    cfg = EquivariantTrainConfig(n_points=pts.shape[1], epochs=80, seed=0)
    model = EquivariantVectorReadout(hidden=24)
    train_vector_predictor(model, pts, tgt, cfg)
    R = random_rotation(42)
    can = eval_canonical_error(model, pts[:16], tgt[:16])
    rot = eval_rotated_error(model, pts[:16], tgt[:16], R)
    assert rot < can * 3.0 + 1e-3


def test_naive_fails_on_rotation(dataset):
    pts, tgt = dataset
    cfg = EquivariantTrainConfig(n_points=pts.shape[1], epochs=80, seed=0)
    model = NaivePointMLP(pts.shape[1], hidden=48)
    train_vector_predictor(model, pts, tgt, cfg)
    R = random_rotation(42)
    can = eval_canonical_error(model, pts[:16], tgt[:16])
    rot = eval_rotated_error(model, pts[:16], tgt[:16], R)
    assert rot > can * 5.0
