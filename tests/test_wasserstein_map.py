"""Tests for Wasserstein occupancy map evaluation."""

import numpy as np

from miniworlds.occupancy_map_world import (
    l2_grid_mse,
    map_pair_drift,
    map_pair_ghost,
    map_pair_identical,
    map_to_points,
)
from robotmath.optimal_transport import compare_map_metrics, wasserstein_between_maps

EPS = 0.01


def test_identical_maps_near_zero():
    a, b, _ = map_pair_identical()
    m = compare_map_metrics(a, b, map_to_points, l2_grid_mse, epsilon=EPS)
    assert m["l2_grid_mse"] < 1e-12
    assert m["wasserstein2"] < 0.02


def test_drift_increases_wasserstein():
    ref, shifted, _ = map_pair_drift()
    ident_a, ident_b, _ = map_pair_identical()
    drift = compare_map_metrics(ref, shifted, map_to_points, l2_grid_mse, epsilon=EPS)
    ident = compare_map_metrics(ident_a, ident_b, map_to_points, l2_grid_mse, epsilon=EPS)
    assert drift["wasserstein2"] > ident["wasserstein2"]


def test_ghost_l2_paradox():
    """Ghost map can look closer in L2 than drifted map — failure case."""
    ref_d, shifted, _ = map_pair_drift()
    ref_g, ghost, _ = map_pair_ghost()
    drift = compare_map_metrics(ref_d, shifted, map_to_points, l2_grid_mse, epsilon=EPS)
    ghost_m = compare_map_metrics(ref_g, ghost, map_to_points, l2_grid_mse, epsilon=EPS)
    assert ghost_m["l2_grid_mse"] < drift["l2_grid_mse"]


def test_wasserstein_between_maps_shapes():
    ref, shifted, _ = map_pair_drift()
    cost, plan, pts_a, pts_b = wasserstein_between_maps(
        ref, shifted, map_to_points, epsilon=EPS, max_iter=500,
    )
    assert cost > 0
    assert plan.shape == (pts_a.shape[0], pts_b.shape[0])
    assert pts_a.shape[1] == 2
