"""Tests for pose graph optimizer."""

import numpy as np

from miniworlds.pose_graph_world import square_loop_graph
from robotmath.lie import se2


def test_pose_graph_lie_reduces_cost():
    graph, _gt = square_loop_graph(seed=0)
    opt, costs = graph.optimize(use_lie=True, max_iters=50)
    assert costs[-1] < costs[0]
    assert len(opt.poses) == len(graph.poses)


def test_lie_beats_euclidean_on_loop():
    graph, gt = square_loop_graph(seed=1, loop_noise_theta=0.5)
    opt_lie, _ = graph.copy().optimize(use_lie=True, max_iters=50)
    opt_euc, _ = graph.copy().optimize(use_lie=False, max_iters=50)
    err_lie = _pose_error(opt_lie.poses, gt)
    err_euc = _pose_error(opt_euc.poses, gt)
    assert err_lie <= err_euc + 1e-3


def _pose_error(poses, gt):
    err = 0.0
    for T, G in zip(poses, gt):
        r = se2.log(se2.between(T, G))
        err += float(r @ r)
    return err
