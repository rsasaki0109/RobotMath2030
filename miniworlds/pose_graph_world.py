"""Synthetic pose graph world with odometry + loop closure."""

from __future__ import annotations

import numpy as np

from robotmath.lie import se2
from robotmath.optimization.tiny_factor_graph import PoseGraph, PoseGraphEdge


def square_loop_graph(
    side: float = 2.0,
    noise_xy: float = 0.05,
    noise_theta: float = 0.02,
    loop_noise_theta: float = 0.35,
    seed: int = 0,
) -> tuple[PoseGraph, list[np.ndarray]]:
    """
    Build a 4-node square trajectory with a noisy loop closure.

    Returns the initial (drifted) graph and ground-truth poses.
    """
    rng = np.random.default_rng(seed)

    gt = [
        se2.from_xytheta(0.0, 0.0, 0.0),
        se2.from_xytheta(side, 0.0, 0.0),
        se2.from_xytheta(side, side, np.pi / 2),
        se2.from_xytheta(0.0, side, np.pi / 2),
    ]

    def noisy_rel(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
        T_rel = se2.between(T_a, T_b)
        dx, dy, th = se2.to_xytheta(T_rel)
        dx += rng.normal(0.0, noise_xy)
        dy += rng.normal(0.0, noise_xy)
        th += rng.normal(0.0, noise_theta)
        return se2.from_xytheta(dx, dy, th)

    # Drifted odometry chain from node 0
    est = [gt[0].copy()]
    for k in range(1, 4):
        rel = noisy_rel(gt[k - 1], gt[k])
        est.append(se2.compose(est[-1], rel))

    edges = [
        PoseGraphEdge(0, 1, se2.between(est[0], est[1])),
        PoseGraphEdge(1, 2, se2.between(est[1], est[2])),
        PoseGraphEdge(2, 3, se2.between(est[2], est[3])),
    ]

    # Loop closure 3 -> 0 with biased rotation (simulates Euler/Lie confusion stress)
    true_loop = se2.between(gt[3], gt[0])
    dx, dy, th = se2.to_xytheta(true_loop)
    th += loop_noise_theta
    loop_meas = se2.from_xytheta(dx, dy, th)
    edges.append(PoseGraphEdge(3, 0, loop_meas))

    graph = PoseGraph(poses=est, edges=edges, fixed_node=0)
    return graph, gt
