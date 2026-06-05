"""Wasserstein distances via entropic regularization."""

from __future__ import annotations

import numpy as np

from robotmath.optimal_transport.sinkhorn import sinkhorn2


def wasserstein2_sinkhorn(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float = 0.05,
    **kwargs,
) -> tuple[float, np.ndarray, list[float]]:
    """
    Approximate squared Wasserstein-2 distance between empirical distributions.

    Returns (W2^2 estimate, transport plan, convergence gaps).
    """
    cost, P, gaps = sinkhorn2(X, Y, epsilon=epsilon, **kwargs)
    return cost, P, gaps
