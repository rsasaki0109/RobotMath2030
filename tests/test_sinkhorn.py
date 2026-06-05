"""Tests for Sinkhorn optimal transport."""

import numpy as np

from robotmath.optimal_transport import sinkhorn, sinkhorn2, wasserstein2_sinkhorn


def test_sinkhorn_marginals():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    Y = np.array([[0.1, 0.0], [1.1, 0.0], [0.0, 1.1]])
    n, m = X.shape[0], Y.shape[0]
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    C = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    P, _, _, gaps = sinkhorn(a, b, C, epsilon=0.05, max_iter=300)
    assert np.allclose(P.sum(), 1.0, atol=1e-5)
    assert np.allclose(P.sum(axis=1), a, atol=1e-4)
    assert np.allclose(P.sum(axis=0), b, atol=1e-4)
    assert gaps[-1] < 1e-3


def test_sinkhorn2_symmetry_zero():
    X = np.random.default_rng(0).normal(size=(10, 2))
    cost, P, _ = sinkhorn2(X, X, epsilon=0.08, max_iter=200)
    assert cost < 1e-2
    assert P.shape == (10, 10)


def test_wasserstein_increases_with_shift():
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    Y_near = np.array([[0.05, 0.0], [1.05, 0.0]])
    Y_far = np.array([[0.5, 0.0], [1.5, 0.0]])
    c_near, _, _ = wasserstein2_sinkhorn(X, Y_near, epsilon=0.05)
    c_far, _, _ = wasserstein2_sinkhorn(X, Y_far, epsilon=0.05)
    assert c_far > c_near
