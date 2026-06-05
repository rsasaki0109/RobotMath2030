"""Entropic optimal transport via Sinkhorn iterations."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def pairwise_sq_distances(X: Array, Y: Array) -> Array:
    """Squared Euclidean distances between rows of X (n,d) and Y (m,d)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    return x2 + y2 - 2.0 * (X @ Y.T)


def sinkhorn(
    a: Array,
    b: Array,
    C: Array,
    epsilon: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> tuple[Array, Array, Array, list[float]]:
    """
    Sinkhorn-Knopp algorithm for entropic OT.

    Returns transport plan P, scaling vectors u, v, and dual gap history.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    C = np.asarray(C, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1D weight vectors")
    if C.shape != (a.size, b.size):
        raise ValueError("C shape must match (len(a), len(b))")

    K = np.exp(-C / epsilon)
    u = np.ones_like(a)
    v = np.ones_like(b)
    gaps: list[float] = []

    for _ in range(max_iter):
        u_prev = u.copy()
        u = a / (K @ v + 1e-16)
        v = b / (K.T @ u + 1e-16)
        P = u[:, None] * K * v[None, :]
        row_err = np.max(np.abs(P.sum(axis=1) - a))
        col_err = np.max(np.abs(P.sum(axis=0) - b))
        gaps.append(float(row_err + col_err))
        if np.max(np.abs(u - u_prev)) < tol and gaps[-1] < tol:
            break

    P = u[:, None] * K * v[None, :]
    return P, u, v, gaps


def sinkhorn2(
    X: Array,
    Y: Array,
    epsilon: float = 0.05,
    weights_x: Array | None = None,
    weights_y: Array | None = None,
    **kwargs,
) -> tuple[float, Array, list[float]]:
    """Sinkhorn distance sqrt(sum(P * C)) for equal-mass point clouds."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, m = X.shape[0], Y.shape[0]
    a = np.full(n, 1.0 / n) if weights_x is None else np.asarray(weights_x, dtype=float)
    b = np.full(m, 1.0 / m) if weights_y is None else np.asarray(weights_y, dtype=float)
    a = a / a.sum()
    b = b / b.sum()
    C = pairwise_sq_distances(X, Y)
    P, _, _, gaps = sinkhorn(a, b, C, epsilon=epsilon, **kwargs)
    cost = float(np.sum(P * C))
    return cost, P, gaps


def transport_cost(P: Array, C: Array) -> float:
    """Transport cost sum(P * C)."""
    return float(np.sum(P * C))
