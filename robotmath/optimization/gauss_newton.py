"""Gauss-Newton helpers."""

from __future__ import annotations

import numpy as np


def numerical_jacobian(
    residual_fn,
    x: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Central-difference Jacobian for residual vector r(x)."""
    x = np.asarray(x, dtype=float)
    r0 = np.asarray(residual_fn(x), dtype=float)
    m = r0.size
    n = x.size
    J = np.zeros((m, n), dtype=float)
    for k in range(n):
        step = np.zeros(n, dtype=float)
        step[k] = eps
        rp = np.asarray(residual_fn(x + step), dtype=float)
        rm = np.asarray(residual_fn(x - step), dtype=float)
        J[:, k] = (rp - rm) / (2.0 * eps)
    return J


def gauss_newton_step(J: np.ndarray, r: np.ndarray, damping: float = 1e-6) -> np.ndarray:
    """Solve (J^T J + λI) dx = -J^T r for Gauss-Newton update."""
    H = J.T @ J + damping * np.eye(J.shape[1])
    g = J.T @ r
    return np.linalg.solve(H, -g)
