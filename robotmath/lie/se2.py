"""SE(2): 2D rigid transforms (translation + rotation)."""

from __future__ import annotations

import numpy as np

from robotmath.lie import so2

Array = np.ndarray


def from_xytheta(x: float, y: float, theta: float) -> Array:
    """Build 3x3 homogeneous transform T in SE(2)."""
    T = np.eye(3, dtype=float)
    T[:2, :2] = so2.from_angle(theta)
    T[:2, 2] = [x, y]
    return T


def to_xytheta(T: Array) -> tuple[float, float, float]:
    """Extract (x, y, theta) from SE(2) matrix."""
    x, y = T[0, 2], T[1, 2]
    theta = so2.to_angle(T[:2, :2])
    return float(x), float(y), float(theta)


def compose(T1: Array, T2: Array) -> Array:
    return T1 @ T2


def inverse(T: Array) -> Array:
    R = T[:2, :2]
    t = T[:2, 2]
    Ti = np.eye(3, dtype=float)
    Ti[:2, :2] = R.T
    Ti[:2, 2] = -R.T @ t
    return Ti


def action(T: Array, point: Array) -> Array:
    """Apply SE(2) transform to 2D point(s). point shape (2,) or (N, 2)."""
    p = np.asarray(point, dtype=float)
    if p.ndim == 1:
        return T[:2, :2] @ p + T[:2, 2]
    return (T[:2, :2] @ p.T).T + T[:2, 2]


def exp(xi: Array) -> Array:
    """Exponential map se(2) -> SE(2). xi = [vx, vy, omega]."""
    vx, vy, omega = float(xi[0]), float(xi[1]), float(xi[2])
    R = so2.from_angle(omega)
    if abs(omega) < 1e-8:
        t = np.array([vx, vy])
    else:
        s = np.sin(omega)
        c = np.cos(omega)
        a = s / omega
        b = (1.0 - c) / omega
        t = np.array([a * vx - b * vy, b * vx + a * vy])
    T = np.eye(3, dtype=float)
    T[:2, :2] = R
    T[:2, 2] = t
    return T


def log(T: Array) -> Array:
    """Logarithm map SE(2) -> se(2)."""
    R = T[:2, :2]
    t = T[:2, 2]
    omega = so2.log(R)
    if abs(omega) < 1e-8:
        v = t.copy()
    else:
        s = np.sin(omega)
        c = np.cos(omega)
        a = s / omega
        b = (1.0 - c) / omega
        det = a * a + b * b
        v = np.array([(a * t[0] + b * t[1]) / det, (-b * t[0] + a * t[1]) / det])
    return np.array([v[0], v[1], omega])


def between(T_i: Array, T_j: Array) -> Array:
    """Relative transform from frame i to j: T_ij = T_i^{-1} T_j."""
    return compose(inverse(T_i), T_j)


def residual(T_i: Array, T_j: Array, T_ij_meas: Array) -> Array:
    """Lie algebra residual for edge (i,j): log(T_ij_meas^{-1} T_i^{-1} T_j)."""
    T_err = compose(inverse(T_ij_meas), between(T_i, T_j))
    return log(T_err)


def euclidean_residual(T_i: Array, T_j: Array, T_ij_meas: Array) -> Array:
    """Naive Euclidean residual on (dx, dy, dtheta) — often wrong at large angles."""
    xi_meas = np.array(to_xytheta(T_ij_meas))
    xi_est = np.array(to_xytheta(between(T_i, T_j)))
    return xi_est - xi_meas
