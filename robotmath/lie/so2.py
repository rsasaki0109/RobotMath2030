"""SO(2): 2D rotations."""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def wrap_angle(theta: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (theta + np.pi) % TWO_PI - np.pi


def from_angle(theta: float) -> np.ndarray:
    """Rotation matrix R in SO(2) from angle theta."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def to_angle(R: np.ndarray) -> float:
    """Extract angle from SO(2) matrix."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def exp(omega: float) -> np.ndarray:
    """Exponential map so(2) -> SO(2). For SO(2), exp(omega) = R(omega)."""
    return from_angle(omega)


def log(R: np.ndarray) -> float:
    """Logarithm map SO(2) -> so(2)."""
    return to_angle(R)


def compose(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    return R1 @ R2


def inverse(R: np.ndarray) -> np.ndarray:
    return R.T
