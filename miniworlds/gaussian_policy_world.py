"""Toy Gaussian action policy for information-geometry demos."""

from __future__ import annotations

import numpy as np


def anisotropic_inv_cov() -> np.ndarray:
    """Ill-conditioned precision — models very different action units/scales."""
    variances = np.array([100.0, 0.01], dtype=float)
    return np.diag(1.0 / variances)


def expert_demos(
    n: int = 512,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expert actions a ~ N(0, Σ) in 2D.

    Returns (demos, target_mean).
    """
    rng = np.random.default_rng(seed)
    cov = np.diag([100.0, 0.01])
    demos = rng.multivariate_normal(np.zeros(2), cov, size=n)
    return demos.astype(float), np.zeros(2, dtype=float)


def bad_policy_init() -> np.ndarray:
    """Start far from expert in both coordinates."""
    return np.array([8.0, 0.12], dtype=float)
