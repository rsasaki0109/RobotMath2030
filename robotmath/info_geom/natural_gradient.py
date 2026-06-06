"""Natural gradient vs Euclidean descent on a Gaussian policy manifold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NaturalGradConfig:
    lr: float = 0.15
    max_steps: int = 80
    tol: float = 1e-5


def gaussian_nll_mean(
    mu: np.ndarray,
    demos: np.ndarray,
    inv_cov: np.ndarray,
) -> float:
    """Negative log-likelihood for fixed covariance (constants omitted)."""
    mu = np.asarray(mu, dtype=float)
    demos = np.asarray(demos, dtype=float)
    diff = demos - mu
    return float(0.5 * np.mean(np.sum(diff @ inv_cov * diff, axis=1)))


def nll_mean_gradient(
    mu: np.ndarray,
    demos: np.ndarray,
    inv_cov: np.ndarray,
) -> np.ndarray:
    """Gradient of Gaussian NLL w.r.t. mean μ."""
    mu = np.asarray(mu, dtype=float)
    demos = np.asarray(demos, dtype=float)
    return inv_cov @ (mu - np.mean(demos, axis=0))


def fisher_mean(inv_cov: np.ndarray) -> np.ndarray:
    """Fisher information matrix for Gaussian mean with fixed covariance."""
    return np.asarray(inv_cov, dtype=float)


def optimize_mean_euclidean(
    mu_init: np.ndarray,
    demos: np.ndarray,
    inv_cov: np.ndarray,
    cfg: NaturalGradConfig | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    cfg = cfg or NaturalGradConfig()
    mu = np.asarray(mu_init, dtype=float).copy()
    path = [mu.copy()]
    losses: list[float] = []

    for _ in range(cfg.max_steps):
        loss = gaussian_nll_mean(mu, demos, inv_cov)
        losses.append(loss)
        grad = nll_mean_gradient(mu, demos, inv_cov)
        if np.linalg.norm(grad) < cfg.tol:
            break
        mu = mu - cfg.lr * grad
        path.append(mu.copy())

    return mu, path, losses


def optimize_mean_natural(
    mu_init: np.ndarray,
    demos: np.ndarray,
    inv_cov: np.ndarray,
    cfg: NaturalGradConfig | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    cfg = cfg or NaturalGradConfig()
    mu = np.asarray(mu_init, dtype=float).copy()
    fisher = fisher_mean(inv_cov)
    path = [mu.copy()]
    losses: list[float] = []

    for _ in range(cfg.max_steps):
        loss = gaussian_nll_mean(mu, demos, inv_cov)
        losses.append(loss)
        grad = nll_mean_gradient(mu, demos, inv_cov)
        if np.linalg.norm(grad) < cfg.tol:
            break
        nat_grad = np.linalg.solve(fisher, grad)
        mu = mu - cfg.lr * nat_grad
        path.append(mu.copy())

    return mu, path, losses
