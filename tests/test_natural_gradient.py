"""Tests for natural gradient vs Euclidean optimization."""

import numpy as np

from miniworlds.gaussian_policy_world import anisotropic_inv_cov, bad_policy_init, expert_demos
from robotmath.info_geom import (
    NaturalGradConfig,
    fisher_mean,
    optimize_mean_euclidean,
    optimize_mean_natural,
)


def test_fisher_matches_inv_cov():
    inv = anisotropic_inv_cov()
    assert np.allclose(fisher_mean(inv), inv)


def test_natural_reaches_target():
    demos, _ = expert_demos(n=512, seed=0)
    target = np.mean(demos, axis=0)
    inv = anisotropic_inv_cov()
    cfg = NaturalGradConfig(lr=0.15, max_steps=80)
    mu, _, losses = optimize_mean_natural(bad_policy_init(), demos, inv, cfg)
    assert np.linalg.norm(mu - target) < 0.05
    assert losses[-1] < losses[0]


def test_natural_beats_euclidean():
    demos, _ = expert_demos(n=512, seed=0)
    target = np.mean(demos, axis=0)
    inv = anisotropic_inv_cov()
    cfg = NaturalGradConfig(lr=0.15, max_steps=80)
    mu_e, _, _ = optimize_mean_euclidean(bad_policy_init(), demos, inv, cfg)
    mu_n, _, _ = optimize_mean_natural(bad_policy_init(), demos, inv, cfg)
    err_e = float(np.linalg.norm(mu_e - target))
    err_n = float(np.linalg.norm(mu_n - target))
    assert err_n < err_e
