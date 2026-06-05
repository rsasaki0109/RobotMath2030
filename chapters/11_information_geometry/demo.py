#!/usr/bin/env python3
"""Chapter 11: Information geometry — natural gradient on a Gaussian policy."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from miniworlds.gaussian_policy_world import anisotropic_inv_cov, bad_policy_init, expert_demos
from robotmath.info_geom import NaturalGradConfig, optimize_mean_euclidean, optimize_mean_natural


def main():
    print("Chapter 11 — Information geometry / natural gradient")
    demos, target = expert_demos(n=512, seed=0)
    emp_mean = np.mean(demos, axis=0)
    inv_cov = anisotropic_inv_cov()
    mu0 = bad_policy_init()
    cfg = NaturalGradConfig(lr=0.15, max_steps=80)

    mu_euc, path_euc, loss_euc = optimize_mean_euclidean(mu0, demos, inv_cov, cfg)
    mu_nat, path_nat, loss_nat = optimize_mean_natural(mu0, demos, inv_cov, cfg)

    err_euc = float(np.linalg.norm(mu_euc - emp_mean))
    err_nat = float(np.linalg.norm(mu_nat - emp_mean))

    print(f"Init mean: {mu0}")
    print(f"Expert demo mean: {emp_mean}")
    print(f"Euclidean final error ||μ-μ̂||: {err_euc:.4f}, steps={len(path_euc)-1}")
    print(f"Natural final error ||μ-μ̂||:   {err_nat:.4f}, steps={len(path_nat)-1}")
    print(f"Final NLL — euclidean={loss_euc[-1]:.4f}, natural={loss_nat[-1]:.4f}")

    path_euc = np.array(path_euc)
    path_nat = np.array(path_nat)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.plot(path_euc[:, 0], path_euc[:, 1], "C3-o", markersize=3, label="Euclidean")
    ax.plot(path_nat[:, 0], path_nat[:, 1], "C0-o", markersize=3, label="Natural")
    ax.scatter(*mu0, c="k", s=40, label="init")
    ax.scatter(*emp_mean, c="gold", edgecolors="k", s=60, label="demo mean")
    ax.set_xlabel("μ₁ (large-variance action)")
    ax.set_ylabel("μ₂ (small-variance action)")
    ax.set_title("Parameter paths in policy space")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(loss_euc, "C3--", label="Euclidean")
    ax.semilogy(loss_nat, "C0-", label="Natural (Fisher)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Gaussian NLL")
    ax.set_title("Same lr — different geometry")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.scatter(demos[::8, 0], demos[::8, 1], s=8, alpha=0.35, label="expert demos")
    ax.scatter(*mu_euc, s=80, marker="x", c="C3", label="Euclidean μ")
    ax.scatter(*mu_nat, s=80, marker="*", c="C0", label="Natural μ")
    ax.set_xlabel("action dim 1")
    ax.set_ylabel("action dim 2")
    ax.set_title("Fitting a Gaussian policy to demos")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Why robot policy learning uses Fisher/natural gradients", fontsize=13)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
