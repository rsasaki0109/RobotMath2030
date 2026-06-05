#!/usr/bin/env python3
"""Chapter 01: Pose is not a vector — SE(2) intuition demo."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from robotmath.lie import se2, so2
from robotmath.viz import draw_pose


def demo_composition():
    T_a = se2.from_xytheta(1.0, 0.0, np.pi / 4)
    T_b = se2.from_xytheta(0.5, 0.0, np.pi / 2)
    T_ab = se2.compose(T_a, T_b)
    T_ba = se2.compose(T_b, T_a)

    fig, ax = plt.subplots(figsize=(6, 5))
    draw_pose(ax, se2.from_xytheta(0, 0, 0), color="gray", label="origin")
    draw_pose(ax, T_a, color="C0", label="A")
    draw_pose(ax, T_b, color="C1", label="B")
    draw_pose(ax, T_ab, color="C2", label="A then B")
    draw_pose(ax, T_ba, color="C3", label="B then A")
    ax.set_title("SE(2) composition is not commutative")
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def demo_exp_log():
    xi = np.array([0.3, 0.1, np.pi / 3])
    T = se2.exp(xi)
    xi_rec = se2.log(T)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    draw_pose(axes[0], se2.from_xytheta(0, 0, 0), color="gray")
    draw_pose(axes[0], T, color="C0", scale=0.5)
    axes[0].set_title(f"exp(xi), xi = {xi.round(2)}")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(["vx", "vy", "omega"], xi, alpha=0.5, label="xi")
    axes[1].bar(["vx", "vy", "omega"], xi_rec, alpha=0.5, label="log(exp(xi))")
    axes[1].set_title("exp / log round-trip")
    axes[1].legend()
    fig.tight_layout()
    return fig


def demo_euler_interpolation_failure():
    """Linear interpolation of Euler angles vs geodesic on SO(2)."""
    t_vals = np.linspace(0, 1, 50)
    theta_a, theta_b = np.deg2rad(170), np.deg2rad(-170)

    euclidean = [so2.wrap_angle(theta_a + t * (theta_b - theta_a)) for t in t_vals]
    geodesic = [so2.wrap_angle(theta_a + t * so2.wrap_angle(theta_b - theta_a)) for t in t_vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.rad2deg(euclidean), "C3--", label="Linear on angles (wrong path)")
    ax.plot(np.rad2deg(geodesic), "C0-", label="Geodesic on SO(2) (shortest turn)")
    ax.axhline(180, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(-180, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("interpolation parameter")
    ax.set_ylabel("heading [deg]")
    ax.set_title("Why pose is not a vector: 170° → -170°")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    print("Chapter 01 — Pose is not a vector")
    print("Close each figure window to continue.")
    demo_composition()
    demo_exp_log()
    demo_euler_interpolation_failure()
    plt.show()


if __name__ == "__main__":
    main()
