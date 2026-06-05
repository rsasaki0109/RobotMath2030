"""Tests for SE(2) retraction vs projection updates."""

import numpy as np

from miniworlds.landmark_pose_world import landmark_pose_problem
from robotmath.lie import se2, so2
from robotmath.optimization.manifold_update import (
    is_se2,
    is_so2,
    matrix_projection_se2,
    optimize_pose_landmarks,
    pose_error,
    project_to_so2,
    projection_se2,
    retraction_se2,
)


def test_project_to_so2_fixes_scale():
    r = np.array([[1.1, -0.05], [0.08, 1.05]])
    r_proj = project_to_so2(r)
    assert is_so2(r_proj)


def test_retraction_stays_on_se2():
    T = se2.from_xytheta(0.4, -0.2, 1.2)
    xi = np.array([0.05, -0.03, -0.7])
    T_new = retraction_se2(T, xi)
    assert is_se2(T_new)


def test_projection_wraps_heading():
    T = se2.from_xytheta(0.0, 0.0, np.deg2rad(170.0))
    T_new = projection_se2(T, np.array([0.0, 0.0, np.deg2rad(20.0)]))
    _, _, theta = se2.to_xytheta(T_new)
    assert np.isclose(theta, np.deg2rad(-170.0), atol=1e-6)


def test_matrix_projection_restores_so2():
    T = se2.from_xytheta(0.2, 0.1, 0.8)
    T_new = matrix_projection_se2(T, np.array([0.01, -0.02, 0.5]), step=0.2)
    assert is_se2(T_new)


def test_retraction_beats_projection_on_landmarks():
    T_true, T_init, body, world = landmark_pose_problem(seed=0)
    T_ret, _, _ = optimize_pose_landmarks(
        T_init, body, world, method="retraction", step=0.1, max_iters=120
    )
    T_proj, _, _ = optimize_pose_landmarks(
        T_init, body, world, method="projection", step=0.1, max_iters=120
    )
    err_ret = pose_error(T_ret, T_true)
    err_proj = pose_error(T_proj, T_true)
    assert err_ret < 1e-3
    assert err_proj > err_ret
