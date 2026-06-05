"""Tests for Lie group utilities."""

import numpy as np
import pytest

from robotmath.lie import se2, so2


def test_so2_roundtrip():
    for theta in [0.0, 0.5, np.pi - 0.1, -2.0]:
        R = so2.from_angle(theta)
        assert pytest.approx(so2.to_angle(R), abs=1e-10) == so2.wrap_angle(theta)


def test_se2_compose_inverse():
    T = se2.from_xytheta(1.2, -0.5, 0.7)
    I = se2.compose(T, se2.inverse(T))
    assert np.allclose(I, np.eye(3), atol=1e-10)


def test_se2_exp_log_roundtrip():
    xi = np.array([0.2, -0.1, 0.4])
    xi2 = se2.log(se2.exp(xi))
    assert np.allclose(xi, xi2, atol=1e-8)


def test_se2_between():
    Ti = se2.from_xytheta(1.0, 2.0, 0.3)
    Tj = se2.from_xytheta(3.0, 1.0, -0.5)
    Tij = se2.between(Ti, Tj)
    Tj_rec = se2.compose(Ti, Tij)
    assert np.allclose(Tj, Tj_rec, atol=1e-10)


def test_residual_zero_at_truth():
    Ti = se2.from_xytheta(0, 0, 0)
    Tj = se2.from_xytheta(1, 0, np.pi / 4)
    Tij = se2.between(Ti, Tj)
    r = se2.residual(Ti, Tj, Tij)
    assert np.allclose(r, 0.0, atol=1e-10)
