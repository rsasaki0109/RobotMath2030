"""Manifold updates on SE(2): retraction vs naive projection."""

from __future__ import annotations

import numpy as np

from robotmath.lie import se2, so2


def is_so2(R: np.ndarray, atol: float = 1e-8) -> bool:
    R = np.asarray(R, dtype=float)
    if R.shape != (2, 2):
        return False
    ortho = np.allclose(R.T @ R, np.eye(2), atol=atol)
    det = np.isclose(np.linalg.det(R), 1.0, atol=atol)
    return bool(ortho and det)


def is_se2(T: np.ndarray, atol: float = 1e-8) -> bool:
    T = np.asarray(T, dtype=float)
    if T.shape != (3, 3):
        return False
    if not np.allclose(T[2], [0.0, 0.0, 1.0], atol=atol):
        return False
    return is_so2(T[:2, :2], atol=atol)


def project_to_so2(R: np.ndarray) -> np.ndarray:
    """Project a nearly-orthogonal 2x2 matrix back to SO(2) via SVD."""
    R = np.asarray(R, dtype=float)
    u, _, vt = np.linalg.svd(R)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def retraction_se2(T: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Group retraction: T' = T exp(xi)."""
    return se2.compose(T, se2.exp(np.asarray(xi, dtype=float)))


def projection_se2(T: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Naive additive update on (x, y, theta) — common but inconsistent at large angles."""
    x, y, theta = se2.to_xytheta(T)
    xi = np.asarray(xi, dtype=float)
    return se2.from_xytheta(
        x + float(xi[0]),
        y + float(xi[1]),
        so2.wrap_angle(theta + float(xi[2])),
    )


def matrix_projection_se2(T: np.ndarray, xi: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Add to rotation matrix entries, then SVD-project back to SO(2)."""
    xi = np.asarray(xi, dtype=float)
    r = T[:2, :2].copy()
    t = T[:2, 2].copy()
    generator = np.array([[0.0, -1.0], [1.0, 0.0]])
    r_delta = -step * float(xi[2]) * (generator @ r)
    r_new = project_to_so2(r + r_delta)
    t_new = t - step * xi[:2]
    out = np.eye(3, dtype=float)
    out[:2, :2] = r_new
    out[:2, 2] = t_new
    return out


def pose_landmark_cost(
    T: np.ndarray,
    landmarks_body: np.ndarray,
    landmarks_world: np.ndarray,
) -> float:
    err = 0.0
    for p, q in zip(landmarks_body, landmarks_world):
        pred = se2.action(T, p)
        diff = pred - q
        err += float(diff @ diff)
    return err


def numerical_tangent_gradient(
    T: np.ndarray,
    landmarks_body: np.ndarray,
    landmarks_world: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central difference in the tangent space at T."""
    grad = np.zeros(3, dtype=float)
    for k in range(3):
        step = np.zeros(3, dtype=float)
        step[k] = eps
        c_plus = pose_landmark_cost(
            retraction_se2(T, step), landmarks_body, landmarks_world
        )
        c_minus = pose_landmark_cost(
            retraction_se2(T, -step), landmarks_body, landmarks_world
        )
        grad[k] = (c_plus - c_minus) / (2.0 * eps)
    return grad


def pose_error(T: np.ndarray, T_ref: np.ndarray) -> float:
    r = se2.log(se2.between(T, T_ref))
    return float(r @ r)


def optimize_pose_landmarks(
    T_init: np.ndarray,
    landmarks_body: np.ndarray,
    landmarks_world: np.ndarray,
    method: str = "retraction",
    max_iters: int = 80,
    step: float = 0.35,
    tol: float = 1e-8,
) -> tuple[np.ndarray, list[float], list[np.ndarray]]:
    """Gradient descent on SE(2) with different update rules."""
    T = T_init.copy()
    costs: list[float] = []
    history: list[np.ndarray] = [T.copy()]

    for _ in range(max_iters):
        cost = pose_landmark_cost(T, landmarks_body, landmarks_world)
        costs.append(cost)
        grad = numerical_tangent_gradient(T, landmarks_body, landmarks_world)
        if np.linalg.norm(grad) < tol:
            break

        if method == "retraction":
            T = retraction_se2(T, -step * grad)
        elif method == "projection":
            T = projection_se2(T, -step * grad)
        elif method == "matrix_projection":
            T = matrix_projection_se2(T, grad, step=step)
        else:
            raise ValueError(f"unknown method: {method}")
        history.append(T.copy())

    return T, costs, history
