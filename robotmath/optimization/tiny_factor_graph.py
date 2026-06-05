"""Tiny pose graph optimizer — readable reference, not production SLAM."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robotmath.lie import se2
from robotmath.optimization.gauss_newton import gauss_newton_step, numerical_jacobian


@dataclass
class PoseGraphEdge:
    i: int
    j: int
    measurement: np.ndarray  # SE(2) relative transform
    information: np.ndarray = field(default_factory=lambda: np.eye(3))


@dataclass
class PoseGraph:
    """Pose graph with SE(2) nodes and relative pose edges."""

    poses: list[np.ndarray]
    edges: list[PoseGraphEdge]
    fixed_node: int = 0

    def copy(self) -> "PoseGraph":
        return PoseGraph(
            poses=[T.copy() for T in self.poses],
            edges=[
                PoseGraphEdge(e.i, e.j, e.measurement.copy(), e.information.copy())
                for e in self.edges
            ],
            fixed_node=self.fixed_node,
        )

    def _state_vector(self) -> np.ndarray:
        parts: list[float] = []
        for k, T in enumerate(self.poses):
            if k == self.fixed_node:
                continue
            parts.extend(se2.to_xytheta(T))
        return np.array(parts, dtype=float)

    def _unpack(self, x: np.ndarray) -> list[np.ndarray]:
        poses = [T.copy() for T in self.poses]
        idx = 0
        for k in range(len(poses)):
            if k == self.fixed_node:
                continue
            poses[k] = se2.from_xytheta(x[idx], x[idx + 1], x[idx + 2])
            idx += 3
        return poses

    def _stack_residuals(
        self,
        poses: list[np.ndarray],
        use_lie: bool = True,
    ) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for edge in self.edges:
            Ti, Tj = poses[edge.i], poses[edge.j]
            if use_lie:
                r = se2.residual(Ti, Tj, edge.measurement)
            else:
                r = se2.euclidean_residual(Ti, Tj, edge.measurement)
            sqrt_info = np.linalg.cholesky(edge.information)
            blocks.append(sqrt_info @ r)
        return np.concatenate(blocks)

    def optimize(
        self,
        max_iters: int = 30,
        use_lie: bool = True,
        damping: float = 1e-6,
        record_poses: bool = False,
    ) -> tuple["PoseGraph", list[float]] | tuple["PoseGraph", list[float], list[list[np.ndarray]]]:
        """Gauss-Newton pose graph optimization. Returns optimized graph and cost history."""
        x = self._state_vector()
        costs: list[float] = []
        pose_history: list[list[np.ndarray]] = []

        for _ in range(max_iters):
            poses = self._unpack(x)
            if record_poses:
                pose_history.append([T.copy() for T in poses])

            def residual_fn(state: np.ndarray) -> np.ndarray:
                return self._stack_residuals(self._unpack(state), use_lie=use_lie)

            r = residual_fn(x)
            cost = 0.5 * float(r @ r)
            costs.append(cost)
            J = numerical_jacobian(residual_fn, x)
            dx = gauss_newton_step(J, r, damping=damping)
            if np.linalg.norm(dx) < 1e-8:
                break
            x = x + dx

        optimized = self.copy()
        optimized.poses = self._unpack(x)
        if record_poses:
            pose_history.append([T.copy() for T in optimized.poses])
            return optimized, costs, pose_history
        return optimized, costs
