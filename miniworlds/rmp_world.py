"""2D navigation setup for Riemannian motion policy demos."""

from __future__ import annotations

import numpy as np

from miniworlds.two_path_world import GOAL, OBSTACLE_CENTER, OBSTACLE_RADIUS, START

# Slightly off-center start breaks symmetry so radial obstacle tasks can deflect sideways.
RMP_START = np.array([0.48, START[1]], dtype=float)

__all__ = [
    "GOAL",
    "OBSTACLE_CENTER",
    "OBSTACLE_RADIUS",
    "RMP_START",
    "START",
]
