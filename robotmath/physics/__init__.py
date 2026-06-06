"""Physics simulation utilities for RobotMath2030."""

from robotmath.physics.bouncing_ball import (
    BounceParams,
    simulate_bounce_hard,
    simulate_bounce_soft,
)
from robotmath.physics.mass_spring import (
    MassSpringParams,
    identify_autograd,
    identify_finite_diff,
    simulate,
    trajectory_mse,
)

__all__ = [
    "MassSpringParams",
    "simulate",
    "trajectory_mse",
    "identify_finite_diff",
    "identify_autograd",
    "BounceParams",
    "simulate_bounce_hard",
    "simulate_bounce_soft",
]
