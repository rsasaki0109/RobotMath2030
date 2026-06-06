"""Synthetic environments for chapter demos."""

from miniworlds.landmark_pose_world import landmark_pose_problem
from miniworlds.rmp_world import GOAL, OBSTACLE_CENTER, OBSTACLE_RADIUS, RMP_START
from miniworlds.grid_world import GridWorld
from miniworlds.mass_spring_world import bouncing_ball_rollout, noisy_mass_spring_rollout
from miniworlds.occupancy_map_world import (
    l2_grid_mse,
    map_pair_drift,
    map_pair_ghost,
    map_pair_identical,
    map_to_points,
    reference_map,
)
from miniworlds.point_cloud_world import l_shape_cloud, misaligned_pair, occupancy_samples
from miniworlds.pose_graph_world import square_loop_graph
from miniworlds.two_path_world import START, GOAL, generate_demonstrations

__all__ = [
    "square_loop_graph",
    "landmark_pose_problem",
    "RMP_START",
    "l_shape_cloud",
    "misaligned_pair",
    "occupancy_samples",
    "generate_demonstrations",
    "START",
    "GOAL",
    "reference_map",
    "map_to_points",
    "l2_grid_mse",
    "map_pair_identical",
    "map_pair_drift",
    "map_pair_ghost",
    "noisy_mass_spring_rollout",
    "bouncing_ball_rollout",
    "GridWorld",
]
