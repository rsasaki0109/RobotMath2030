"""Synthetic environments for chapter demos."""

from miniworlds.point_cloud_world import l_shape_cloud, misaligned_pair, occupancy_samples
from miniworlds.pose_graph_world import square_loop_graph
from miniworlds.two_path_world import START, GOAL, generate_demonstrations

__all__ = [
    "square_loop_graph",
    "l_shape_cloud",
    "misaligned_pair",
    "occupancy_samples",
    "generate_demonstrations",
    "START",
    "GOAL",
]
