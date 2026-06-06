"""Synthetic occupancy maps for map evaluation demos."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def empty_grid(width: int = 32, height: int = 32) -> Array:
    return np.zeros((height, width), dtype=float)


def add_rect(grid: Array, x0: int, y0: int, x1: int, y1: int, value: float = 1.0) -> Array:
    out = grid.copy()
    out[y0:y1, x0:x1] = value
    return out


def reference_map(width: int = 32, height: int = 32) -> Array:
    """Simple indoor-like occupancy: border walls + two obstacle blobs."""
    grid = empty_grid(width, height)
    grid[0, :] = 1.0
    grid[-1, :] = 1.0
    grid[:, 0] = 1.0
    grid[:, -1] = 1.0
    grid = add_rect(grid, 8, 8, 12, 14, 1.0)
    grid = add_rect(grid, 20, 18, 26, 24, 1.0)
    return grid


def shift_map(grid: Array, dx: int, dy: int) -> Array:
    """Translate occupied cells — models localization / loop-closure drift."""
    out = empty_grid(grid.shape[1], grid.shape[0])
    # Keep outer walls fixed; shift interior occupied cells only.
    border = 1
    ys, xs = np.where(grid > 0.5)
    for y, x in zip(ys, xs):
        on_border = y <= border or y >= grid.shape[0] - 1 - border
        on_border |= x <= border or x >= grid.shape[1] - 1 - border
        ny, nx = (y, x) if on_border else (y + dy, x + dx)
        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
            out[ny, nx] = 1.0
    return out


def add_ghost_obstacle(grid: Array, x0: int, y0: int, x1: int, y1: int) -> Array:
    """Insert a spurious occupied region — false loop closure / dynamic object."""
    return add_rect(grid, x0, y0, x1, y1, 1.0)


def map_to_points(grid: Array) -> Array:
    """Occupied cell centers in normalized [0,1]^2 coordinates."""
    h, w = grid.shape
    ys, xs = np.where(grid > 0.5)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=float)
    return np.column_stack([xs / w, ys / h])


def occupied_count(grid: Array) -> int:
    return int(np.sum(grid > 0.5))


def l2_grid_mse(map_a: Array, map_b: Array) -> float:
    """Pixel-wise MSE on aligned occupancy grids."""
    if map_a.shape != map_b.shape:
        raise ValueError("maps must have the same shape")
    return float(np.mean((map_a - map_b) ** 2))


def map_pair_drift(seed: int = 0) -> tuple[Array, Array, str]:
    """Reference map vs horizontally shifted map."""
    ref = reference_map()
    shifted = shift_map(ref, dx=4, dy=0)
    return ref, shifted, "drift (4 cells right)"


def map_pair_ghost() -> tuple[Array, Array, str]:
    ref = reference_map()
    ghost = add_ghost_obstacle(ref, 14, 14, 18, 18)
    return ref, ghost, "ghost obstacle"


def map_pair_identical() -> tuple[Array, Array, str]:
    ref = reference_map()
    return ref, ref.copy(), "identical"
