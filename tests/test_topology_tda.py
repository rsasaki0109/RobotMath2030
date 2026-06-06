"""Tests for Ch.14 topology / TDA."""

from __future__ import annotations

from miniworlds.loop_cloud_world import circle_cloud, corridor_cloud, shuffled_scan, two_loop_cloud
from robotmath.topology import naive_component_count, topological_loop_count


def test_circle_has_one_loop():
    pts = shuffled_scan(circle_cloud(n=36, noise=0.01, seed=0), seed=1)
    assert topological_loop_count(pts) == 1


def test_corridor_has_no_loops():
    pts = shuffled_scan(corridor_cloud(n=40, seed=2), seed=3)
    assert topological_loop_count(pts) == 0


def test_two_loops_cloud_has_two_loops():
    pts = shuffled_scan(two_loop_cloud(n_per_loop=20, noise=0.01, seed=4), seed=5)
    assert topological_loop_count(pts) == 2


def test_naive_components_miss_loops_on_circle():
    pts = shuffled_scan(circle_cloud(n=32, noise=0.01, seed=6), seed=7)
    from robotmath.topology import rips_persistence

    diagram = rips_persistence(pts)
    loops = topological_loop_count(pts)
    components = naive_component_count(pts, epsilon=float(diagram.local_scale * 1.5))
    assert loops == 1
    assert components == 1
