"""Optimal transport utilities."""

from robotmath.optimal_transport.sinkhorn import (
    pairwise_sq_distances,
    sinkhorn,
    sinkhorn2,
    transport_cost,
)
from robotmath.optimal_transport.wasserstein import (
    compare_map_metrics,
    wasserstein2_sinkhorn,
    wasserstein_between_maps,
)

__all__ = [
    "pairwise_sq_distances",
    "sinkhorn",
    "sinkhorn2",
    "transport_cost",
    "wasserstein2_sinkhorn",
    "wasserstein_between_maps",
    "compare_map_metrics",
]
