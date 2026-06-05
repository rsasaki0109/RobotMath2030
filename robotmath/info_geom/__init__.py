"""Information geometry utilities."""

from robotmath.info_geom.natural_gradient import (
    NaturalGradConfig,
    fisher_mean,
    gaussian_nll_mean,
    nll_mean_gradient,
    optimize_mean_euclidean,
    optimize_mean_natural,
)

__all__ = [
    "NaturalGradConfig",
    "fisher_mean",
    "gaussian_nll_mean",
    "nll_mean_gradient",
    "optimize_mean_euclidean",
    "optimize_mean_natural",
]
