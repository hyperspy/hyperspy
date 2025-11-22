"""
This module implements the Lowess function for nonparametric regression.
Functions:
lowess Fit a smooth nonparametric regression curve to a scatterplot.
For more information, see
William S. Cleveland: "Robust locally weighted regression and smoothing
scatterplots", Journal of the American Statistical Association, December 1979,
volume 74, number 368, pp. 829-836.
William S. Cleveland and Susan J. Devlin: "Locally weighted regression: An
approach to regression analysis by local fitting", Journal of the American
Statistical Association, September 1988, volume 83, number 403, pp. 596-610.
"""

import importlib

# ruff: noqa: F822

__all__ = [
    "lowess",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        return getattr(importlib.import_module("hyperspy.misc._lowess_smooth"), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
