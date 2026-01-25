# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>

import importlib

# ruff: noqa: F822

__all__ = [
    "get_array_memory_size_in_GiB",
    "are_aligned",
    "homogenize_ndim",
    "_requires_linear_rebin",
    "numba_histogram",
    "numba_closest_index_round",
    "numba_closest_index_floor",
    "numba_closest_index_ceil",
    "round_half_towards_zero",
    "round_half_away_from_zero",
    "get_value_at_index",
    "rebin",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        return getattr(importlib.import_module("hyperspy.misc._array_tools"), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
