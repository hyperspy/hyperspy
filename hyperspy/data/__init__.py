# -*- coding: utf-8 -*-
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
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

"""
The :mod:`hyperspy.api.data` module includes synthetic data signal.
"""

import importlib

__all__ = [
    "atomic_resolution_image",
    "luminescence_signal",
    "two_gaussians",
    "wave_image",
]


_import_mapping = {
    "atomic_resolution_image": "artificial_data",
    "luminescence_signal": "artificial_data",
    "two_gaussians": "two_gaussians",
    "wave_image": "artificial_data",
}


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        import_path = f"hyperspy.data._{_import_mapping.get(name)}"
        return getattr(importlib.import_module(import_path), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
