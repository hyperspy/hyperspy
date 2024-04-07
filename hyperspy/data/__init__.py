# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

from .artificial_data import (
    atomic_resolution_image,
    luminescence_signal,
    wave_image,
)
from .two_gaussians import two_gaussians

__all__ = [
    "atomic_resolution_image",
    "luminescence_signal",
    "two_gaussians",
    "wave_image",
]


def __dir__():
    return sorted(__all__)
