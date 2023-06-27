# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

"""Model functions.


The model module contains the following submodules:

:mod:`~.components1d`
    1D components for HyperSpy model.

:mod:`~.components2d`
    2D components for HyperSpy model.

"""

import hyperspy.components1d as components1D
import hyperspy.components2d as components2D


__all__ = [
    'components1D',
    'components2D',
    ]


def __dir__():
    return sorted(__all__)
