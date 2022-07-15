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
# along with HyperSpy.  If not, see <https://www.gnu.org/licenses/#GPL>.

import hyperspy.roi
from hyperspy.roi import (
    CircleROI,
    Line2DROI,
    Point1DROI,
    Point2DROI,
    RectangularROI,
    SpanROI
    )


__doc__ = hyperspy.roi.__doc__


__all__ = [
    'CircleROI',
    'Line2DROI',
    'Point1DROI',
    'Point2DROI',
    'RectangularROI',
    'SpanROI',
    ]


def __dir__():
    return sorted(__all__)
