# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

"""Markers that can be added to `Signal` plots.

Example
-------

>>> import scipy.misc
>>> im = hs.signals.Signal2D(scipy.misc.ascent())
>>> m = hs.plot.markers.rectangle(x1=150, y1=100, x2=400, y2=400, color='red')
>>> im.add_marker(m)

"""

from hyperspy.drawing._markers.horizontal_line import \
    HorizontalLine as horizontal_line
from hyperspy.drawing._markers.horizontal_line_segment import \
    HorizontalLineSegment as horizontal_line_segment
from hyperspy.drawing._markers.line_segment import LineSegment as line_segment
from hyperspy.drawing._markers.point import Point as point
from hyperspy.drawing._markers.rectangle import Rectangle as rectangle
from hyperspy.drawing._markers.text import Text as text
from hyperspy.drawing._markers.vertical_line import \
    VerticalLine as vertical_line
from hyperspy.drawing._markers.vertical_line_segment import \
    VerticalLineSegment as vertical_line_segment
