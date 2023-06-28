# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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

"""Markers that can be added to `Signal` plots.

Example
-------

>>> import skimage
>>> im = hs.signals.Signal2D(skimage.data.camera())
>>> m = hs.plot.markers.Rectangle(x1=150, y1=100, x2=400, y2=400, color='red')
>>> im.add_marker(m)

"""
from hyperspy.drawing._markers.arrow import Arrow
from hyperspy.drawing._markers.ellipse import Ellipse
from hyperspy.drawing._markers.horizontal_line import HorizontalLine
from hyperspy.drawing._markers.horizontal_line_segment import (
    HorizontalLineSegment,
    )
from hyperspy.drawing._markers.line_segment import LineSegment
from hyperspy.drawing._markers.point import Point
from hyperspy.drawing._markers.rectangle import Rectangle
from hyperspy.drawing._markers.text import Text
from hyperspy.drawing._markers.vertical_line import VerticalLine
from hyperspy.drawing._markers.vertical_line_segment import VerticalLineSegment

__all__ = [
    'Arrow',
    'Ellipse',
    'HorizontalLine',
    'HorizontalLineSegment',
    'LineSegment',
    'Point',
    'Rectangle',
    'Text',
    'VerticalLine',
    'VerticalLineSegment',

    ]


def __dir__():
    return sorted(__all__)

