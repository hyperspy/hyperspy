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
from hyperspy.drawing.markers import Markers
from hyperspy.drawing._markers.relative_markers import RelativeMarkers
from hyperspy.drawing._markers.texts import Texts, RelativeTextCollection
from hyperspy.drawing._markers.horizontal_lines import HorizontalLines
from hyperspy.drawing._markers.vertical_lines import VerticalLines
from hyperspy.drawing._markers.circles import Circles
from hyperspy.drawing._markers.ellipses import Ellipses
from hyperspy.drawing._markers.arrows import Arrows
from hyperspy.drawing._markers.points import Points
from hyperspy.drawing._markers.line_segments import Lines
from hyperspy.drawing._markers.rectangles import Rectangles


__all__ = ['Arrows',
           'Circles',
           'Ellipses',
           'HorizontalLines',
           'Lines',
           'Markers',
           'Points','Rectangles',
           'RelativeMarkers',
           'RelativeTextCollection',
           'Texts',
           'VerticalLines',
           ]


def __dir__():
    return sorted(__all__)

