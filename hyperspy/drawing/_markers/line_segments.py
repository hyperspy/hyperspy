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

from hyperspy.drawing.markers import Markers
from matplotlib.collections import LineCollection


class Lines(Markers):
    """A set of Line Segments Markers
    """
    def __init__(self,
                 segments,
                 **kwargs):
        """ Initialize the set of Segments Markers.

        Parameters
        ----------
        segments: [n, 2, 2] array-like or ragged array with shape (n, 2,3) at every navigation position
            Defines the lines[[[x1,y1],[x2,y2]], ...] of the center of the ellipse.
        kwargs:
            Additional keyword arguments are passed to matplotlib.collections.LineCollection.
        """
        super().__init__(collection_class=LineCollection,
                         segments=segments,
                         **kwargs)
        self.name = "LineSegments"
