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
from hyperspy.external.matplotlib.collections import RectangleCollection


class Rectangles(Markers):
    """A Collection of Rectangles Markers
    """
    marker_type = "Rectangles"

    def __init__(self,
                 offsets,
                 widths,
                 heights,
                 offsets_transform="data",
                 transform="xaxis",
                 **kwargs):
        """ Initialize the set of Segments Markers.

        Parameters
        ----------
        rectangles: [n, 4] array-like or ragged array with shape (n, 4) at every navigation position
            Defines the rectangle[[x1,y1,x2,y2], ...].
        kwargs:
            Additional keyword arguments are passed to matplotlib.collections.PolyCollection.
        """
        super().__init__(collection_class=RectangleCollection,
                         offsets=offsets,
                         widths=widths,
                         heights=heights,
                         offsets_transform=offsets_transform,
                         transform=transform,
                         **kwargs)
