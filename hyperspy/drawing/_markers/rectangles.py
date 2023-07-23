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
from matplotlib.collections import PolyCollection


class Rectangles(Markers):
    """A Collection of Rectangles for faster plotting. A collection is a set of
    markers which have the same properties.
    """
    def __init__(self,
                 rectangles,
                 **kwargs):
        """ Initialize the set of Segments Markers.

        Parameters
        ----------
        rectangles: [n, 4] array-like or ragged array with shape (n, 4) at every navigation position
            Defines the rectangle[[x1,y1,x2,y2], ...].
        kwargs:
            Additional keyword arguments are passed to matplotlib.collections.PolyCollection.
        """
        super().__init__(collection_class=PolyCollection,
                         rectangles=rectangles,
                         **kwargs)
        self.name = "Rectangles"

    def get_data_position(self, **kwargs):
        current_kwds = super().get_data_position(**kwargs)
        rect = current_kwds.pop("rectangles")
        verts = [[[x1, y1], [x1, y2], [x2, y2], [x2, y1]] for x1, y1, x2, y2 in rect]
        current_kwds["verts"] = verts
        return current_kwds

