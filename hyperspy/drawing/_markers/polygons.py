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


class Polygons(Markers):
    """A Collection of Rectangles Markers"""

    marker_type = "Polygons"

    def __init__(self, verts, offsets_transform="display", transform="data", **kwargs):
        """Initialize the set of Segments Markers.

        Parameters
        ----------
        verts: list of lists of lists or array of list of lists of lists
        The verts define the vertices of the polygons. Note that this can be
        a ragged list and as such it is not automatically cast to a numpy array as that
        would result in an array of objects.
        In the form [[[x1,y1], [x2,y2], ... [xn, yn]],[[x1,y1], [x2,y2], ...[xm,ym]], ...].
        kwargs:
            Additional keyword arguments are passed to matplotlib.collections.PolyCollection.
        """
        super().__init__(
            collection_class=PolyCollection,
            verts=verts,
            offsets_transform=offsets_transform,
            transform=transform,
            **kwargs
        )
