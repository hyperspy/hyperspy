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

    _position_key = "verts"

    def __init__(self, verts,
                 offsets_transform="display", transform="data", **kwargs):
        """
        Initialize the set of Segments Markers.

        Parameters
        ----------
        verts : list of lists of lists or array of list of lists of lists
            The verts define the vertices of the polygons. Note that this can be
            a ragged list and as such it is not automatically cast to a numpy
            array as that would result in an array of objects.
            In the form [[[x1,y1], [x2,y2], ... [xn, yn]],[[x1,y1], [x2,y2], ...[xm,ym]], ...].
        **kwargs : dict
            Additional keyword arguments are passed to
            :py:class:`matplotlib.collections.PolyCollection`

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np

        >>> # Create a Signal2D with 2 navigation dimensions
        >>> data = np.ones((25, 25, 100, 100))
        >>> s = hs.signals.Signal2D(data)

        >>> poylgon1 = [[1, 1], [20, 20], [1, 20], [25, 5]]
        >>> poylgon2 = [[50, 60], [90, 40], [60, 40], [23, 60]]
        >>> verts = [poylgon1, poylgon2]

        >>> m = hs.plot.markers.Polygons(
        ...     verts=verts,
        ...     linewidth=3,
        ...     facecolors=('g',),
        ...     )

        >>> s.plot()
        >>> s.add_marker(m)

        Notes
        -----
        Unlike markers using ``offsets`` argument, the positions of the polygon
        are defined by the ``verts`` argument and the tranform specifying the
        coordinate system of the ``verts`` is ``transform``.

        """
        super().__init__(
            collection=PolyCollection,
            verts=verts,
            offsets_transform=offsets_transform,
            transform=transform,
            **kwargs
        )
