# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

from matplotlib.collections import PolyCollection

from hyperspy.drawing.markers import Markers


class Polygons(Markers):
    """A Collection of Rectangles Markers"""

    _position_key = "verts"

    def __init__(self, verts, transform="data", **kwargs):
        """
        Initialize the set of Segments Markers.

        Parameters
        ----------
        verts : list of numpy.ndarray or list of list
            The verts define the vertices of the polygons. Note that this can be
            a ragged list and as such it is not automatically cast to a numpy
            array as that would result in an array of objects.
            In the form [[[x1,y1], [x2,y2], ... [xn, yn]],[[x1,y1], [x2,y2], ...[xm,ym]], ...].
        **kwargs : dict
            Additional keyword arguments are passed to
            :class:`matplotlib.collections.PolyCollection`

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> import numpy as np
        >>> # Create a Signal2D with 2 navigation dimensions
        >>> data = np.ones((25, 25, 100, 100))
        >>> s = hs.signals.Signal2D(data)
        >>> poylgon1 = [[1, 1], [20, 20], [1, 20], [25, 5]]
        >>> poylgon2 = [[50, 60], [90, 40], [60, 40], [23, 60]]
        >>> verts = [poylgon1, poylgon2]
        >>> # Create the markers
        >>> m = hs.plot.markers.Polygons(
        ...     verts=verts,
        ...     linewidth=3,
        ...     facecolors=('g',),
        ...     )
        >>> # Add the marker to the signal
        >>> s.plot()
        >>> s.add_marker(m)

        Notes
        -----
        Unlike markers using ``offsets`` argument, the positions of the polygon
        are defined by the ``verts`` argument and the tranform specifying the
        coordinate system of the ``verts`` is ``transform``.

        """
        if kwargs.setdefault("offset_transform", "display") != "display":
            raise ValueError(
                "The `offset_transform` argument is not supported for Polygons Markers. "
                "Instead, use the `transform` argument to specify the transform "
                "of the polygons."
            )

        super().__init__(
            collection=PolyCollection, verts=verts, transform=transform, **kwargs
        )
