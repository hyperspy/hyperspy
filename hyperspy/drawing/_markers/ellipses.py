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
from hyperspy.external.matplotlib.collections import EllipseCollection  # Update


class Ellipses(Markers):
    """A Collection of Ellipses for faster plotting. A collection is a set of
    markers which have the same properties.

    This uses the same syntax and the MarkerCollection, where offsets are
    used to define the position of text on some plot.
    """
    def __init__(self,
                 offsets,
                 heights,
                 widths,
                 angles=(0,),
                 **kwargs):
        """ Initialize the set of Ellipse Markers.

        Parameters
        ----------
        offsets: [n, 2] array-like or ragged array with shape (n, 2) at every navigation position
            The positions [x, y] of the center of the ellipse.
        widths: array-like
            The lengths of the first axes (e.g., major axis lengths).
        heights: array-like
            The lengths of second axes.
        angles:array-like
            The angles of the first axes, degrees CCW from the x-axis.
        """
        super().__init__(collection_class=EllipseCollection,
                         offsets=offsets,
                         heights=heights,
                         widths=widths,
                         angles=angles,
                         **kwargs)
        self.name = "Ellipses"
