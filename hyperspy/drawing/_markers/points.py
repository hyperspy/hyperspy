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


class Points(Markers):
    """A Collection of points for faster plotting. A collection is a set of
    markers which have the same properties.

    This uses the same syntax and the MarkerCollection, where offsets are
    used to define the position of text on some plot.
    """
    def __init__(self,
                 offsets,
                 **kwargs):
        """ Initialize the set of Ellipse Markers.

        Parameters
        ----------
        offsets: [n, 2] array-like or ragged array with shape (n, 2) at every navigation position
            The positions [x, y] of the center of the ellipse.
        """
        super().__init__(collection_class=None,
                         offsets=offsets,
                         **kwargs)
        self.name = "Points"
