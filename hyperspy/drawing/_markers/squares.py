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

from matplotlib.collections import RegularPolyCollection

from hyperspy.docstrings.markers import OFFSET_DOCSTRING
from hyperspy.drawing.markers import Markers


class Squares(Markers):
    """
    A Collection of square markers using
    :py:class`matplotlib.collections.RegularPolyCollection`.
    """
    marker_type = "Squares"

    def __init__(self, offsets, sizes, **kwargs):
        """
        Initialize the set of square Markers.

        Parameters
        ----------
        %s
        sizes : Array-like
            The size of the squares.
        """
        # TODO: check the units of sizes
        super().__init__(collection_class=RegularPolyCollection,
                         offsets=offsets,
                         sizes=sizes,
                         numsides=4,
                         **kwargs)
        self.name = self.__class__.__name__

    __init__.__doc__ %= OFFSET_DOCSTRING
