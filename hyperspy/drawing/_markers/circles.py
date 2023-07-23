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
from matplotlib.collections import CircleCollection
from hyperspy.docstrings.markers import OFFSET_DOCSTRING


class Circles(Markers):
    """A set of Circle Markers
    """
    def __init__(self,
                 offsets,
                 sizes,
                 **kwargs):
        """ Initialize the set of Circle Markers.

        Parameters
        ----------
        %s
        sizes: array-like
            The size of the circles in points.
        kwargs:
            Additional keyword arguments are passed to matplotlib.collections.CircleCollection.
        """

        super().__init__(collection_class=CircleCollection,
                         offsets=offsets,
                         sizes=sizes,
                         facecolors='none',
                         **kwargs)
        self.name = "Circles"

    __init__.__doc__ %= OFFSET_DOCSTRING
