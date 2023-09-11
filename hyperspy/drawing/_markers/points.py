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

from matplotlib.collections import CircleCollection

from hyperspy.drawing.markers import Markers
from hyperspy.docstrings.markers import OFFSET_DOCSTRING


class Points(Markers):
    """A set of Points Markers."""
    marker_type = "Points"

    def __init__(self, offsets, sizes=2,
                 offsets_transform="data",
                 transform="xaxis_scale",
                 **kwargs):
        """ Initialize the set of points Markers.

        Parameters
        ----------
        %s
        kwargs : dict
        	Keyword arguments are passed to :py:class:`matplotlib.collections.CircleCollection`
        """
        super().__init__(collection_class=CircleCollection,
                         offsets=offsets,
                         sizes=sizes,
                         offsets_transform=offsets_transform,
                         transform=transform,
                         **kwargs)

    __init__.__doc__ %= OFFSET_DOCSTRING
