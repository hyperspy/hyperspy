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
from matplotlib.collections import LineCollection
import numpy as np


class VerticalLines(Markers):
    def __init__(self,
                 x,
                 **kwargs):
        """
        Initialize a Marker Collection.

        Parameters
        ----------
        x: [n]
            Positions of the markers
        kwargs: dict
            Keyword arguments passed to the underlying marker collection. Any argument
            that is array-like and has `dtype=object` is assumed to be an iterating
            argument and is treated as such.
        """
        # Data attributes
        Markers.__init__(self,
                         x=x,
                         collection_class=LineCollection,
                         **kwargs)
        self.name = "VerticalLines"

    def get_data_position(self,
                          get_static_kwargs=True):
        kwargs = super().get_data_position(get_static_kwargs=get_static_kwargs)
        y_extent = self.ax.get_ylim()
        x_pos = kwargs.pop("x")
        new_segments = np.array([[[x, y_extent[0]], [x, y_extent[1]]] for x in x_pos])
        kwargs["segments"] = new_segments
        return kwargs
