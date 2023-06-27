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

from hyperspy.drawing._markers.marker_collection import MarkerCollection
from matplotlib.collections import LineCollection
import numpy as np


class HorizontalLineCollection(MarkerCollection):
    def __init__(self,
                 **kwargs):
        """
        Initialize a Marker Collection.

        Parameters
        ----------
        offsets: [n]
            Positions of the markers
        args: tuple
            Arguments passed to the underlying marker collection. Any argument
            that is array-like and has `dtype=object` is assumed to be an iterating
            argument and is treated as such.
        kwargs: dict
            Keyword arguments passed to the underlying marker collection. Any argument
            that is array-like and has `dtype=object` is assumed to be an iterating
            argument and is treated as such.
        """
        # Data attributes
        MarkerCollection.__init__(self,
                                  collection_class=LineCollection,
                                  **kwargs)
        self.name = "LineCollection"

    def get_data_position(self,
                          get_static_kwargs=True):
        kwargs = super().get_data_position(get_static_kwargs=get_static_kwargs)
        x_extent = self.ax.get_xlim()
        if "segments" in kwargs:
            y_pos = kwargs["segments"]
            new_segments = np.array([[[x_extent[0]*2, -y*2], [x_extent[1]*2, -y*2]] for y in y_pos])
            kwargs["segments"] = new_segments
        else:
            raise ValueError("No segments found in kwargs")
        return kwargs


class VerticalLineCollection(MarkerCollection):
    def __init__(self,
                 **kwargs):
        """
        Initialize a Marker Collection.

        Parameters
        ----------
        offsets: [n]
            Positions of the markers
        args: tuple
            Arguments passed to the underlying marker collection. Any argument
            that is array-like and has `dtype=object` is assumed to be an iterating
            argument and is treated as such.
        kwargs: dict
            Keyword arguments passed to the underlying marker collection. Any argument
            that is array-like and has `dtype=object` is assumed to be an iterating
            argument and is treated as such.
        """
        # Data attributes
        MarkerCollection.__init__(self,
                                  collection_class=LineCollection,
                                  **kwargs)
        self.name = "LineCollection"

    def get_data_position(self,
                          get_static_kwargs=True):
        kwargs = super().get_data_position(get_static_kwargs=get_static_kwargs)
        y_extent = self.ax.get_ylim()
        if "segments" in kwargs:
            x_pos = kwargs["segments"]
            new_segments = np.array([[[x*2, -y_extent[0]*2], [x*2, -y_extent[1]*2]] for x in x_pos])
            kwargs["segments"] = new_segments
        else:
            raise ValueError("No segments found in kwargs")
        return kwargs
