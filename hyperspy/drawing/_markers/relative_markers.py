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
from copy import deepcopy
import numpy as np


class RelativeMarkers(Markers):
    """A Set of Relative Markers which are plotted relative to the current data value or index.
    """

    def __init__(self,
                 reference="data",
                 indexes=None,
                 shift=None,
                 **kwargs):
        """
        Initialize the relative marker collection

        Parameters
        ----------
        reference: str
            "data" or "data_index".  If "data" the marker positions are scaled by the current
        indexes: array-like
            The indexes to use if reference is "data_index".  Useful for when you want to
            scale off of a point that isn't the value of the x index.
        shift: float
            The amount to shift the markers by.  This is useful for adjusting the markers
        kwargs:
            Additional keyword arguments are passed to the matplotlib collection
        """
        super().__init__(**kwargs)
        self.reference = reference
        self.shift=shift

        if self.reference is "data_index" and indexes is not None:
            self.indexes = indexes
        elif self.reference is "data_index" and indexes is None:
            raise ValueError("Must supply indexes if reference is data_index")

        if self.reference not in ["data", "data_index"]:
            raise ValueError("reference must be one of 'data', or 'data_index'")

    def get_data_position(self, get_static_kwargs=True):
        kwds = super().get_data_position()
        new_kwds = deepcopy(kwds)
        if "offsets" in new_kwds:
            new_kwds = self._scale_kwarg(new_kwds, "offsets")
        if "segments" in kwds:
            new_kwds = self._scale_kwarg(new_kwds, "segments")
        return new_kwds

    def _scale_kwarg(self, kwds, key):
        """
        Scale the kwarg by the current data.  This is useful for scaling the
        marker position by the current index or data value.

        When self.reference is "data" the kwarg is scaled by the current data value of the
        "offset" or "segments" key

        When self.reference is "data_index" the kwarg is scaled by the current data value of the
        "offset" or "segments" key and the given value of the index.  This is useful when you want
        to scale things by some value in the data that is not the same value.
        """
        if self.reference is "data":
            current_data = self.temp_signal(as_numpy=True)
            x_positions = kwds[key][..., 0]
            ax = self.axes_manager.signal_axes[0]
            indexes = np.round((x_positions - ax.offset)/ax.scale).astype(int)
            y_positions = kwds[key][..., 1]
            new_y_positions = current_data[indexes]*y_positions
        else:  # self.reference is "data_index"
            current_data = self.temp_signal(as_numpy=True)
            x_positions = self.indexes
            y_positions = kwds[key][..., 1]
            new_y_positions = current_data[x_positions]*y_positions

        if self.shift is not None:
            yrange = np.max(current_data)-np.min(current_data)
            new_y_positions = new_y_positions + self.shift*yrange
        kwds[key][..., 1] = new_y_positions
        return kwds

    def update(self):
        kwds = deepcopy(self.get_data_position(get_static_kwargs=False))
        self.collection.set(**kwds)
