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

from hyperspy.drawing.marker_collection import MarkerCollection
from copy import deepcopy

class RelativeCollection(MarkerCollection):
    """
    A Collection of Lines for faster plotting. This marker collection operates in a
    Just in Time fashion.  The marker positions are defined relative to the signal or
    as a function of the Figure size.

    This is good for adjusting markers
    """

    def __init__(self,
                 reference="data",
                 indexes=None,
                 **kwargs):
        """
        Initialize the relative marker collection

        Parameters
        ----------
        reference_framex : str
        """
        super().__init__(**kwargs)
        self.reference = reference
        if self.reference is "data_index" and indexes is not None:
            self.indexes = indexes
        elif self.reference is "data_index" and indexes is None:
            raise ValueError("Must supply indexes if reference is data_index")

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
            x_positions = kwds[key][..., 0].astype(int)
            y_positions = kwds[key][..., 1]
            new_y_positions = current_data[x_positions]*y_positions
            kwds[key][..., 1] = new_y_positions
        elif self.reference is "data_index":
            current_data = self.temp_signal(as_numpy=True)
            x_positions = self.indexes
            y_positions = kwds[key][..., 1]
            new_y_positions = current_data[x_positions]*y_positions
            kwds[key][..., 1] = new_y_positions
        else:
            raise ValueError("reference must be 'data' or 'data_index'")
        return kwds

    def update(self):
        if self.is_iterating:
            kwds = deepcopy(self.get_data_position(get_static_kwargs=False))
        else:
            kwds = deepcopy(self.kwargs)
        if "offsets" in kwds:
            kwds = self._scale_kwarg(kwds, "offsets")
        if "segments" in kwds:
            kwds = self._scale_kwarg(kwds, "segments")
        self.collection.set(**kwds)
