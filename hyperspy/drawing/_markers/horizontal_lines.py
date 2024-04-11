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

import copy

import numpy as np
from matplotlib.collections import LineCollection

from hyperspy.drawing.markers import Markers


class HorizontalLines(Markers):
    """A set of HorizontalLines markers"""

    _position_key = "offsets"
    _position_key_to_set = "segments"

    def __init__(self, offsets, **kwargs):
        """
        Initialize a set of HorizontalLines markers.

        Parameters
        ----------
        offsets : array-like
            Positions of the markers
        kwargs : dict
            Keyword arguments passed to the underlying marker collection. Any argument
            that is array-like and has `dtype=object` is assumed to be an iterating
            argument and is treated as such.

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np

        >>> # Create a Signal2D with 2 navigation dimensions
        >>> rng = np.random.default_rng(0)
        >>> data = rng.random((25, 25, 100)) * 100
        >>> s = hs.signals.Signal1D(data)
        >>> offsets = np.array([10, 20, 40])

        >>> m = hs.plot.markers.HorizontalLines(
        ...     offsets=offsets,
        ...     linewidth=4,
        ...     colors=['r', 'g', 'b'],
        ...     )

        >>> s.plot()
        >>> s.add_marker(m)
        """

        if (
            kwargs.setdefault("offset_transform", "display") != "display"
            or kwargs.setdefault("transform", "yaxis") != "yaxis"
        ):
            raise ValueError(
                "Setting 'offset_transform' or 'transform' argument is not "
                "supported with the HorizontalLines markers."
            )

        super().__init__(collection=LineCollection, offsets=offsets, **kwargs)

    def get_current_kwargs(self, only_variable_length=False):
        kwargs = super().get_current_kwargs(only_variable_length=only_variable_length)
        # Need to take a deepcopy to avoid changing `self.kwargs`
        kwds = copy.deepcopy(kwargs)
        kwds[self._position_key_to_set] = np.array(
            [[[0, y], [1, y]] for y in kwds.pop(self._position_key)]
        )
        return kwds
