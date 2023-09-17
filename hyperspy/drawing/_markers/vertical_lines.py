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
    """A set of Vertical Line Markers"""

    _key = "offsets"

    def __init__(self, offsets, **kwargs):
        """
        Initialize the set of Vertical Line Markers.

        Parameters
        ----------
        x: [n]
            Positions of the markers
        kwargs: dict
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
        >>> data = rng.random((25, 25, 100))
        >>> s = hs.signals.Signal1D(data)
        >>> offsets = np.array([10, 20, 40])

        >>> m = hs.plot.markers.VerticalLines(
        ...     offsets=offsets,
        ...     linewidth=3,
        ...     colors=['r', 'g', 'b'],
        ...     )

        >>> s.plot()
        >>> s.add_marker(m)
        """
        if "transform" in kwargs or "offset_transform" in kwargs:
            raise ValueError(
                "Setting 'offset_transform' or 'transform' argument is not "
                "supported with the VerticalLines markers."
            )

        super().__init__(
            collection=LineCollection,
            offsets=offsets,
            offsets_transform="display",
            transform="xaxis",  # so that the markers span the whole y-axis
            **kwargs
        )

    def get_data_position(self, get_static_kwargs=True):
        kwargs = super().get_data_position(get_static_kwargs=get_static_kwargs)
        x_pos = kwargs.pop("offsets")
        kwargs["segments"] = np.array([[[x, 0], [x, 1]] for x in x_pos])
        return kwargs

    def _to_dictionary(self):
        class_name = self.__class__.__name__
        marker_dict = {
            "class": class_name,
            "name": self.name,
            "plot_on_signal": self._plot_on_signal,
            "kwargs": self.kwargs,
        }
        return marker_dict
