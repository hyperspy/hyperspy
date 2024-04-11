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

from matplotlib.collections import LineCollection

from hyperspy.drawing.markers import Markers


class Lines(Markers):
    """A set of Line Segments Markers."""

    _position_key = "segments"

    def __init__(self, segments, transform="data", **kwargs):
        """Initialize the set of Segments Markers.

        Parameters
        ----------
        segments : numpy.ndarray
            Must be with shape [n, 2, 2] ragged array with shape (n, 2, 3) at every navigation position.
            Defines the lines[[[x1,y1],[x2,y2]], ...] of the center of the ellipse.
        kwargs : dict
            Additional keyword arguments are passed to :class:`matplotlib.collections.LineCollection`.

        Notes
        -----
        Unlike markers using ``offsets`` argument, the positions of the segments
        are defined by the ``segments`` argument and the tranform specifying the
        coordinate system of the ``segments`` is ``transform``.

        """

        if kwargs.setdefault("offset_transform", "display") != "display":
            raise ValueError(
                "The `offset_transform` argument is not supported for Lines Markers. "
                "Instead, use the `transform` argument to specify the transform "
                "of the lines."
            )

        super().__init__(
            collection=LineCollection, segments=segments, transform=transform, **kwargs
        )
