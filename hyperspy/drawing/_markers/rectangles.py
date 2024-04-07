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

from hyperspy.docstrings.markers import (
    ANGLES_DOCSTRING,
    HEIGHTS_DOCSTRING,
    OFFSET_DOCSTRING,
    UNITS_DOCSTRING,
    WIDTHS_DOCSTRING,
)
from hyperspy.drawing.markers import Markers
from hyperspy.external.matplotlib.collections import RectangleCollection


class Rectangles(Markers):
    """A Collection of Rectangles Markers"""

    _position_key = "offsets"

    def __init__(
        self,
        offsets,
        widths,
        heights,
        angles=0,
        offset_transform="data",
        units="xy",
        **kwargs,
    ):
        """Initialize the set of Segments Markers.

        Parameters
        ----------
        %s
        %s
        %s
        %s
        %s
        kwargs:
            Additional keyword arguments are passed to
            :class:`hyperspy.external.matplotlib.collections.RectangleCollection`.
        """
        if kwargs.setdefault("transform", "display") != "display":
            raise ValueError(
                "The `transform` argument is not supported for Rectangle Markers. Instead, "
                "use the `offset_transform` argument to specify the transform of the "
                "`offsets` and use the `units` argument to specify transform of the "
                "`sizes` argument."
            )

        super().__init__(
            collection=RectangleCollection,
            offsets=offsets,
            widths=widths,
            heights=heights,
            angles=angles,
            offset_transform=offset_transform,
            units=units,
            **kwargs,
        )

    __init__.__doc__ %= (
        OFFSET_DOCSTRING,
        HEIGHTS_DOCSTRING,
        WIDTHS_DOCSTRING,
        ANGLES_DOCSTRING,
        UNITS_DOCSTRING,
    )
