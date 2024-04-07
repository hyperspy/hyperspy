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

from hyperspy.docstrings.markers import OFFSET_DOCSTRING, UNITS_DOCSTRING
from hyperspy.drawing.markers import Markers
from hyperspy.external.matplotlib.collections import CircleCollection


class Circles(Markers):
    """A set of Circle Markers."""

    _position_key = "offsets"

    def __init__(
        self,
        offsets,
        sizes,
        offset_transform="data",
        units="x",
        facecolors="none",
        **kwargs,
    ):
        """
        Create a set of Circle Markers.

        Parameters
        ----------
        %s
        sizes : numpy.ndarray
            The size of the circles in units defined by the argument units.
        facecolors : matplotlib color or list of color
            Set the facecolor(s) of the markers. It can be a color
            (all patches have same color), or a sequence of colors;
            if it is a sequence the patches will cycle through the sequence.
            If c is 'none', the patch will not be filled.
        %s
        kwargs : dict
            Keyword arguments are passed to :class:`matplotlib.collections.CircleCollection`.
        """

        if kwargs.setdefault("transform", "display") != "display":
            raise ValueError(
                "The `transform` argument is not supported for Circles Markers. Instead, "
                "use the `offset_transform` argument to specify the transform of the "
                "`offsets` and use the `units` argument to specify transform of the "
                "`sizes` argument."
            )

        super().__init__(
            collection=CircleCollection,
            offsets=offsets,
            sizes=sizes,
            facecolors=facecolors,
            offset_transform=offset_transform,
            units=units,
            **kwargs,
        )

    __init__.__doc__ %= (OFFSET_DOCSTRING, UNITS_DOCSTRING)
