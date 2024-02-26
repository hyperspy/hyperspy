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

from hyperspy.docstrings.markers import OFFSET_DOCSTRING
from hyperspy.drawing.markers import Markers
from hyperspy.external.matplotlib.collections import TextCollection


class Texts(Markers):
    """
    A set of text markers
    """

    _position_key = "offsets"

    def __init__(self, offsets, offset_transform="data", transform="display", **kwargs):
        """
        Initialize the set of Circle Markers.

        Parameters
        ----------
        %s
        sizes : array-like
            The size of the text in points.
        facecolors : (list of) matplotlib color
            Set the facecolor(s) of the markers. It can be a color
            (all patches have same color), or a sequence of colors;
            if it is a sequence the patches will cycle through the sequence.
            If c is 'none', the patch will not be filled.
        kwargs : dict
            Keyword arguments are passed to :class:`matplotlib.collections.CircleCollection`.
        """
        super().__init__(
            collection=TextCollection,
            offsets=offsets,
            offset_transform=offset_transform,
            transform=transform,
            **kwargs,
        )

    __init__.__doc__ %= OFFSET_DOCSTRING
