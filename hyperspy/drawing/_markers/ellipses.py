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
from hyperspy.external.matplotlib.collections import EllipseCollection  # Update
from hyperspy.docstrings.markers import OFFSET_DOCSTRING

class Ellipses(Markers):
    """A set of Ellipse Markers
    """
    def __init__(self,
                 offsets,
                 heights,
                 widths,
                 angles=(0,),
                 **kwargs):
        """ Initialize the set of Ellipse Markers.

        Parameters
        ----------
        %s
        widths: array-like
            The lengths of the first axes (e.g., major axis lengths).
        heights: array-like
            The lengths of second axes.
        angles:array-like
            The angles of the first axes, degrees CCW from the x-axis.
        kwargs:
            Additional keyword arguments are passed to matplotlib.collections.EllipseCollection.
        """
        super().__init__(collection_class=EllipseCollection,
                         offsets=offsets,
                         heights=heights,
                         widths=widths,
                         angles=angles,
                         **kwargs)
        self.name = "Ellipses"

    __init__.__doc__ %= OFFSET_DOCSTRING
