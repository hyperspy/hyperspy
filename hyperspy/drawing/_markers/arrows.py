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
from hyperspy.external.matplotlib.quiver import Quiver
from hyperspy.docstrings.markers import OFFSET_DOCSTRING

class Arrows(Markers):
    """A set of Arrow markers based on the matplotlib.quiver.Quiver class.
    """
    def __init__(self,
                 offsets,
                 dx,
                 dy,
                 **kwargs):
        """ Initialize the set of Arrows Markers.

        Parameters
        ----------
        %s
        dx: array-like
            The change in x for the arrows.
        dy: array-like
            The change in y for the arrows.
        angles:array-like
            The angles of the first axes, degrees CCW from the x-axis.
        kwargs:
            Additional keyword arguments are passed to matplotlib.quiver.Quiver.
        """
        super().__init__(collection_class=Quiver,
                         offsets=offsets,
                         U=dx,
                         V=dy,
                         **kwargs)
        self.name = "Arrows"

    __init__.__doc__ %= OFFSET_DOCSTRING

    def get_data_position(self,
                          get_static_kwargs=True):
        kwargs = super().get_data_position(get_static_kwargs=get_static_kwargs)
        return kwargs

    def _initialize_collection(self):
        current_kwds = self.get_data_position()
        U = current_kwds.pop("U")
        V = current_kwds.pop("V")
        C = current_kwds.pop("C", None)
        offsets = current_kwds.pop("offsets")
        X = offsets[:, 0]
        Y = offsets[:, 1]

        if C is None:
            args = (X, Y, U, V)
        else:
            args = (X, Y, U, V, C)

        self.collection = Quiver(self.ax, *args, scale=1, angles="xy", scale_units="xy", **current_kwds)
