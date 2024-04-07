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
from hyperspy.external.matplotlib.quiver import Quiver


class Arrows(Markers):
    """A set of Arrow markers based on the matplotlib.quiver.Quiver class."""

    _position_key = "offsets"

    def __init__(
        self, offsets, U, V, C=None, scale=1, angles="xy", scale_units="xy", **kwargs
    ):
        """
        Initialize the set of Arrows Markers.

        Parameters
        ----------
        %s
        U : array-like
            The change in x (horizontal) diraction for the arrows.
        V : array-like
            The change in y (vertical) diraction for the arrows.
        C : array-like or None
        kwargs : dict
            Keyword arguments are passed to :class:`matplotlib.quiver.Quiver`.
        """

        super().__init__(
            collection=Quiver,
            # iterating arguments
            offsets=offsets,
            U=U,
            V=V,
            C=C,
            **kwargs,
        )
        self._init_kwargs = dict(scale=scale, angles=angles, scale_units=scale_units)

    __init__.__doc__ %= OFFSET_DOCSTRING

    def _initialize_collection(self):
        if self._collection is None:
            kwds = self.get_current_kwargs()
            offsets = kwds["offsets"]
            X = offsets[:, 0]
            Y = offsets[:, 1]
            U, V, C = kwds["U"], kwds["V"], kwds["C"]

            if C is None:
                args = (X, Y, U, V)
            else:
                args = (X, Y, U, V, C)

            self._collection = self._collection_class(
                *args, offset_transform=self.ax.transData, **self._init_kwargs
            )

    def update(self):
        if self._is_iterating or "relative" in [
            self._offset_transform,
            self._transform,
        ]:
            kwds = self.get_current_kwargs(only_variable_length=True)
            # in case 'U', 'V', 'C' are not position dependent
            kwds.setdefault("U", self.kwargs["U"])
            kwds.setdefault("V", self.kwargs["V"])
            kwds.setdefault("C", self.kwargs["C"])
            self._collection.set_offsets(kwds["offsets"])
            # Need to use `set_UVC` and pass all 'U', 'V' and 'C' at once,
            # because matplotlib expect same shape
            UVC = {k: v for k, v in kwds.items() if k in ["U", "V", "C"]}
            self._collection.set_UVC(**UVC)
