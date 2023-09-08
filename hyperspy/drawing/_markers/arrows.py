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
    marker_type = "Arrows"
    def __init__(
            self,
            offsets,
            U,
            V,
            C=None,
            scale=1,
            angles="xy",
            scale_units="xy",
            **kwargs
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
            Keyword arguments are passed to :py:class:`matplotlib.quiver.Quiver`.
        """
        super().__init__(
            collection_class=Quiver,
            # iterating arguments
            offsets=offsets,
            U=U,
            V=V,
            C=C,
            **kwargs
        )
        self._init_kwargs = dict(scale=scale, angles=angles, scale_units=scale_units)

    __init__.__doc__ %= OFFSET_DOCSTRING

    def _initialize_collection(self):
        kwds = self.get_data_position()
        offsets = kwds["offsets"]
        X = offsets[:, 0]
        Y = offsets[:, 1]
        U, V, C = kwds['U'], kwds['V'], kwds['C']

        if C is None:
            args = (X, Y, U, V)
        else:
            args = (X, Y, U, V, C)

        self.collection = self.collection_class(
            *args, offset_transform=self.ax.transData, **self._init_kwargs)

    def update(self):
        if not self._is_iterating:
            return
        else:
            kwds = self.get_data_position(get_static_kwargs=False)
            # in case 'U', 'V', 'C' are not position dependent
            kwds.setdefault('U', self.kwargs['U'])
            kwds.setdefault('V', self.kwargs['V'])
            kwds.setdefault('C', self.kwargs['C'])
            self.collection.set_offsets(kwds['offsets'])
            # Need to use `set_UVC` and pass all 'U', 'V' and 'C' at once,
            # because matplotlib expect same shape
            UVC = {k:v for k, v in kwds.items() if k in ['U', 'V', 'C']}
            self.collection.set_UVC(**UVC)
