# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np

from hyperspy.component import Component
from hyperspy.docstrings.parameters import FUNCTION_ND_DOCSTRING


class Offset(Component):

    """Component to add a constant value in the y-axis

    f(x) = k + x

    +------------+-----------+
    | Parameter  | Attribute |
    +------------+-----------+
    +------------+-----------+
    |     k      |  offset   |
    +------------+-----------+

    """

    def __init__(self, offset=0.):
        Component.__init__(self, ('offset',))
        self.offset.free = True
        self.offset.value = offset

        self.isbackground = True
        self.convolved = False

        # Gradients
        self.offset.grad = self.grad_offset

    def function(self, x):
        return self._function(x, self.offset.value)

    def _function(self, x, o):
        return np.ones_like(x) * o

    @staticmethod
    def grad_offset(x):
        return np.ones_like(x)

    def estimate_parameters(self, signal, x1, x2, only_current=False):
        """Estimate the parameters by the two area method

        Parameters
        ----------
        signal : BaseSignal instance
        x1 : float
            Defines the left limit of the spectral range to use for the
            estimation.
        x2 : float
            Defines the right limit of the spectral range to use for the
            estimation.

        only_current : bool
            If False estimates the parameters for the full dataset.

        Returns
        -------
        bool

        """
        super(Offset, self)._estimate_parameters(signal)
        axis = signal.axes_manager.signal_axes[0]
        i1, i2 = axis.value_range_to_indices(x1, x2)

        if only_current is True:
            self.offset.value = signal()[i1:i2].mean()
            if self.binned:
                self.offset.value /= axis.scale
            return True
        else:
            if self.offset.map is None:
                self._create_arrays()
            dc = signal.data
            gi = [slice(None), ] * len(dc.shape)
            gi[axis.index_in_array] = slice(i1, i2)
            self.offset.map['values'][:] = dc[tuple(
                gi)].mean(axis.index_in_array)
            if self.binned:
                self.offset.map['values'] /= axis.scale
            self.offset.map['is_set'][:] = True
            self.fetch_stored_values()
            return True

    def function_nd(self, axis):
        """%s

        """
        x = axis[np.newaxis, :]
        o = self.offset.map['values'][..., np.newaxis]
        return self._function(x, o)

    function_nd.__doc__ %= FUNCTION_ND_DOCSTRING
