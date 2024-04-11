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


import numpy as np

from hyperspy.component import Component
from hyperspy.docstrings.parameters import FUNCTION_ND_DOCSTRING


class Offset(Component):
    r"""Component to add a constant value in the y-axis.

    .. math::

        f(x) = k

    ============ =============
    Variable      Parameter
    ============ =============
    :math:`k`     offset
    ============ =============

    Parameters
    ----------
    offset : float
        The offset to be fitted

    """

    def __init__(self, offset=0.0):
        Component.__init__(self, ("offset",), ["offset"])
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
        signal : :class:`~.api.signals.Signal1D`
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
        super()._estimate_parameters(signal)
        axis = signal.axes_manager.signal_axes[0]
        i1, i2 = axis.value_range_to_indices(x1, x2)
        if axis.is_binned:
            # using the mean of the gradient for non-uniform axes is a best
            # guess to the scaling of binned signals for the estimation
            scaling_factor = (
                axis.scale
                if axis.is_uniform
                else np.mean(np.gradient(axis.axis), axis=-1)
            )

        if only_current is True:
            self.offset.value = signal._get_current_data()[i1:i2].mean()
            if axis.is_binned:
                self.offset.value /= scaling_factor
            return True
        else:
            if self.offset.map is None:
                self._create_arrays()
            dc = signal.data
            gi = [
                slice(None),
            ] * len(dc.shape)
            gi[axis.index_in_array] = slice(i1, i2)
            self.offset.map["values"][:] = dc[tuple(gi)].mean(axis.index_in_array)
            if axis.is_binned:
                self.offset.map["values"] /= scaling_factor
            self.offset.map["is_set"][:] = True
            self.fetch_stored_values()
            return True

    def function_nd(self, axis):
        """%s"""
        if self._is_navigation_multidimensional:
            x = axis[np.newaxis, :]
            o = self.offset.map["values"][..., np.newaxis]
        else:
            x = axis
            o = self.offset.value
        return self._function(x, o)

    function_nd.__doc__ %= FUNCTION_ND_DOCSTRING

    @property
    def _constant_term(self):
        "Get value of constant term of component"
        # First get currently constant parameters
        if self.offset.free:
            return 0
        else:
            return self.offset.value
