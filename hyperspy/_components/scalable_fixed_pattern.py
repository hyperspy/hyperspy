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
from scipy.interpolate import make_interp_spline

from hyperspy.component import Component
from hyperspy.docstrings.parameters import FUNCTION_ND_DOCSTRING
from hyperspy.ui_registry import add_gui_method


@add_gui_method(toolkey="hyperspy.ScalableFixedPattern_Component")
class ScalableFixedPattern(Component):
    r"""Fixed pattern component with interpolation support.

    .. math::

        f(x) = a \cdot s \left(b \cdot x - x_0\right) + c

    ============ =============
     Variable     Parameter
    ============ =============
     :math:`a`    yscale
     :math:`b`    xscale
     :math:`x_0`  shift
    ============ =============

    Parameters
    ----------
    yscale : float
        The scaling factor in y (intensity axis).
    xscale : float
        The scaling factor in x.
    shift : float
        The shift of the component
    interpolate : bool
        If False no interpolation is performed and only a y-scaled spectrum is
        returned.

    Attributes
    ----------
    yscale : :class:`~.component.Parameter`
        The scaling factor in y (intensity axis).
    xscale : :class:`~.component.Parameter`
        The scaling factor in x.
    shift : :class:`~.component.Parameter`
        The shift of the component
    interpolate : bool
        If False no interpolation is performed and only a y-scaled spectrum is
        returned.

    Methods
    -------
    prepare_interpolator

    Examples
    --------

    The fixed pattern is defined by a Signal1D of navigation 0 which must be
    provided to the ScalableFixedPattern constructor, e.g.:

    >>> s = hs.load('data.hspy') # doctest: +SKIP
    >>> my_fixed_pattern = hs.model.components1D.ScalableFixedPattern(s) # doctest: +SKIP

    """

    def __init__(self, signal1D, yscale=1.0, xscale=1.0, shift=0.0, interpolate=True):
        Component.__init__(self, ["yscale", "xscale", "shift"], ["yscale"])

        self._position = self.shift
        self._whitelist["signal1D"] = ("init,sig", signal1D)
        self._whitelist["interpolate"] = None
        self.signal = signal1D
        self.yscale.free = True
        self.yscale.value = yscale
        self.xscale.value = xscale
        self.shift.value = shift

        self.prepare_interpolator()
        # Options
        self.isbackground = True
        self.convolved = False
        self.interpolate = interpolate

    @property
    def interpolate(self):
        return self._interpolate

    @interpolate.setter
    def interpolate(self, value):
        self._interpolate = value
        self.xscale.free = value
        self.shift.free = value

    def prepare_interpolator(self, **kwargs):
        """Fine-tune the interpolation.

        Parameters
        ----------
        x : array
            The spectral axis of the fixed pattern
        **kwargs : dict
            Keywords argument are passed to
            :func:`scipy.interpolate.make_interp_spline`
        """

        self.f = make_interp_spline(
            self.signal.axes_manager.signal_axes[0].axis,
            self.signal.data.squeeze(),
            **kwargs,
        )

    def _function(self, x, xscale, yscale, shift):
        if self.interpolate is True:
            result = yscale * self.f(x * xscale - shift)
        else:
            result = yscale * self.signal.data
        axis = self.signal.axes_manager.signal_axes[0]
        if axis.is_binned:
            if axis.is_uniform:
                return result / axis.scale
            else:
                return result / np.gradient(axis.axis)
        else:
            return result

    def function(self, x):
        return self._function(x, self.xscale.value, self.yscale.value, self.shift.value)

    def function_nd(self, axis):
        """%s"""
        if self._is_navigation_multidimensional:
            x = axis[np.newaxis, :]
            xscale = self.xscale.map["values"][..., np.newaxis]
            yscale = self.yscale.map["values"][..., np.newaxis]
            shift = self.shift.map["values"][..., np.newaxis]
            return self._function(x, xscale, yscale, shift)
        else:
            return self.function(axis)

    function_nd.__doc__ %= FUNCTION_ND_DOCSTRING

    def grad_yscale(self, x):
        return self.function(x) / self.yscale.value
