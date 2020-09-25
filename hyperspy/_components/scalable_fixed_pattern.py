# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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
from scipy.interpolate import interp1d

from hyperspy.component import Component
from hyperspy.ui_registry import add_gui_method
from hyperspy.docstrings.parameters import FUNCTION_ND_DOCSTRING


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


    The fixed pattern is defined by a single spectrum which must be provided to
    the ScalableFixedPattern constructor, e.g.:

    .. code-block:: ipython

        In [1]: s = load('my_spectrum.hspy')
        In [2]: my_fixed_pattern = components.ScalableFixedPattern(s))

    Parameters
    ----------

    yscale : Float
    xscale : Float
    shift : Float
    interpolate : Bool
        If False no interpolation is performed and only a y-scaled spectrum is
        returned.

    Methods
    -------

    prepare_interpolator : method to fine tune the interpolation

    """

    def __init__(self, signal1D, yscale=1.0, xscale=1.0,
                 shift=0.0, interpolate=True):

        Component.__init__(self, ['yscale', 'xscale', 'shift'])

        self._position = self.shift
        self._whitelist['signal1D'] = ('init,sig', signal1D)
        self._whitelist['interpolate'] = None
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

    def prepare_interpolator(self, kind='linear', fill_value=0, **kwargs):
        """Prepare interpolation.

        Parameters
        ----------
        x : array
            The spectral axis of the fixed pattern
        kind : str or int, optional
            Specifies the kind of interpolation as a string
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic')
            or as an integer specifying the order of the spline interpolator
            to use. Default is 'linear'.

        fill_value : float, optional
            If provided, then this value will be used to fill in for requested
            points outside of the data range. If not provided, then the default
            is NaN.

        Notes
        -----
        Any extra keyword argument is passed to `scipy.interpolate.interp1d`

        """

        self.f = interp1d(
            self.signal.axes_manager.signal_axes[0].axis,
            self.signal.data.squeeze(),
            kind=kind,
            bounds_error=False,
            fill_value=fill_value,
            **kwargs)

    def _function(self, x, xscale, yscale, shift):
        if self.interpolate is True:
            result = yscale * self.f(x * xscale - shift)
        else:
            result = yscale * self.signal.data
        if self.signal.metadata.Signal.binned is True:
            return result / self.signal.axes_manager.signal_axes[0].scale
        else:
            return result

    def function(self, x):
        return self._function(x, self.xscale.value, self.yscale.value,
                              self.shift.value)

    def function_nd(self, axis):
        """%s

        """
        if self._is_navigation_multidimensional:
            x = axis[np.newaxis, :]
            xscale = self.xscale.map['values'][..., np.newaxis]
            yscale = self.yscale.map['values'][..., np.newaxis]
            shift = self.shift.map['values'][..., np.newaxis]
            return self._function(x, xscale, yscale, shift)
        else:
            return self.function(axis)

    function_nd.__doc__ %= FUNCTION_ND_DOCSTRING

    def grad_yscale(self, x):
        return self.function(x) / self.yscale.value
