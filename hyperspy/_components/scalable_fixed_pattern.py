# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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
import scipy

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
        Component.__init__(self, ["xscale", "yscale", "shift"], ["yscale"])

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

        self.f = scipy.interpolate.make_interp_spline(
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

    def function_nd(self, axis, parameters_values=None):
        """
        Calculate the component over given axes and with given parameter values.

        Parameters
        ----------
        axis : numpy.ndarray
            The axis onto which the component is calculated.
        %s

        Returns
        -------
        numpy.ndarray
            The component values.
        """
        if self._is_navigation_multidimensional:
            x = axis[np.newaxis, :]
            if parameters_values is None:
                parameters_values = [p.map["values"] for p in self.parameters]
            parameters_values = [p[..., np.newaxis] for p in parameters_values]
            return self._function(x, *parameters_values)
        else:
            return self.function(axis)

    function_nd.__doc__ %= FUNCTION_ND_DOCSTRING

    def grad_yscale(self, x):
        return self.function(x) / self.yscale.value

    def estimate_parameters(self, signal, x1, x2, only_current=False, yscale=False):
        super()._estimate_parameters(signal)

        if yscale is not False:
            axis = signal.axes_manager[-1]
            i1, i2 = axis.value_range_to_indices(x1, x2)
            signal_axis = axis.axis[i1:i2]
            component = np.asarray(
                self._function(signal_axis, 1.0, 1.0, 0.0),
                dtype=float
            )
            valid = np.isfinite(component) & (component != 0)

        def _estimate_yscale(data):
            data = np.asarray(data, dtype=float)
            ratio = np.full(data.shape, np.nan, dtype=float)
            np.divide(data, component, out=ratio, where=valid)
            value = np.nanmean(ratio, axis=-1)
            return np.where(np.isfinite(value), value, 1.0)

        if only_current:
            self.xscale.value = 1.0
            self.shift.value = 0.0
            if yscale is False:
                self.yscale.value = 1.0
            elif yscale is True:
                current_data = signal._get_current_data()[i1:i2]
                self.yscale.value = float(_estimate_yscale(current_data))
            else:
                self.yscale.value = yscale
        else:
            if self.xscale.map is None:
                self._create_arrays()
            self.xscale.map["values"][:] = 1.0
            self.xscale.map["is_set"][:] = True
            self.shift.map["values"][:] = 0.0
            self.shift.map["is_set"][:] = True
            if yscale is False:
                self.yscale.map["values"][:] = 1.0
                self.yscale.map["is_set"][:] = True
            elif yscale is True:
                data = np.moveaxis(signal.data, axis.index_in_array, -1)[..., i1:i2]
                self.yscale.map["values"][:] = _estimate_yscale(data) 
                self.yscale.map["is_set"][:] = True
            else:
                self.yscale.map["values"][:] = yscale
                self.yscale.map["is_set"][:] = True

        self.fetch_stored_values()
        return True

