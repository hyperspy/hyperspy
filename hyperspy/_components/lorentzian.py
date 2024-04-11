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

import dask.array as da
import numpy as np

from hyperspy._components.expression import Expression
from hyperspy.component import _get_scaling_factor


def _estimate_lorentzian_parameters(signal, x1, x2, only_current):
    axis = signal.axes_manager.signal_axes[0]
    i1, i2 = axis.value_range_to_indices(x1, x2)
    X = axis.axis[i1:i2]

    if only_current is True:
        data = signal._get_current_data()[i1:i2]
        i = 0
        centre_shape = (1,)
    else:
        i = axis.index_in_array
        data_gi = [
            slice(None),
        ] * len(signal.data.shape)
        data_gi[axis.index_in_array] = slice(i1, i2)
        data = signal.data[tuple(data_gi)]
        centre_shape = list(data.shape)
        centre_shape[i] = 1

    cdf = np.cumsum(data, i)
    cdfnorm = cdf / np.max(cdf, i).reshape(centre_shape)

    icentre = np.argmin(abs(0.5 - cdfnorm), i)
    igamma1 = np.argmin(abs(0.75 - cdfnorm), i)
    igamma2 = np.argmin(abs(0.25 - cdfnorm), i)

    if isinstance(data, da.Array):
        icentre, igamma1, igamma2 = da.compute(icentre, igamma1, igamma2)

    centre = X[icentre]
    gamma = (X[igamma1] - X[igamma2]) / 2
    height = data.max(i)

    return centre, height, gamma


class Lorentzian(Expression):
    r"""Cauchy-Lorentz distribution (a.k.a. Lorentzian function) component.

    .. math::

        f(x)=\frac{A}{\pi}\left[\frac{\gamma}{\left(x-x_{0}\right)^{2}
            +\gamma^{2}}\right]

    ============== =============
    Variable        Parameter
    ============== =============
    :math:`A`       A
    :math:`\gamma`  gamma
    :math:`x_0`     centre
    ============== =============


    Parameters
    ----------
    A : float
        Area parameter, where :math:`A/(\gamma\pi)` is the maximum (height) of
        peak.
    gamma : float
        Scale parameter corresponding to the half-width-at-half-maximum of the
        peak, which corresponds to the interquartile spread.
    centre : float
        Location of the peak maximum.
    **kwargs
        Extra keyword arguments are passed to the
        :class:`~.api.model.components1D.Expression` component.


    For convenience the `fwhm` and `height` attributes can be used to get and set
    the full-with-half-maximum and height of the distribution, respectively.
    """

    def __init__(self, A=1.0, gamma=1.0, centre=0.0, module=None, **kwargs):
        # We use `_gamma` internally to workaround the use of the `gamma`
        # function in sympy
        super().__init__(
            expression="A / pi * (gamma_ / ((x - centre)**2 + gamma_**2))",
            name="Lorentzian",
            A=A,
            gamma=gamma,
            centre=centre,
            position="centre",
            module=module,
            autodoc=False,
            rename_pars={"gamma_": "gamma"},
            **kwargs,
        )

        # Boundaries
        self.A.bmin = 0.0
        self.A.bmax = None

        self.gamma.bmin = None
        self.gamma.bmax = None

        self.isbackground = False
        self.convolved = True

    def estimate_parameters(self, signal, x1, x2, only_current=False):
        """Estimate the Lorentzian by calculating the median (centre) and half
        the interquartile range (gamma).

        Note that an insufficient range will affect the accuracy of this
        method.

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

        Notes
        -----
        Adapted from gaussian.py and
        https://en.wikipedia.org/wiki/Cauchy_distribution

        Examples
        --------

        >>> g = hs.model.components1D.Lorentzian()
        >>> x = np.arange(-10, 10, 0.01)
        >>> data = np.zeros((32, 32, 2000))
        >>> data[:] = g.function(x).reshape((1, 1, 2000))
        >>> s = hs.signals.Signal1D(data)
        >>> s.axes_manager[-1].offset = -10
        >>> s.axes_manager[-1].scale = 0.01
        >>> g.estimate_parameters(s, -10, 10, False)
        True
        """

        super()._estimate_parameters(signal)
        axis = signal.axes_manager.signal_axes[0]
        centre, height, gamma = _estimate_lorentzian_parameters(
            signal, x1, x2, only_current
        )
        scaling_factor = _get_scaling_factor(signal, axis, centre)

        if only_current is True:
            self.centre.value = centre
            self.gamma.value = gamma
            self.A.value = height * gamma * np.pi
            if axis.is_binned:
                self.A.value /= scaling_factor
            return True
        else:
            if self.A.map is None:
                self._create_arrays()
            self.A.map["values"][:] = height * gamma * np.pi
            if axis.is_binned:
                self.A.map["values"] /= scaling_factor
            self.A.map["is_set"][:] = True
            self.gamma.map["values"][:] = gamma
            self.gamma.map["is_set"][:] = True
            self.centre.map["values"][:] = centre
            self.centre.map["is_set"][:] = True
            self.fetch_stored_values()
            return True

    @property
    def fwhm(self):
        return self.gamma.value * 2

    @fwhm.setter
    def fwhm(self, value):
        self.gamma.value = value / 2

    @property
    def height(self):
        return self.A.value / (self.gamma.value * np.pi)

    @height.setter
    def height(self, value):
        self.A.value = value * self.gamma.value * np.pi
