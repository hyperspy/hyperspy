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

import math

import dask.array as da
import numpy as np

from hyperspy._components.expression import Expression
from hyperspy.component import _get_scaling_factor

sqrt2pi = math.sqrt(2 * math.pi)
sigma2fwhm = 2 * math.sqrt(2 * math.log(2))


def _estimate_gaussian_parameters(signal, x1, x2, only_current):
    axis = signal.axes_manager.signal_axes[0]
    i1, i2 = axis.value_range_to_indices(x1, x2)
    X = axis.axis[i1:i2]

    if only_current is True:
        data = signal._get_current_data()[i1:i2]
        X_shape = (len(X),)
        i = 0
        centre_shape = (1,)
    else:
        i = axis.index_in_array
        data_gi = [
            slice(None),
        ] * len(signal.data.shape)
        data_gi[axis.index_in_array] = slice(i1, i2)
        data = signal.data[tuple(data_gi)]
        X_shape = [
            1,
        ] * len(signal.data.shape)
        X_shape[axis.index_in_array] = data.shape[i]
        centre_shape = list(data.shape)
        centre_shape[i] = 1

    centre = np.sum(X.reshape(X_shape) * data, i) / np.sum(data, i)
    sigma = np.sqrt(
        abs(
            np.sum((X.reshape(X_shape) - centre.reshape(centre_shape)) ** 2 * data, i)
            / np.sum(data, i)
        )
    )
    height = data.max(i)

    if isinstance(data, da.Array):
        return da.compute(centre, height, sigma)
    else:
        return centre, height, sigma


class Gaussian(Expression):
    r"""Normalized Gaussian function component.

    .. math::

        f(x) = \frac{A}{\sigma \sqrt{2\pi}}\exp\left[
               -\frac{\left(x-x_0\right)^{2}}{2\sigma^{2}}\right]

    ============== ===========
    Variable        Parameter
    ============== ===========
    :math:`A`       A
    :math:`\sigma`  sigma
    :math:`x_0`     centre
    ============== ===========


    Parameters
    ----------
    A : float
        Area, equals height scaled by :math:`\sigma\sqrt{(2\pi)}`.
        ``GaussianHF`` implements the Gaussian function with a height parameter
        corresponding to the peak height.
    sigma : float
        Scale parameter of the Gaussian distribution.
    centre : float
        Location of the Gaussian maximum (peak position).
    **kwargs
        Extra keyword arguments are passed to the
        :class:`~.api.model.components1D.Expression` component.

    Attributes
    ----------
    fwhm : float
        Convenience attribute to get and set the full width at half maximum.
    height : float
        Convenience attribute to get and set the height.


    See Also
    --------
    GaussianHF
    """

    def __init__(self, A=1.0, sigma=1.0, centre=0.0, module=None, **kwargs):
        super().__init__(
            expression="A * (1 / (sigma * sqrt(2*pi))) * exp(-(x - centre)**2 \
                        / (2 * sigma**2))",
            name="Gaussian",
            A=A,
            sigma=sigma,
            centre=centre,
            position="centre",
            module=module,
            autodoc=False,
            **kwargs,
        )

        # Boundaries
        self.A.bmin = 0.0
        self.A.bmax = None

        self.sigma.bmin = 0.0
        self.sigma.bmax = None

        self.isbackground = False
        self.convolved = True

    def estimate_parameters(self, signal, x1, x2, only_current=False):
        """Estimate the Gaussian by calculating the momenta.

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
        Adapted from https://scipy-cookbook.readthedocs.io/items/FittingData.html

        Examples
        --------

        >>> g = hs.model.components1D.Gaussian()
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
        centre, height, sigma = _estimate_gaussian_parameters(
            signal, x1, x2, only_current
        )
        scaling_factor = _get_scaling_factor(signal, axis, centre)

        if only_current is True:
            self.centre.value = centre
            self.sigma.value = sigma
            self.A.value = height * sigma * sqrt2pi
            if axis.is_binned:
                self.A.value /= scaling_factor
            return True
        else:
            if self.A.map is None:
                self._create_arrays()
            self.A.map["values"][:] = height * sigma * sqrt2pi
            if axis.is_binned:
                self.A.map["values"] /= scaling_factor
            self.A.map["is_set"][:] = True
            self.sigma.map["values"][:] = sigma
            self.sigma.map["is_set"][:] = True
            self.centre.map["values"][:] = centre
            self.centre.map["is_set"][:] = True
            self.fetch_stored_values()
            return True

    @property
    def fwhm(self):
        return self.sigma.value * sigma2fwhm

    @fwhm.setter
    def fwhm(self, value):
        self.sigma.value = value / sigma2fwhm

    @property
    def height(self):
        return self.A.value / (self.sigma.value * sqrt2pi)

    @height.setter
    def height(self, value):
        self.A.value = value * self.sigma.value * sqrt2pi
