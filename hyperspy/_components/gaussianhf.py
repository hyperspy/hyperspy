# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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

import math

import numpy as np

from hyperspy.component import Component

sqrt2pi = math.sqrt(2 * math.pi)
sigma2fwhm = 2 * math.sqrt(2 * math.log(2))


class Gaussian2(Component):

    """Normalized gaussian function component, with a fwhm parameter instead
    of the sigma parameter, and a height parameter instead of the A parameter 
    (scaling difference of sigma * sqrt(2*Pi)). This makes the parameter vs. 
    peak maximum independent of sigma, and thereby makes locking of the 
    parameter more viable. As long as there it no binning, the height parameter
    corresponds directly to the peak maximum, if not, the value is scaled by a 
    linear constant (signal_axis.scale).

    .. math::

        f(x) = \\frac{a}{\sqrt{2\pi c^{2}}}e^{-\\frac{\left(x-b\\right)^{2}}{2c^{2}}}

    +------------+-----------+
    | Parameter  | Attribute |
    +------------+-----------+
    +------------+-----------+
    |     a      |  height   |
    +------------+-----------+
    |     b      |  centre   |
    +------------+-----------+
    |     c      |   fwhm    |
    +------------+-----------+

    """

    def __init__(self, height=1., fwhm=1., centre=0.):
        Component.__init__(self, ['height', 'fwhm', 'centre'])
        self.height.value = height
        self.fwhm.value = fwhm
        self.centre.value = centre
        self._position = self.centre

        # Boundaries
        self.height.bmin = None
        self.height.bmax = None

        self.fwhm.bmin = 0.
        self.fwhm.bmax = None

        self.isbackground = False
        self.convolved = True

        # Gradients
        self.height.grad = self.grad_height
        self.fwhm.grad = self.grad_fwhm
        self.centre.grad = self.grad_centre

    def function(self, x):
        s = self.sigma
        c = self.centre.value
        h = self.height.value
        x1 = (x-c)
        return h * np.exp(-x1**2 / (2 * s**2))

    def grad_height(self, x):
        return self.function(x) / self.A

    def grad_fwhm(self, x):
        c = self.centre.value
        s2 = self.sigma ** 2
        A = self.A
        x1 = (x-c)
        return ((x1**2 * np.exp(-x1**2 / (2 * s2)) * A) / (sqrt2pi * s2 ** 2)) \
                - (np.exp(-x1**2 / (2 * s2)) * A) / (sqrt2pi * s2)

    def grad_centre(self, x):
        c = self.centre.value
        s = self.sigma
        A = self.A
        x1 = (x-c)
        return (x1 * np.exp(-x1 ** 2 / (2 * s**2)) * A) / (sqrt2pi * s**3)

    def estimate_parameters(self, signal, x1, x2, only_current=False):
        """Estimate the gaussian by calculating the momenta.

        Parameters
        ----------
        signal : Signal instance
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
        Adapted from http://www.scipy.org/Cookbook/FittingData

        Examples
        --------

        >>> import numpy as np
        >>> from hyperspy.hspy import *
        >>> from hyperspy.signals import Spectrum
        >>> g = components.Gaussian()
        >>> x = np.arange(-10,10, 0.01)
        >>> data = np.zeros((32,32,2000))
        >>> data[:] = g.function(x).reshape((1,1,2000))
        >>> s = Spectrum(data)
        >>> s.axes_manager._axes[-1].offset = -10
        >>> s.axes_manager._axes[-1].scale = 0.01
        >>> g.estimate_parameters(s, -10,10, False)

        """
        axis = signal.axes_manager.signal_axes[0]
        binned = signal.metadata.Signal.binned
        i1, i2 = axis.value_range_to_indices(x1, x2)
        X = axis.axis[i1:i2]
        if only_current is True:
            data = signal()[i1:i2]
            X_shape = (len(X),)
            i = 0
            center_shape = (1,)
        else:
            # TODO: write the rest of the code to estimate the parameters of
            # the full dataset
            i = axis.index_in_array
            data_gi = [slice(None), ] * len(signal.data.shape)
            data_gi[axis.index_in_array] = slice(i1, i2)
            data = signal.data[data_gi]
            X_shape = [1, ] * len(signal.data.shape)
            X_shape[axis.index_in_array] = data.shape[i]
            center_shape = list(data.shape)
            center_shape[i] = 1

        center = np.sum(X.reshape(X_shape) * data, i) / np.sum(data, i)

        sigma = np.sqrt(np.abs(np.sum((X.reshape(X_shape) - center.reshape(
            center_shape)) ** 2 * data, i) / np.sum(data, i)))
        fwhm = sigma * sigma2fwhm
        height = data.max(i)
        if only_current is True:
            self.centre.value = center
            self.fwhm.value = fwhm
            self.height.value = float(height)
            if binned is True:
                self.height.value /= axis.scale
            return True
        else:
            if self.height.map is None:
                self._create_arrays()
            self.height.map['values'][:] = height

            if binned is True:
                self.height.map['values'][:] /= axis.scale
            self.height.map['is_set'][:] = True
            self.fwhm.map['values'][:] = fwhm
            self.fwhm.map['is_set'][:] = True
            self.centre.map['values'][:] = center
            self.centre.map['is_set'][:] = True
            self.fetch_stored_values()
            return True

    @property
    def sigma(self):
        return self.fwhm.value / sigma2fwhm

    @sigma.setter
    def sigma(self, value):
        self.fwhm.value = value * sigma2fwhm

    @property
    def A(self):
        return self.height.value * self.sigma * sqrt2pi

    @A.setter
    def A(self, value):
        self.height.value = value / (self.sigma * sqrt2pi)

    def integral_as_signal(self):
        """
        Utility function to get gaussian integral as Signal
        """
        return self.height.as_signal() * self.fwhm.as_signal() * \
                sqrt2pi / sigma2fwhm
