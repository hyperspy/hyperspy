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

import math

from hyperspy._components.expression import Expression
from hyperspy._components.gaussian import _estimate_gaussian_parameters

sqrt2pi = math.sqrt(2 * math.pi)
sigma2fwhm = 2 * math.sqrt(2 * math.log(2))


class GaussianHF(Expression):

    """Normalized gaussian function component, with a fwhm parameter instead
    of the sigma parameter, and a height parameter instead of the A parameter
    (scaling difference of sigma * sqrt(2*Pi)). This makes the parameter vs.
    peak maximum independent of sigma, and thereby makes locking of the
    parameter more viable. As long as there is no binning, the height parameter
    corresponds directly to the peak maximum, if not, the value is scaled by a
    linear constant (signal_axis.scale).

    .. math::

        f(x) = h \\sqrt{2\\pi}\\mathrm{exp}{\\left[-\\frac{4 \\log{2}\\left(x-c\\right)^{2}}{W^{2}}\\right]}


    Parameters
    -----------
        height: float
            The height of the peak. If there is no binning, this corresponds
            directly to the maximum, otherwise the maximum divided by
            signal_axis.scale
        centre: float
            Location of the gaussian maximum, also the mean position.
        fwhm: float
            The full width half maximum value, i.e. the width of the gaussian
            at half the value of gaussian peak (at centre).

    The helper properties `sigma` and `A` are also defined for compatibility
    with `Gaussian` component.

    See also
    --------
    hyperspy.components.Gaussian

    """

    def __init__(self, height=1., fwhm=1., centre=0.):
        super(GaussianHF, self).__init__(
            expression="height * exp(-(x - centre)**2 * 4 * log(2)/fwhm**2)",
            name="GaussianHF",
            height=height,
            fwhm=fwhm,
            centre=centre,
            position="centre",
            autodoc=False,
        )

        # Boundaries
        self.height.bmin = None
        self.height.bmax = None

        self.fwhm.bmin = 0.
        self.fwhm.bmax = None

        self.isbackground = False
        self.convolved = True

    def estimate_parameters(self, signal, x1, x2, only_current=False):
        """Estimate the gaussian by calculating the momenta.

        Parameters
        ----------
        signal : Signal1D instance
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

        >>> g = hs.model.components1D.GaussianHF()
        >>> x = np.arange(-10, 10, 0.01)
        >>> data = np.zeros((32, 32, 2000))
        >>> data[:] = g.function(x).reshape((1, 1, 2000))
        >>> s = hs.signals.Signal1D(data)
        >>> s.axes_manager._axes[-1].offset = -10
        >>> s.axes_manager._axes[-1].scale = 0.01
        >>> g.estimate_parameters(s, -10, 10, False)

        """

        binned = signal.metadata.Signal.binned
        axis = signal.axes_manager.signal_axes[0]
        centre, height, sigma = _estimate_gaussian_parameters(signal, x1, x2,
                                                              only_current)

        if only_current is True:
            self.centre.value = centre
            self.fwhm.value = sigma * sigma2fwhm
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
            self.fwhm.map['values'][:] = sigma * sigma2fwhm
            self.fwhm.map['is_set'][:] = True
            self.centre.map['values'][:] = centre
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
        Utility function to get gaussian integral as Signal1D
        """
        return (self.height.as_signal() * self.fwhm.as_signal() *
                sqrt2pi / sigma2fwhm)
