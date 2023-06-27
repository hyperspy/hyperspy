# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

from hyperspy._components.expression import Expression
from hyperspy._components.gaussian import _estimate_gaussian_parameters
from hyperspy.component import _get_scaling_factor
from hyperspy.misc.utils import is_binned # remove in v2.0

sqrt2pi = math.sqrt(2 * math.pi)
sigma2fwhm = 2 * math.sqrt(2 * math.log(2))


class GaussianHF(Expression):

    r"""Normalized gaussian function component, with a ``fwhm`` parameter
    instead of the ``sigma`` parameter, and a ``height`` parameter instead of
    the area parameter ``A`` (scaling difference of
    :math:`\sigma \sqrt{\left(2\pi\right)}`).
    This makes the parameter vs. peak maximum independent of :math:`\sigma`,
    and thereby makes locking of the parameter more viable. As long as there
    is no binning, the `height` parameter corresponds directly to the peak
    maximum, if not, the value is scaled by a linear constant
    (`signal_axis.scale`).

    .. math::

        f(x) = h\cdot\exp{\left[-\frac{4 \log{2}
            \left(x-c\right)^{2}}{W^{2}}\right]}

    ============= =============
     Variable      Parameter
    ============= =============
     :math:`h`     height
     :math:`W`     fwhm
     :math:`c`     centre
    ============= =============


    Parameters
    ----------
    height: float
        The height of the peak. If there is no binning, this corresponds
        directly to the maximum, otherwise the maximum divided by
        signal_axis.scale
    fwhm: float
        The full width half maximum value, i.e. the width of the gaussian
        at half the value of gaussian peak (at centre).
    centre: float
        Location of the gaussian maximum, also the mean position.
    **kwargs
        Extra keyword arguments are passed to the
        :py:class:`~._components.expression.Expression` component.

    Attributes
    ----------
    A : float
        Convenience attribute to get, set the area and defined for
        compatibility with `Gaussian` component.
    sigma : float
        Convenience attribute to get, set the width and defined for
        compatibility with `Gaussian` component.

    See also
    --------
    ~._components.gaussian.Gaussian

    """

    def __init__(self, height=1., fwhm=1., centre=0., module="numexpr",
                 **kwargs):
        super().__init__(
            expression="height * exp(-(x - centre)**2 * 4 * log(2)/fwhm**2)",
            name="GaussianHF",
            height=height,
            fwhm=fwhm,
            centre=centre,
            position="centre",
            module=module,
            autodoc=False,
            **kwargs,
        )

        # Boundaries
        self.height.bmin = 0.
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
        Adapted from https://scipy-cookbook.readthedocs.io/items/FittingData.html

        Examples
        --------

        >>> g = hs.model.components1D.GaussianHF()
        >>> x = np.arange(-10, 10, 0.01)
        >>> data = np.zeros((32, 32, 2000))
        >>> data[:] = g.function(x).reshape((1, 1, 2000))
        >>> s = hs.signals.Signal1D(data)
        >>> s.axes_manager[-1].offset = -10
        >>> s.axes_manager[-1].scale = 0.01
        >>> g.estimate_parameters(s, -10, 10, False)
        """

        super()._estimate_parameters(signal)
        axis = signal.axes_manager.signal_axes[0]
        centre, height, sigma = _estimate_gaussian_parameters(signal, x1, x2,
                                                              only_current)
        scaling_factor = _get_scaling_factor(signal, axis, centre)

        if only_current is True:
            self.centre.value = centre
            self.fwhm.value = sigma * sigma2fwhm
            self.height.value = float(height)
            if is_binned(signal):
            # in v2 replace by
            #if axis.is_binned:
                self.height.value /= scaling_factor
            return True
        else:
            if self.height.map is None:
                self._create_arrays()
            self.height.map['values'][:] = height
            if is_binned(signal):
            # in v2 replace by
            #if axis.is_binned:
                self.height.map['values'][:] /= scaling_factor
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
