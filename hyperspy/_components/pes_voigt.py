# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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
import math

from hyperspy.component import Component
from hyperspy._components.gaussian import _estimate_gaussian_parameters


sqrt2pi = math.sqrt(2 * math.pi)
sigma2fwhm = 2 * math.sqrt(2 * math.log(2))


def voigt(x, FWHM=1, gamma=1, center=0, scale=1):
    r"""Voigt lineshape.

    The voigt peak is the convolution of a Lorentz peak with a Gaussian peak:

    .. math::
        f(x) = G(x) \cdot L(x)

    where :math:`G(x)` is the Gaussian function and :math:`L(x)` is the
    Lorentzian function. In this case using an approximate formula by David
    (see Notes). This approximation improves on the pseudo-Voigt function
    (linear combination instead of convolution of the distributions) and is,
    to a very good approximation, equivalent to a Voigt function:

    .. math::
        z(x) &= \frac{x + i \gamma}{\sqrt{2} \sigma} \\
        w(z) &= \frac{e^{-z^2} \text{erfc}(-i z)}{\sqrt{2 \pi} \sigma} \\
        f(x) &= A \cdot \Re\left\{ w \left[ z(x - x_0) \right] \right\}


    ============== =============
    Variable        Parameter
    ============== =============
    :math:`x_0`     center
    :math:`A`       scale
    :math:`\gamma`  gamma
    :math:`\sigma`  sigma
    ============== =============


    Parameters
    ----------
    gamma : real
       The half-width half-maximum of the Lorentzian.
    FWHM : real
       The FWHM = :math:`2 \sigma \sqrt{(2 \log(2))}` of the Gaussian.
    center : real
       Location of the center of the peak.
    scale : real
       Value at the highest point of the peak.

    Notes
    -----
    Ref: W.I.F. David, J. Appl. Cryst. (1986). 19, 63-64
    doi:10.1107/S0021889886089999
    """
    # wofz function = w(z) = Fad[d][e][y]eva function = exp(-z**2)erfc(-iz)
    from scipy.special import wofz
    sigma = FWHM / 2.3548200450309493
    z = (np.asarray(x) - center + 1j * gamma) / (sigma * math.sqrt(2))
    V = wofz(z) / (math.sqrt(2 * np.pi) * sigma)
    return scale * V.real


class Voigt(Component):
    # Legacy class to be removed in v2.0

    """This is the legacy Voigt profile component dedicated to photoemission
    spectroscopy data analysis that will renamed to `PESVoigt` in v2.0. To use
    the new Voigt lineshape component set `legacy=False`. See the
    documentation of :meth:`hyperspy._components.voigt.Voigt` for details on
    the usage of the new Voigt component and
    :meth:`hyperspy._components.pes_voigt.PESVoigt` for the legacy component.

    .. math::
        f(x) = G(x) \cdot L(x)

    where :math:`G(x)` is the Gaussian function and :math:`L(x)` is the
    Lorentzian function. This component uses an approximate formula by David
    (see Notes).


    Notes
    -----
    Uses an approximate formula according to
    W.I.F. David, J. Appl. Cryst. (1986). 19, 63-64.
    doi:10.1107/S0021889886089999
    """

    def __init__(self, legacy=True, **kwargs):
        self.legacy = legacy
        if legacy:
            from hyperspy.misc.utils import deprecation_warning
            msg = (
                "The API of the `Voigt` component will change in v2.0. "
                "This component will become `PESVoigt`. "
                "To use the new API set `legacy=False`.")
            deprecation_warning(msg)

            self.__class__ = PESVoigt
            self.__init__(**kwargs)
        else:
            from hyperspy._components.voigt import Voigt
            self.__class__ = Voigt
            self.__init__(**kwargs)

    @property
    def gwidth(self):
        if not self.legacy:
            return super().sigma.value * sigma2fwhm

    @gwidth.setter
    def gwidth(self, value):
        if not self.legacy:
            super(Voigt, self.__class__).sigma.value.fset(self, value
                                                                / sigma2fwhm)

    @property
    def FWHM(self):
        if not self.legacy:
            return super().sigma.value * sigma2fwhm

    @FWHM.setter
    def FWHM(self, value):
        if not self.legacy:
            super(Voigt, self.__class__).sigma.value.fset(self, value
                                                                / sigma2fwhm)

    @property
    def lwidth(self):
        if not self.legacy:
            return super().gamma.value * 2

    @lwidth.setter
    def lwidth(self, value):
        if not self.legacy:
            super(Voigt, self.__class__).gamma.value.fset(self, value / 2)


class PESVoigt(Component):

    """ Voigt component for photoemission spectroscopy data analysis.

    Voigt profile component with support for shirley background,
    non_isochromaticity, transmission_function corrections and spin orbit
    splitting specially suited for photoemission spectroscopy data analysis.

    .. math::
        f(x) = G(x) \cdot L(x)

    where :math:`G(x)` is the Gaussian function and :math:`L(x)` is the
    Lorentzian function. This component uses an approximate formula by David
    (see Notes).


    Parameters
    ----------

    area : Parameter
        Intensity below the peak.
    centre: Parameter
        Location of the maximum of the peak.
    FWHM : Parameter
        FWHM = :math:`2 \sigma \sqrt{(2 \log(2))}` of the Gaussian distribution.
    gamma : Parameter
        :math:`\gamma` of the Lorentzian distribution.
    resolution : Parameter
    shirley_background : Parameter
    non_isochromaticity : Parameter
    transmission_function : Parameter
    spin_orbit_splitting : Bool
    spin_orbit_branching_ratio : float
    spin_orbit_splitting_energy : float

    Notes
    -----
    Uses an approximate formula according to
    W.I.F. David, J. Appl. Cryst. (1986). 19, 63-64.
    doi:10.1107/S0021889886089999
    """

    def __init__(self):
        Component.__init__(self, (
            'area',
            'centre',
            'FWHM',
            'gamma',
            'resolution',
            'shirley_background',
            'non_isochromaticity',
            'transmission_function'))
        self._position = self.centre
        self.FWHM.value = 1
        self.gamma.value = 0
        self.area.value = 1
        self.resolution.value = 0
        self.resolution.free = False
        self.shirley_background.free = False
        self.non_isochromaticity.value = 0
        self.non_isochromaticity.free = False
        self.transmission_function.value = 1
        self.transmission_function.free = False
        # Options
        self.shirley_background.active = False
        self.spin_orbit_splitting = False
        self.spin_orbit_branching_ratio = 0.5
        self.spin_orbit_splitting_energy = 0.61
        self.isbackground = False
        self.convolved = True

    def function(self, x):
        area = self.area.value * self.transmission_function.value
        centre = self.centre.value
        ab = self.non_isochromaticity.value
        if self.resolution.value == 0:
            FWHM = self.FWHM.value
        else:
            FWHM = math.sqrt(self.FWHM.value ** 2 + self.resolution.value ** 2)
        gamma = self.gamma.value
        k = self.shirley_background.value
        f = voigt(x,
                  FWHM=FWHM, gamma=gamma, center=centre - ab, scale=area)
        if self.spin_orbit_splitting is True:
            ratio = self.spin_orbit_branching_ratio
            shift = self.spin_orbit_splitting_energy
            f2 = voigt(x, FWHM=FWHM, gamma=gamma,
                       center=centre - ab - shift, scale=area * ratio)
            f += f2
        if self.shirley_background.active:
            cf = np.cumsum(f)
            cf = cf[-1] - cf
            self.cf = cf
            return cf * k + f
        else:
            return f

    def estimate_parameters(self, signal, E1, E2, only_current=False):
        """Estimate the Voigt function by calculating the momenta of the
        Gaussian.

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
         : bool
            Exit status required for the :meth:`remove_background` function.

        Notes
        -----
        Adapted from http://www.scipy.org/Cookbook/FittingData

        Examples
        --------

        >>> g = hs.model.components1D.PESVoigt()
        >>> x = np.arange(-10, 10, 0.01)
        >>> data = np.zeros((32, 32, 2000))
        >>> data[:] = g.function(x).reshape((1, 1, 2000))
        >>> s = hs.signals.Signal1D(data)
        >>> s.axes_manager[-1].offset = -10
        >>> s.axes_manager[-1].scale = 0.01
        >>> g.estimate_parameters(s, -10, 10, False)

        """
        super(PESVoigt, self)._estimate_parameters(signal)
        axis = signal.axes_manager.signal_axes[0]
        centre, height, sigma = _estimate_gaussian_parameters(signal, E1, E2,
                                                              only_current)

        if only_current is True:
            self.centre.value = centre
            self.FWHM.value = sigma * sigma2fwhm
            self.area.value = height * sigma * sqrt2pi
            if self.binned:
                self.area.value /= axis.scale
            return True
        else:
            if self.area.map is None:
                self._create_arrays()
            self.area.map['values'][:] = height * sigma * sqrt2pi
            if self.binned:
                self.area.map['values'][:] /= axis.scale
            self.area.map['is_set'][:] = True
            self.FWHM.map['values'][:] = sigma * sigma2fwhm
            self.FWHM.map['is_set'][:] = True
            self.centre.map['values'][:] = centre
            self.centre.map['is_set'][:] = True
            self.fetch_stored_values()
            return True
