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

import numpy as np
import math

from hyperspy.component import Component

sqrt2pi = math.sqrt(2 * math.pi)


def voigt(x, FWHM=1, gamma=1, center=0, scale=1):
    """Voigt lineshape.

    The voigt peak is the convolution of a Lorentz peak with a Gaussian peak.

    The formula used to calculate this is::

        z(x) = (x + 1j gamma) / (sqrt(2) sigma)
        w(z) = exp(-z**2) erfc(-1j z) / (sqrt(2 pi) sigma)

        V(x) = scale Re(w(z(x-center)))

    Parameters
    ----------
    gamma : real
       The half-width half-maximum of the Lorentzian
    FWHM : real
       The FWHM of the Gaussian
    center : real
       Location of the center of the peak
    scale : real
       Value at the highest point of the peak

    Notes
    -----
    Ref: W.I.F. David, J. Appl. Cryst. (1986). 19, 63-64

    adjusted to use stddev and HWHM rather than FWHM parameters

    """
    # wofz function = w(z) = Fad[d][e][y]eva function = exp(-z**2)erfc(-iz)
    from scipy.special import wofz
    sigma = FWHM / 2.3548200450309493
    z = (np.asarray(x) - center + 1j * gamma) / (sigma * math.sqrt(2))
    V = wofz(z) / (math.sqrt(2 * np.pi) * sigma)
    return scale * V.real


class Voigt(Component):

    """Voigt profile component with support for shirley background,
    non_isochromaticity,transmission_function corrections and spin orbit
    splitting specially suited for Photoemission spectroscopy data analysis.

    f(x) = G(x)*L(x) where G(x) is the Gaussian function and L(x) is the
    Lorentzian function

    Attributes
    ----------

    area : Parameter
    centre: Parameter
    FWHM : Parameter
    gamma : Parameter
    resolution : Parameter
    shirley_background : Parameter
    non_isochromaticity : Parameter
    transmission_function : Parameter
    spin_orbit_splitting : Bool
    spin_orbit_branching_ratio : float
    spin_orbit_splitting_energy : float

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
        """Estimate the voigt function by calculating the momenta the gaussian.

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

        >>> g = hs.model.components1D.Gaussian()
        >>> x = np.arange(-10,10, 0.01)
        >>> data = np.zeros((32,32,2000))
        >>> data[:] = g.function(x).reshape((1,1,2000))
        >>> s = hs.signals.Signal1D({'data' : data})
        >>> s.axes_manager.axes[-1].offset = -10
        >>> s.axes_manager.axes[-1].scale = 0.01
        >>> g.estimate_parameters(s, -10,10, False)

        """
        super(Voigt, self)._estimate_parameters(signal)
        axis = signal.axes_manager.signal_axes[0]

        energy2index = axis._get_index
        i1 = energy2index(E1) if energy2index(E1) else 0
        i2 = energy2index(E2) if energy2index(E2) else len(axis.axis) - 1
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

        center = np.sum(X.reshape(X_shape) * data, i
                        ) / np.sum(data, i)

        sigma = np.sqrt(np.abs(np.sum((X.reshape(X_shape) - center.reshape(
            center_shape)) ** 2 * data, i) / np.sum(data, i)))
        height = data.max(i)
        if only_current is True:
            self.centre.value = center
            self.FWHM.value = sigma * 2.3548200450309493
            self.area.value = height * sigma * sqrt2pi
            return True
        else:
            if self.area.map is None:
                self.create_arrays(signal.axes_manager.navigation_shape)
            self.area.map['values'][:] = height * sigma * sqrt2pi
            self.area.map['is_set'][:] = True
            self.FWHM.map['values'][:] = sigma * 2.3548200450309493
            self.FWHM.map['is_set'][:] = True
            self.centre.map['values'][:] = center
            self.centre.map['is_set'][:] = True
            return True
