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
from scipy import constants
from scipy.integrate import simps, cumtrapz

from hyperspy._signals.complex_signal1d import (ComplexSignal1D,
                                                LazyComplexSignal1D)
from hyperspy.misc.eels.tools import eels_constant


class DielectricFunction_mixin:

    _signal_type = "DielectricFunction"
    _alias_signal_types = ["dielectric function"]

    def get_number_of_effective_electrons(self, nat, cumulative=False):
        """Compute the number of effective electrons using the Bethe f-sum
        rule.

        The Bethe f-sum rule gives rise to two definitions of the effective
        number (see [Egerton2011]_):
        $n_{\mathrm{eff}}\left(-\Im\left(\epsilon^{-1}\right)\right)$ that
        we'll call neff1 and
        $n_{\mathrm{eff}}\left(\epsilon_{2}\right)$ that we'll call neff2. This
        method computes both.

        Parameters
        ----------
        nat: float
            Number of atoms (or molecules) per unit volume of the
            sample.
        cumulative : bool
            If False calculate the number of effective electrons up to the
            higher energy-loss of the spectrum. If True, calculate the
            number of effective electrons as a function of the energy-loss up
            to the higher energy-loss of the spectrum. *True is only supported
            by SciPy newer than 0.13.2*.

        Returns
        -------
        neff1, neff2: Signal1D
            Signal1D instances containing neff1 and neff2. The signal and
            navigation dimensions are the same as the current signal if
            `cumulative` is True, otherwise the signal dimension is 0
            and the navigation dimension is the same as the current
            signal.

        Notes
        -----
        .. [Egerton2011] Ray Egerton, "Electron Energy-Loss
        Spectroscopy in the Electron Microscope", Springer-Verlag, 2011.

        """

        m0 = constants.value("electron mass")
        epsilon0 = constants.epsilon_0    # Vacuum permittivity [F/m]
        hbar = constants.hbar     # Reduced Plank constant [J·s]
        k = 2 * epsilon0 * m0 / (np.pi * nat * hbar ** 2)

        axis = self.axes_manager.signal_axes[0]
        if cumulative is False:
            dneff1 = k * simps((-1. / self.data).imag * axis.axis,
                               x=axis.axis,
                               axis=axis.index_in_array)
            dneff2 = k * simps(self.data.imag * axis.axis,
                               x=axis.axis,
                               axis=axis.index_in_array)
            neff1 = self._get_navigation_signal(data=dneff1)
            neff2 = self._get_navigation_signal(data=dneff2)
        else:
            neff1 = self._deepcopy_with_new_data(
                k * cumtrapz((-1. / self.data).imag * axis.axis,
                             x=axis.axis,
                             axis=axis.index_in_array,
                             initial=0))
            neff2 = self._deepcopy_with_new_data(
                k * cumtrapz(self.data.imag * axis.axis,
                             x=axis.axis,
                             axis=axis.index_in_array,
                             initial=0))

        # Prepare return
        neff1.metadata.General.title = (
            r"$n_{\mathrm{eff}}\left(-\Im\left(\epsilon^{-1}\right)\right)$ "
            "calculated from " +
            self.metadata.General.title +
            " using the Bethe f-sum rule.")
        neff2.metadata.General.title = (
            r"$n_{\mathrm{eff}}\left(\epsilon_{2}\right)$ "
            "calculated from " +
            self.metadata.General.title +
            " using the Bethe f-sum rule.")

        return neff1, neff2

    def get_electron_energy_loss_spectrum(self, zlp, t):
        data = ((-1 / self.data).imag * eels_constant(self, zlp, t).data *
                self.axes_manager.signal_axes[0].scale)
        s = self._deepcopy_with_new_data(data)
        s.set_signal_type("EELS")
        s.metadata.General.title = ("EELS calculated from " +
                                    self.metadata.General.title)
        return s


class DielectricFunction(DielectricFunction_mixin, ComplexSignal1D):
    pass


class LazyDielectricFunction(DielectricFunction_mixin, LazyComplexSignal1D):
    pass
