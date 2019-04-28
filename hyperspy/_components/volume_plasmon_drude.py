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

from hyperspy.component import Component


class VolumePlasmonDrude(Component):

    r"""Drude volume plasmon energy loss function component, the energy loss
    function is defined as:

    .. math::

       f(E) = \frac{E(\Delta E_p)E_p^2}{(E^2-E_p^2)^2+(E\Delta E_p)^2}

    +------------------+-----------------+
    | Parameter        |    Attribute    |
    +------------------+-----------------+
    |:math:`E_p`       |  plasmon_energy |
    +------------------+-----------------+
    |:math:`\Delta E_p`|      fwhm       |
    +------------------+-----------------+
    | intensity        |   intensity     |
    +------------------+-----------------+

    Notes
    -----
    Refer to Egerton, R. F., Electron Energy-Loss Spectroscopy in the
    Electron Microscope, 2nd edition, Plenum Press 1996, pp. 154-158
    for details, including original equations.


    """

    def __init__(self, intensity=1., plasmon_energy=15., fwhm=1.5):
        Component.__init__(self, ['intensity', 'plasmon_energy',
                                  'fwhm'])
        self._position = self.plasmon_energy
        self.intensity.value = intensity
        self.plasmon_energy.value = plasmon_energy
        self.fwhm.value = fwhm
        self.plasmon_energy.grad = self.grad_plasmon_energy
        self.fwhm.grad = self.grad_fwhm
        self.intensity.grad = self.grad_intensity

    def function(self, x):
        plasmon_energy = self.plasmon_energy.value
        fwhm = self.fwhm.value
        intensity = self.intensity.value
        pe2 = plasmon_energy ** 2
        return np.where(
            x > 0,
            intensity * (pe2 * x * fwhm) /
            ((x ** 2 - pe2) ** 2 + (x * fwhm) ** 2),
            0)

    # Partial derivative with respect to the plasmon energy E_p
    def grad_plasmon_energy(self, x):
        plasmon_energy = self.plasmon_energy.value
        fwhm = self.fwhm.value
        intensity = self.intensity.value

        return np.where(
            x > 0,
            2 * x * fwhm * plasmon_energy * intensity * (
                (x ** 4 + (x * fwhm) ** 2 - plasmon_energy ** 4) /
                (x ** 4 + x ** 2 * (fwhm ** 2 - 2 * plasmon_energy ** 2) +
                    plasmon_energy ** 4) ** 2),
            0)

    # Partial derivative with respect to the plasmon linewidth delta_E_p
    def grad_fwhm(self, x):
        plasmon_energy = self.plasmon_energy.value
        fwhm = self.fwhm.value
        intensity = self.intensity.value

        return np.where(
            x > 0,
            x * plasmon_energy * intensity * (
                (x ** 4 - x ** 2 * (2 * plasmon_energy ** 2 + fwhm ** 2) +
                    plasmon_energy ** 4) /
                (x ** 4 + x ** 2 * (fwhm ** 2 - 2 * plasmon_energy ** 2) +
                    plasmon_energy ** 4) ** 2),
            0)

    def grad_intensity(self, x):
        return self.function(x) / self.intensity.value
