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

import math
import logging

import numpy as np
import scipy as sp
import scipy.interpolate

from hyperspy.misc.eels.base_gos import GOSBase

_logger = logging.getLogger(__name__)

XU = [
    .82, .52, .52, .42, .30, .29, .22, .30, .22, .16, .12, .13, .13, .14, .16,
    .18, .19, .22, .14, .11, .12, .12, .12, .10, .10, .10]
# IE3=[73,99,135,164,200,245,294,347,402,455,513,575,641,710,
# 779,855,931,1021,1115,1217,1323,1436,1550,1675]

# IE1=[118,149,189,229,270,320,377,438,500,564,628,695,769,846,
# 926,1008,1096,1194,1142,1248,1359,1476,1596,1727]

R = sp.constants.value("Rydberg constant times hc in eV")


class HydrogenicGOS(GOSBase):

    """Computes the K and L GOS using R. Egerton's  routines.

    Parameters
    ----------
    element_subshell : str
        For example, 'Ti_L3' for the GOS of the titanium L3 subshell

    Methods
    -------
    parametrize_GOS()
        Parametrize the GOS to speed up the calculation.
    get_qaxis_and_gos(ienergy, qmin, qmax)
        Given the energy axis index and qmin and qmax values returns
        the qaxis and gos between qmin and qmax using linear
        interpolation to include qmin and qmax in the range.


    Attributes
    ----------
    energy_axis : array
        The tabulated energy axis
    qaxis : array
        The tabulated qaxis
    energy_onset: float
        The energy onset for the given element subshell as obtained
        from iternal tables.

    Notes
    -----
    The Hydrogeninc GOS are calculated using R. Egerton's SIGMAK3 and
    SIGMAL3 routines that has been translated from Matlab to Python by
    I. Iyengar. See http://www.tem-eels.ca/ for the original code.

    """
    _name = 'hydrogenic'

    def __init__(self, element_subshell):
        """
        Parameters
        ----------

        element_subshell : str
            For example, 'Ti_L3' for the GOS of the titanium L3 subshell

        """
        # Check if the Peter Rez's Hartree Slater GOS distributed by
        # Gatan are available. Otherwise exit

        self.element, self.subshell = element_subshell.split('_')
        self.read_elements()
        self.energy_shift = 0

        if self.subshell[:1] == 'K':
            self.gosfunc = self.gosfuncK
            self.rel_energy_axis = self.get_parametrized_energy_axis(
                50, 3, 50)
        elif self.subshell[:1] == 'L':
            self.gosfunc = self.gosfuncL
            self.onset_energy_L3 = self.element_dict['Atomic_properties'][
                'Binding_energies']['L3']['onset_energy (eV)']
            self.onset_energy_L1 = self.element_dict['Atomic_properties'][
                'Binding_energies']['L1']['onset_energy (eV)']
            self.onset_energy = self.onset_energy_L3
            relative_axis = self.get_parametrized_energy_axis(
                50, 3, 50)
            dL3L2 = self.onset_energy_L1 - self.onset_energy_L3
            self.rel_energy_axis = np.hstack((
                relative_axis[:relative_axis.searchsorted(dL3L2)],
                relative_axis + dL3L2))
        else:
            raise ValueError(
                'The Hydrogenic GOS currently can only'
                'compute K or L shells. Try using Hartree-Slater GOS')

        self.energy_axis = self.rel_energy_axis + self.onset_energy

        info_str = (
            "\nHydrogenic GOS\n" +
            ("\tElement: %s " % self.element) +
            ("\tSubshell: %s " % self.subshell) +
            ("\tOnset Energy = %s " % self.onset_energy))
        _logger.info(info_str)

    def integrateq(self, onset_energy, angle, E0):
        energy_shift = onset_energy - self.onset_energy
        self.energy_shift = energy_shift
        gamma = 1 + E0 / 511.06
        T = 511060 * (1 - 1 / gamma ** 2) / 2
        qint = np.zeros((self.energy_axis.shape[0]))
        for i, E in enumerate(self.energy_axis + energy_shift):
            qa0sqmin = (E ** 2) / (4 * R * T) + (E ** 3) / (
                8 * gamma ** 3 * R * T ** 2)
            p02 = T / (R * (1 - 2 * T / 511060))
            pp2 = p02 - E / R * (gamma - E / 1022120)
            qa0sqmax = qa0sqmin + 4 * np.sqrt(p02 * pp2) * \
                (math.sin(angle / 2)) ** 2

            # dsbyde IS THE ENERGY-DIFFERENTIAL X-SECN (barn/eV/atom)
            qint[i] = 3.5166e8 * (R / T) * (R / E) * (
                scipy.integrate.quad(
                    lambda x: self.gosfunc(E, np.exp(x)),
                    math.log(qa0sqmin), math.log(qa0sqmax))[0])
        self.qint = qint
        return sp.interpolate.interp1d(self.energy_axis + energy_shift,
                                       qint)

    def gosfuncK(self, E, qa02):
        # gosfunc calculates (=DF/DE) which IS PER EV AND PER ATOM
        z = self.Z
        r = 13.606
        zs = 1.0
        rnk = 1
        if z != 1:
            zs = z - 0.5
            rnk = 2

        q = qa02 / zs ** 2
        kh2 = E / (r * zs ** 2) - 1
        akh = np.sqrt(np.abs(kh2))
        if akh < 0.01:
            akh = 0.01
        if kh2 >= 0.0:
            d = 1 - np.e ** (-2 * np.pi / kh2)
            bp = np.arctan(2 * akh / (q - kh2 + 1))
            if bp < 0:
                bp = bp + np.pi
            c = np.e ** ((-2 / akh) * bp)
        else:
            d = 1
            y = -1 / akh * np.log((q + 1 - kh2 + 2 * akh) / (
                q + 1 - kh2 - 2 * akh))
            c = np.e ** y
        a = ((q - kh2 + 1) ** 2 + 4 * kh2) ** 3
        return 128 * rnk * E / (
            r * zs ** 4) * c / d * (q + kh2 / 3 + 1 / 3) / (a * r)

    def gosfuncL(self, E, qa02):
        # gosfunc calculates (=DF/DE) which IS PER EV AND PER ATOM
        # Note: quad function only works with qa02 due to IF statements in
        # function

        z = self.Z
        r = 13.606
        zs = z - 0.35 * (8 - 1) - 1.7
        iz = z - 11
        if iz >= len(XU):
            # Egerton does not tabulate the correction for Z>36.
            # This produces XSs that are within 10% of Hartree-Slater XSs
            # for these elements.
            u = .1
        else:
            # Egerton's correction to the Hydrogenic XS
            u = XU[int(iz)]
        el3 = self.onset_energy_L3 + self.energy_shift
        el1 = self.onset_energy_L1 + self.energy_shift

        q = qa02 / zs ** 2
        kh2 = E / (r * zs ** 2) - 0.25
        akh = np.sqrt(np.abs(kh2))
        if kh2 >= 0.0:
            d = 1 - np.exp(-2 * np.pi / akh)
            bp = np.arctan(akh / (q - kh2 + 0.25))
            if bp < 0:
                bp = bp + np.pi
            c = np.exp((-2 / akh) * bp)
        else:
            d = 1
            y = -1 / akh * \
                np.log((q + 0.25 - kh2 + akh) / (q + 0.25 - kh2 - akh))
            c = np.exp(y)

        if E - el1 <= 0:
            g = 2.25 * q ** 4 - (0.75 + 3 * kh2) * q ** 3 + (
                0.59375 - 0.75 * kh2 - 0.5 * kh2 ** 2) * q * q + (
                0.11146 + 0.85417 * kh2 + 1.8833 * kh2 * kh2 + kh2 ** 3) * \
                q + 0.0035807 + kh2 / 21.333 + kh2 * kh2 / 4.5714 + kh2 ** 3 \
                / 2.4 + kh2 ** 4 / 4

            a = ((q - kh2 + 0.25) ** 2 + kh2) ** 5
        else:
            g = q ** 3 - (5 / 3 * kh2 + 11 / 12) * q ** 2 + (
                kh2 * kh2 / 3 + 1.5 * kh2 + 65 / 48) * q + kh2 ** 3 / 3 + \
                0.75 * kh2 * kh2 + 23 / 48 * kh2 + 5 / 64
            a = ((q - kh2 + 0.25) ** 2 + kh2) ** 4
        rf = ((E + 0.1 - el3) / 1.8 / z / z) ** u
        # The following commented lines are to give a more accurate GOS
        # for edges presenting white lines. However, this is not relevant
        # for quantification by curve fitting.
        # if np.abs(iz - 11) <= 5 and E - el3 <= 20:
        #     rf = 1
        return rf * 32 * g * c / a / d * E / r / r / zs ** 4
