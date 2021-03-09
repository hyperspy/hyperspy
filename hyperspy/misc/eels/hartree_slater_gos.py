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

from pathlib import Path

from hyperspy.defaults_parser import preferences
from hyperspy.misc.eels.base_gos import GOSBase
from hyperspy.misc.elements import elements
from hyperspy.misc.export_dictionary import (
    export_to_dictionary, load_from_dictionary)

_logger = logging.getLogger(__name__)


R = sp.constants.value("Rydberg constant times hc in eV")
a0 = sp.constants.value("Bohr radius")


class HartreeSlaterGOS(GOSBase):

    """Read Hartree-Slater Generalized Oscillator Strenght parametrized
    from files.

    Parameters
    ----------
    element_subshell : {str, dict}
        Usually a string, for example, 'Ti_L3' for the GOS of the titanium L3
        subshell. If a dictionary is passed, it is assumed that Hartree Slater
        GOS was exported using `GOS.as_dictionary`, and will be reconstructed.

    Methods
    -------
    readgosfile()
        Read the GOS files of the element subshell from the location
        defined in Preferences.
    get_qaxis_and_gos(ienergy, qmin, qmax)
        given the energy axis index and qmin and qmax values returns
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

    """

    _name = 'Hartree-Slater'

    def __init__(self, element_subshell):
        """
        Parameters
        ----------

        element_subshell : str
            For example, 'Ti_L3' for the GOS of the titanium L3 subshell

        """
        self._whitelist = {'gos_array': None,
                           'rel_energy_axis': None,
                           'qaxis': None,
                           'element': None,
                           'subshell': None
                           }
        if isinstance(element_subshell, dict):
            self.element = element_subshell['element']
            self.subshell = element_subshell['subshell']
            self.read_elements()
            self._load_dictionary(element_subshell)
        else:
            self.element, self.subshell = element_subshell.split('_')
            self.read_elements()
            self.readgosfile()

    def _load_dictionary(self, dictionary):
        load_from_dictionary(self, dictionary)
        self.energy_axis = self.rel_energy_axis + self.onset_energy

    def as_dictionary(self, fullcopy=True):
        """Export the GOS as a dictionary.
        """
        dic = {}
        export_to_dictionary(self, self._whitelist, dic, fullcopy)
        return dic

    def readgosfile(self):
        _logger.info(
            "Hartree-Slater GOS\n"
            f"\tElement: {self.element} "
            f"\tSubshell: {self.subshell}"
            f"\tOnset Energy = {self.onset_energy}"
        )
        element = self.element
        subshell = self.subshell

        # Check if the Peter Rez's Hartree Slater GOS distributed by
        # Gatan are available. Otherwise exit
        gos_root = Path(preferences.EELS.eels_gos_files_path)
        gos_file = gos_root.joinpath(
            (
                elements[element]["Atomic_properties"]["Binding_energies"][subshell][
                    "filename"
                ]
            )
        )

        if not gos_root.is_dir():
            raise FileNotFoundError(
                "Parametrized Hartree-Slater GOS files not "
                f"found in {gos_root}. Please define a valid "
                "location for the files in the preferences as "
                "`preferences.EELS.eels_gos_files_path`."
            )

        with open(gos_file) as f:
            GOS_list = f.read().replace('\r', '').split()

        # Map the parameters
        info1_1 = float(GOS_list[2])
        info1_2 = float(GOS_list[3])
        ncol = int(GOS_list[5])
        info2_1 = float(GOS_list[6])
        info2_2 = float(GOS_list[7])
        nrow = int(GOS_list[8])
        self.gos_array = np.array(GOS_list[9:], dtype=np.float64)
        # The division by R is not in the equations, but it seems that
        # the the GOS was tabulated this way
        self.gos_array = self.gos_array.reshape(nrow, ncol) / R
        del GOS_list

        # Calculate the scale of the matrix
        self.rel_energy_axis = self.get_parametrized_energy_axis(
            info2_1, info2_2, nrow)
        self.qaxis = self.get_parametrized_qaxis(
            info1_1, info1_2, ncol)
        self.energy_axis = self.rel_energy_axis + self.onset_energy

    def integrateq(self, onset_energy, angle, E0):
        energy_shift = onset_energy - self.onset_energy
        self.energy_shift = energy_shift
        qint = np.zeros((self.energy_axis.shape[0]))
        # Calculate the cross section at each energy position of the
        # tabulated GOS
        gamma = 1 + E0 / 511.06
        T = 511060 * (1 - 1 / gamma ** 2) / 2
        for i in range(0, self.gos_array.shape[0]):
            E = self.energy_axis[i] + energy_shift
            # Calculate the limits of the q integral
            qa0sqmin = (E ** 2) / (4 * R * T) + (E ** 3) / (
                8 * gamma ** 3 * R * T ** 2)
            p02 = T / (R * (1 - 2 * T / 511060))
            pp2 = p02 - E / R * (gamma - E / 1022120)
            qa0sqmax = qa0sqmin + 4 * np.sqrt(p02 * pp2) * \
                (math.sin(angle / 2)) ** 2
            qmin = math.sqrt(qa0sqmin) / a0
            qmax = math.sqrt(qa0sqmax) / a0
            # Perform the integration in a log grid
            qaxis, gos = self.get_qaxis_and_gos(i, qmin, qmax)
            logsqa0qaxis = np.log((a0 * qaxis) ** 2)
            qint[i] = sp.integrate.simps(gos, logsqa0qaxis)
        E = self.energy_axis + energy_shift
        # Energy differential cross section in (barn/eV/atom)
        qint *= (4.0 * np.pi * a0 ** 2.0 * R ** 2 / E / T *
                 self.subshell_factor) * 1e28
        self.qint = qint
        return sp.interpolate.interp1d(E, qint, kind=3)
