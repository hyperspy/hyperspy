# -*- coding: utf-8 -*-
# Copyright 2007-2023 The exSpy developers
#
# This file is part of exSpy.
#
# exSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# exSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with exSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import logging
from pathlib import Path

import numpy as np
from scipy import constants

from hyperspy.defaults_parser import preferences
from exspy.misc.eels.base_gos import TabulatedGOS


_logger = logging.getLogger(__name__)

R = constants.value("Rydberg constant times hc in eV")
a0 = constants.value("Bohr radius")

# This dictionary accounts for conventions chosen in naming the data files, as well as normalisation.
# These cross sections contain only odd-number edges such as N3, or M5, and are normalised accordingly.
# Other edges can be obtained as scaled copies of the provided ones.
conventions = { 'K' : {'table': 'K1', 'factor': 1},
                'L1': {'table': 'L1', 'factor': 1},
                    'L2,3': {'table':'L3', 'factor': 3/2}, 'L2': {'table':'L3', 'factor': 1/2}, 'L3': {'table':'L3', 'factor': 1},
                'M1': {'table': 'M1', 'factor': 1},
                    'M2,3': {'table':'M3', 'factor': 3/2}, 'M2': {'table':'M3', 'factor': 1/2}, 'M3': {'table':'M3', 'factor': 1},
                            'M4,5': {'table':'M5', 'factor': 5/3}, 'M4': {'table':'M5', 'factor': 2/3}, 'M5': {'table':'M5', 'factor': 1},
                'N1': {'table': 'N1', 'factor': 1},
                    'N2,3': {'table':'N3', 'factor': 3/2}, 'N2': {'table':'N3', 'factor': 1/2}, 'N3': {'table':'N3', 'factor': 1},
                            'N4,5': {'table':'N5', 'factor': 5/3}, 'N4': {'table':'N5', 'factor': 2/3}, 'N5': {'table':'N5', 'factor': 1},
                                'N6,7': {'table': 'N7', 'factor': 7/4}, 'N6': {'table':'N7', 'factor': 4/7}, 'N7': {'table':'N7', 'factor': 1},
                'O1': {'table': 'O1', 'factor': 1},
                    'O2,3': {'table':'O3', 'factor': 3/2}, 'O2': {'table':'O3', 'factor': 1/2}, 'O3': {'table':'O3', 'factor': 1},
                        'O4,5': {'table':'O5', 'factor': 5/3}, 'O4': {'table':'O5', 'factor': 2/3}, 'O5': {'table':'O5', 'factor': 1}}


class HartreeSlaterGOS(TabulatedGOS):

    """Read Hartree-Slater Generalized Oscillator Strength parametrized
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
    _whitelist = {
        'gos_array': None,
        'rel_energy_axis': None,
        'qaxis': None,
        'element': None,
        'subshell': None,
        }

    def read_elements(self):
        super().read_elements()
        self.subshell_factor = conventions[self.subshell]['factor']

    def read_gos_data(self):  # pragma: no cover
        _logger.info(
            "Hartree-Slater GOS\n"
            f"\tElement: {self.element} "
            f"\tSubshell: {self.subshell}"
            f"\tOnset Energy = {self.onset_energy}"
        )
        element = self.element
        subshell = self.subshell
        table = conventions[subshell]['table']

        # Check if the Peter Rez's Hartree Slater GOS distributed by
        # Gatan are available. Otherwise exit
        gos_root = Path(preferences.EELS.eels_gos_files_path)
        gos_file = gos_root / f"{element}.{table}"

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
        self.gos_array = np.array(GOS_list[9:], dtype=float)
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
