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

import logging
import math
from pathlib import Path

import numpy as np
from scipy import constants, integrate, interpolate
import h5py

from hyperspy.defaults_parser import preferences
from hyperspy.misc.eels.base_gos import GOSBase
from hyperspy.misc.export_dictionary import (
    export_to_dictionary, load_from_dictionary)

_logger = logging.getLogger(__name__)

R = constants.value("Rydberg constant times hc in eV")
a0 = constants.value("Bohr radius")


class Hdf5GOS(GOSBase):
    """Read Generalized Oscillator Strength from a GOS5 database.

    Parameters
    ----------
    element_subshell : {str, dict}
        Usually a string, for example, 'Ti_L3' for the GOS of the titanium L3
        subshell. If a dictionary is passed, it is assumed that a GOS5 GOS was
        exported using `GOS.as_dictionary`, and will be reconstructed.

    Methods
    -------
    readgosarray()
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
    
    _name = 'GOS5'
    
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
                           'subshell': None,
                           'doi': None
                           }
        if isinstance(element_subshell, dict):
            self.element = element_subshell['element']
            self.subshell = element_subshell['subshell']
            self.read_elements()
            self._load_dictionary(element_subshell)
        else:
            self.element, self.subshell = element_subshell.split('_')
            self.read_elements()
            self.readgosarray()
    
    def _load_dictionary(self, dictionary):
        load_from_dictionary(self, dictionary)
        self.energy_axis = self.rel_energy_axis + self.onset_energy
    
    def as_dictionary(self, fullcopy=True):
        """Export the GOS as a dictionary.
        """
        dic = {}
        export_to_dictionary(self, self._whitelist, dic, fullcopy)
        return dic
    
    def readgosarray(self):
        _logger.info(
            "GOS5 precomputed GOS\n"
            f"\tElement: {self.element} "
            f"\tSubshell: {self.subshell}"
            f"\tOnset Energy = {self.onset_energy}"
        )
        element = self.element
        subshell = self.subshell
        
        # Check if the specified data file exists, otherwise
        # exit.
        gos_file = Path(preferences.EELS.eels_gos5_file_path)
        
        if not gos_file.is_file():
            raise FileNotFoundError(
                "The GOS5 Parametrized GOS database file not "
                f"found in {gos_file}. Please define a valid "
                "location for the files in the preferences as "
                "`preferences.EELS.eels_gos5_file_path`."
            )
        
        def edge_not_in_database():
            raise KeyError(
                "The GOS5 Parametrized GOS database does not "
                f"contain a valid entry the {subshell} subshell"
                f"of {element}. Please select a different database"
            )
        
        with h5py.File(gos_file, 'r') as h:
            conventions = h['metadata/edges_info']
            if subshell not in conventions:
                edge_not_in_database()
            table = conventions[subshell].attrs['table']
            self.subshell_factor = conventions[subshell].attrs['occupancy_ratio']
            stem = '/{}/{}'.format(element, table)
            if stem not in h:
                edge_not_in_database()
            gos_group = h[stem]
            gos = gos_group['data'][:]
            q = gos_group['q'][:]
            free_energies = gos_group['free_energies'][:]
            doi = h['/metadata/data_ref'].attrs['data_doi']
        
        gos = np.squeeze(gos.T)
        self.doi = doi
        self.gos_array = gos * self.subshell_factor
        self.qaxis = q
        self.rel_energy_axis = free_energies - min(free_energies)
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
            qint[i] = integrate.simps(gos, logsqa0qaxis)
        E = self.energy_axis + energy_shift
        # Energy differential cross section in (barn/eV/atom)
        qint *= (4.0 * np.pi * a0 ** 2.0 * R ** 2 / E / T *
                 self.subshell_factor) * 1e28
        self.qint = qint
        return interpolate.interp1d(E, qint, kind=3)
