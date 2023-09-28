# -*- coding: utf-8 -*-
# Copyright 2007-2023 The exspy developers
#
# This file is part of exspy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# exspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with exspy. If not, see <https://www.gnu.org/licenses/#GPL>.

import logging

import h5py
import numpy as np
import pooch
from scipy import constants

from hyperspy.defaults_parser import preferences
from exspy.misc.eels.base_gos import TabulatedGOS


_logger = logging.getLogger(__name__)

R = constants.value("Rydberg constant times hc in eV")
a0 = constants.value("Bohr radius")

_GOSH_DOI = "10.5281/zenodo.7645765"
_GOSH_URL = f"doi:{_GOSH_DOI}/Segger_Guzzinati_Kohl_1.5.0.gosh"
_GOSH_KNOWN_HASH = "md5:7fee8891c147a4f769668403b54c529b"


class GoshGOS(TabulatedGOS):
    """Read Generalized Oscillator Strength from a GOSH database.

    Parameters
    ----------
    element_subshell : {str, dict}
        Usually a string, for example, 'Ti_L3' for the GOS of the titanium L3
        subshell. If a dictionary is passed, it is assumed that a GOSH GOS was
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

    _name = 'gosh'
    _whitelist = {
        'gos_array': None,
        'rel_energy_axis': None,
        'qaxis': None,
        'element': None,
        'subshell': None,
        }

    def __init__(self, element_subshell, gos_file_path=None):
        """
        Parameters
        ----------
        element_subshell : str
            For example, 'Ti_L3' for the GOS of the titanium L3 subshell
        gos_file_path : str
            The path of the gosh file to use.
        """

        if gos_file_path is None:
            gos_file_path = pooch.retrieve(
                url=_GOSH_URL,
                known_hash=_GOSH_KNOWN_HASH,
                progressbar=preferences.General.show_progressbar,
            )
        self.gos_file_path = gos_file_path
        super().__init__(element_subshell=element_subshell)

    def read_gos_data(self):
        _logger.info(
            "GOSH precomputed GOS\n"
            f"\tElement: {self.element} "
            f"\tSubshell: {self.subshell}"
            f"\tOnset Energy = {self.onset_energy}"
        )
        element = self.element
        subshell = self.subshell

        error_message = (
            "The GOSH Parametrized GOS database does not "
            f"contain a valid entry the {subshell} subshell "
            f"of {element}. Please select a different database."
        )

        with h5py.File(self.gos_file_path, 'r') as h:
            conventions = h['metadata/edges_info']
            if subshell not in conventions:
                raise ValueError(error_message)
            table = conventions[subshell].attrs['table']
            self.subshell_factor = conventions[subshell].attrs['occupancy_ratio']
            stem = f'/{element}/{table}'
            if stem not in h:
                raise ValueError(error_message)
            gos_group = h[stem]
            gos = gos_group['data'][:]
            q = gos_group['q'][:]
            free_energies = gos_group['free_energies'][:]
            doi = h['/metadata/data_ref'].attrs['data_doi']

        gos = np.squeeze(gos.T)
        self.doi = doi
        self.gos_array = gos
        self.qaxis = q
        self.rel_energy_axis = free_energies - min(free_energies)
        self.energy_axis = self.rel_energy_axis + self.onset_energy
