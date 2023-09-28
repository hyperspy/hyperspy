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

from pathlib import Path

import h5py
import pooch
import pytest

from hyperspy.defaults_parser import preferences
from exspy.misc.eels.gosh_gos import GoshGOS
from exspy.misc.eels.hartree_slater_gos import HartreeSlaterGOS
from exspy.misc.eels import HydrogenicGOS
from hyperspy.misc.elements import elements


GOSH10 = pooch.retrieve(
    url="doi:10.5281/zenodo.6599071/Segger_Guzzinati_Kohl_1.0.0.gos",
    known_hash="md5:d65d5c23142532fde0a80e160ab51574",
    progressbar=False,
)


@pytest.mark.skipif(not Path(preferences.EELS.eels_gos_files_path).exists(),
                    reason="Hartree-Slater GOS not available")
def test_hartree_slater_gos():
    gos = HartreeSlaterGOS('Ti_L3')
    gos.read_elements()


def test_hydrogenic_gos_error_M_shells():
    with pytest.raises(ValueError):
        _ = HydrogenicGOS('Ti_M2')


def test_element_not_in_database():
    with pytest.raises(ValueError):
        _ = HydrogenicGOS('Lr_M2')


def test_subshell_not_in_database():
    with pytest.raises(ValueError):
        _ = HydrogenicGOS('Ti_L4')


def test_gosh_not_in_conventions():
    gos = GoshGOS('Ti_L2')
    gos.subshell = 'L234'
    with pytest.raises(ValueError):
        gos.read_gos_data()


def test_gosh_not_in_file():
    # Use version 1.0 which doesn't have the Ac element
    with pytest.raises(ValueError):
        _ = GoshGOS('Ac_L3', gos_file_path=GOSH10)


def test_binding_energy_database():
    gos = GoshGOS('Ti_L3')
    gosh15 = h5py.File(gos.gos_file_path)
    for element in gosh15.keys():
        # These elements are not in the database
        if element not in ['Bk', 'Cf', 'Cm', 'metadata']:
            assert 'Binding_energies' in elements[element]['Atomic_properties'].keys()
