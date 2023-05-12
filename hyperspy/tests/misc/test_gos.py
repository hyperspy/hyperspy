# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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

from pathlib import Path

import h5py
import pooch
import pytest

from hyperspy.defaults_parser import preferences
from hyperspy.misc.eels.gosh_gos import GoshGOS
from hyperspy.misc.eels.hartree_slater_gos import HartreeSlaterGOS
from hyperspy.misc.eels.hydrogenic_gos import HydrogenicGOS


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
