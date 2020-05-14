# Copyright 2007-2020 The HyperSpy developers
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
import pytest

from hyperspy.misc.eels.tools import get_edges_near_energy

def test_single_edge():
    edges = get_edges_near_energy(532, width=0)
    assert len(edges) == 1
    assert edges == ['O_K']

def test_multiple_edges():
    edges = get_edges_near_energy(640, width=100)
    assert len(edges) == 12
    assert edges == ['Mn_L3','I_M4','Cd_M2','Mn_L2','V_L1','I_M5','Cd_M3',
                     'In_M3','Xe_M5','Ag_M2','F_K','Xe_M4']
    
def test_negative_energy_width():
    with pytest.raises(Exception):
        get_edges_near_energy(849, width=-5)
        