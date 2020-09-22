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

import pytest

from hyperspy.misc.eels.tools import get_edges_near_energy, get_info_from_edges


def test_single_edge():
    edges = get_edges_near_energy(532, width=0)
    assert len(edges) == 1
    assert edges == ['O_K']

def test_multiple_edges():
    edges = get_edges_near_energy(640, width=100)
    assert len(edges) == 12
    assert edges == ['Mn_L3','I_M4','Cd_M2','Mn_L2','V_L1','I_M5','Cd_M3',
                     'In_M3','Xe_M5','Ag_M2','F_K','Xe_M4']

def test_multiple_edges_ascending():
    edges = get_edges_near_energy(640, width=100, order='ascending')
    assert len(edges) == 12
    assert edges == ['Ag_M2','Cd_M3','I_M5','V_L1','I_M4','Mn_L3','Cd_M2',
                     'Mn_L2','In_M3','Xe_M5','F_K','Xe_M4']

def test_multiple_edges_descending():
    edges = get_edges_near_energy(640, width=100, order='descending')
    assert len(edges) == 12
    assert edges == ['F_K','Xe_M4','Xe_M5','In_M3','Cd_M2','Mn_L2','Mn_L3',
                     'I_M4','V_L1','I_M5','Cd_M3','Ag_M2']

def test_negative_energy_width():
    with pytest.raises(Exception):
        get_edges_near_energy(849, width=-5)

def test_wrong_ordering():
    with pytest.raises(ValueError):
        get_edges_near_energy(532, order='random')

def test_info_one_edge():
    info = get_info_from_edges('O_K')
    assert len(info) == 1

def test_info_multiple_edges():
    info = get_info_from_edges(['O_K', 'N_K', 'Cr_L3'])
    assert len(info) == 3

def test_info_wrong_edge_format():
    with pytest.raises(ValueError):
        get_info_from_edges(['O_K', 'NK'])
