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

import numpy as np

from hyperspy.misc.eds.utils import get_xray_lines_near_energy, take_off_angle


def test_xray_lines_near_energy():
    E = 1.36
    lines = get_xray_lines_near_energy(E)
    assert (
        lines ==
        ['Pm_M2N4', 'Ho_Ma', 'Eu_Mg', 'Se_La', 'Br_Ln', 'W_Mz', 'As_Lb3',
         'Kr_Ll', 'Ho_Mb', 'Ta_Mz', 'Dy_Mb', 'As_Lb1', 'Gd_Mg', 'Er_Ma',
         'Sm_M2N4', 'Mg_Kb', 'Se_Lb1', 'Ge_Lb3', 'Br_Ll', 'Sm_Mg', 'Dy_Ma',
         'Nd_M2N4', 'As_La', 'Re_Mz', 'Hf_Mz', 'Kr_Ln', 'Er_Mb', 'Tb_Mb'])
    lines = get_xray_lines_near_energy(E, 0.02)
    assert lines == ['Pm_M2N4']
    E = 5.4
    lines = get_xray_lines_near_energy(E)
    assert (
        lines ==
        ['Cr_Ka', 'La_Lb2', 'V_Kb', 'Pm_La', 'Pm_Ln', 'Ce_Lb3', 'Gd_Ll',
         'Pr_Lb1', 'Xe_Lg3', 'Pr_Lb4'])
    lines = get_xray_lines_near_energy(E, only_lines=('a', 'b'))
    assert (
        lines ==
        ['Cr_Ka', 'V_Kb', 'Pm_La', 'Pr_Lb1'])
    lines = get_xray_lines_near_energy(E, only_lines=('a'))
    assert (
        lines ==
        ['Cr_Ka', 'Pm_La'])

def test_takeoff_angle():
    np.testing.assert_allclose(40.,take_off_angle(30.,0.,10.))
    np.testing.assert_allclose(40.,take_off_angle(0.,90.,10.,beta_tilt=30.))
    np.testing.assert_allclose(73.15788376370121,take_off_angle(45.,45.,45.,
                               45.))
