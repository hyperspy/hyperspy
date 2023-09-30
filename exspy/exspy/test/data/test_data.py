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

import pytest

import exspy


def test_eds_sem():
    s = exspy.data.EDS_SEM_TM002()
    assert isinstance(s, exspy.signals.EDSSEMSpectrum)
    assert s.axes_manager.navigation_dimension == 0
    assert s.metadata.Sample.elements == ['Al', 'C', 'Cu', 'Mn', 'Zr']


def test_eds_tem():
    s = exspy.data.EDS_TEM_FePt_nanoparticles()
    assert isinstance(s, exspy.signals.EDSTEMSpectrum)
    assert s.axes_manager.navigation_dimension == 0
    assert s.metadata.Sample.elements == ['Fe', 'Pt']


@pytest.mark.parametrize('navigation_shape', [(), (2, ), (3, 4), (5, 6, 7)])
@pytest.mark.parametrize(
    ['add_noise', 'random_state'],
    [[True, 0], [True, None], [False, None]]
    )
def test_EELS_low_loss(add_noise, random_state, navigation_shape):
    s = exspy.data.EELS_low_loss(add_noise, random_state, navigation_shape)
    assert s.axes_manager.navigation_shape == navigation_shape


@pytest.mark.parametrize('add_powerlaw', [True, False])
@pytest.mark.parametrize('navigation_shape', [(1,), (2, )])
@pytest.mark.parametrize(
    ['add_noise', 'random_state'],
    [[True, 0], [True, None], [False, None]]
    )
def test_EELS_MnFe(add_powerlaw, add_noise, random_state, navigation_shape):
    s = exspy.data.EELS_MnFe(add_powerlaw, add_noise, random_state, navigation_shape)
    if navigation_shape == (1, ):
        navigation_shape = ()
    assert s.axes_manager.navigation_shape == navigation_shape
