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
import pytest

import hyperspy.api as hs
from hyperspy.components1d import PESCoreLineShape


def test_PESCoreLineShape():
    core_line = PESCoreLineShape(A=10, FWHM=1.5, origin=0.5)
    x = np.linspace(-5, 15, 10)
    np.testing.assert_allclose(
        core_line.function(x),
        np.array([8.97054744e-04, 0.365234208, 7.09463858, 6.57499512,
                  0.290714653, 6.13260141e-04, 6.17204216e-08, 2.96359844e-13,
                  6.78916184e-20, 7.42026292e-28])
        )
    assert core_line._position is core_line.origin


def test_PESCoreLineShape_shirley():
    core_line = PESCoreLineShape(A=10, FWHM=1.5, origin=0.5)
    core_line.Shirley = True
    core_line.shirley.value = 0.01
    x = np.linspace(-5, 15, 10)
    np.testing.assert_allclose(
        core_line.function(x),
        np.array([0.144159014, 0.504843825, 7.16330182, 6.57790840,
                  0.290720786, 6.13260758e-04, 6.17204245e-08, 2.96359844e-13,
                  6.78916184e-20, 7.42026292e-28])
        )
    np.testing.assert_allclose(core_line.function(x), core_line.function_nd(x))


@pytest.mark.parametrize('Shirley', [False, True])
def test_PESCoreLineShape_fit(Shirley):
    # Component parameter values
    A = 10
    FWHM = 0.5
    origin = 5.0
    shirley = 0.01 if Shirley else 0.0

    offset, scale, size = 0, 0.1, 100
    x = np.linspace(offset, scale*size, size)
    comp = PESCoreLineShape(A=A, FWHM=FWHM, origin=origin)
    comp.Shirley = Shirley
    comp.shirley.value = shirley

    s = hs.signals.Signal1D(comp.function(x))
    axis = s.axes_manager[0]
    axis.offset, axis.scale = offset, scale
    s.add_gaussian_noise(0.1, random_state=1)
    m = s.create_model()
    core_line = PESCoreLineShape(A=1, FWHM=1.5, origin=0.5)
    core_line.Shirley = Shirley
    m.append(core_line)
    m.fit(grad='analytical')
    np.testing.assert_allclose(core_line.A.value, A, rtol=0.1)
    np.testing.assert_allclose(abs(core_line.FWHM.value), FWHM, rtol=0.1)
    np.testing.assert_allclose(core_line.origin.value, origin, rtol=0.1)
    np.testing.assert_allclose(core_line.shirley.value, shirley, rtol=0.1)


@pytest.mark.parametrize('Shirley', [False, True])
def test_PESCoreLineShape_function_nd(Shirley):
    A, FWHM, origin = 10, 1.5, 0.
    core_line = PESCoreLineShape(A=A, FWHM=FWHM, origin=origin)
    core_line.Shirley = Shirley
    core_line.shirley.value = 0.01 if Shirley else 0.0
    x = np.linspace(-5, 15, 1000)
    s = hs.signals.Signal1D(np.array([x]*2))

    # Manually set to test function_nd
    core_line._axes_manager = s.axes_manager
    core_line._create_arrays()
    core_line.A.map['values'] = [A] * 2
    core_line.FWHM.map['values'] = [FWHM] * 2
    core_line.origin.map['values'] = [origin] * 2
    core_line.shirley.map['values'] = [core_line.shirley.value] * 2

    values = core_line.function_nd(x)
    assert values.shape == (2, len(x))
    for v in values:
        np.testing.assert_allclose(v, core_line.function(x), rtol=0.5)


@pytest.mark.parametrize('Shirley', [False, True])
def test_recreate_component(Shirley):
    core_line = PESCoreLineShape(A=10, FWHM=1.5, origin=0.5)
    core_line.Shirley = Shirley

    s = hs.signals.Signal1D(np.zeros(10))
    m = s.create_model()
    m.append(core_line)
    model_dict = m.as_dictionary()

    m2 = s.create_model()
    m2._load_dictionary(model_dict)
    assert m2[0].Shirley == Shirley


