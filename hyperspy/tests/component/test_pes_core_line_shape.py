# -*- coding: utf-8 -*-
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


@pytest.mark.parametrize('Shirley', [False, True])
def test_see_fit(Shirley):
    # Component parameter values
    A = 10
    FWHM = 0.5
    origin = 5.0
    shirley = 0.01

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
    see = PESCoreLineShape(A=1, FWHM=1.5, origin=0.5)
    see.Shirley = Shirley
    see.shirley.free = True
    m.append(see)
    m.fit()
    np.testing.assert_allclose(see.A.value, A, rtol=0.1)
    np.testing.assert_allclose(see.FWHM.value, FWHM, rtol=0.1)
    np.testing.assert_allclose(see.origin.value, origin, rtol=0.1)
    np.testing.assert_allclose(see.shirley.value, shirley, rtol=0.1)

