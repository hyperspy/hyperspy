# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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
from numpy.testing import assert_allclose

from hyperspy.components2d import Gaussian2D
sigma2fwhm = 2 * np.sqrt(2 * np.log(2))


def test_function():
    g = Gaussian2D()
    g.A.value = 14
    g.sigma_x.value = 1.
    g.sigma_y.value = 2.
    g.centre_x.value = -5.
    g.centre_y.value = -5.
    assert_allclose(g.function(-5, -5), 1.1140846)
    assert_allclose(g.function(-2, -3), 0.007506643)
    assert g._is2D
    assert g._position_x == g.centre_x
    assert g._position_y == g.centre_y


def test_util_fwhm_set():
    g1 = Gaussian2D()
    g1.fwhm_x = 0.33
    g1.fwhm_y = 0.33
    g1.A.value = 1.0
    assert_allclose(g1.fwhm_x, g1.sigma_x.value * sigma2fwhm)
    assert_allclose(g1.fwhm_y, g1.sigma_y.value * sigma2fwhm)


def test_util_fwhm_get():
    g1 = Gaussian2D(sigma_x=0.33, sigma_y=0.33)
    g1.A.value = 1.0
    assert_allclose(g1.fwhm_x, g1.sigma_x.value * sigma2fwhm)
    assert_allclose(g1.fwhm_y, g1.sigma_y.value * sigma2fwhm)


def test_util_fwhm_getset():
    g1 = Gaussian2D()
    g1.fwhm_x = 0.33
    g1.fwhm_y = 0.33
    assert g1.fwhm_x == 0.33
    assert g1.fwhm_y == 0.33
