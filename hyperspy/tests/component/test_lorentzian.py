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

from hyperspy.components1d import Lorentzian


def test_function():
    g = Lorentzian()
    g.A.value = 1.5 * np.pi
    g.gamma.value = 1
    g.centre.value = 2
    assert_allclose(g.function(2), 1.5)
    assert_allclose(g.function(4), 0.3)


def test_util_gamma_getset():
    g1 = Lorentzian()
    g1.gamma.value = 3.0
    assert_allclose(g1.gamma.value, 3.0)


def test_util_fwhm_set():
    g1 = Lorentzian()
    g1.fwhm = 1.0
    assert_allclose(g1.gamma.value, 0.5)


def test_util_fwhm_get():
    g1 = Lorentzian()
    g1.gamma.value = 2.0
    assert_allclose(g1.fwhm, 4.0)


def test_util_fwhm_getset():
    g1 = Lorentzian()
    g1.fwhm = 4.0
    assert_allclose(g1.fwhm, 4.0)
