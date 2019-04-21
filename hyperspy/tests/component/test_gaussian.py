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

import itertools
import numpy as np
from numpy.testing import assert_allclose
import pytest

from hyperspy.components1d import Gaussian
from hyperspy.signals import Signal1D
from hyperspy.utils import stack

sqrt2pi = np.sqrt(2 * np.pi)
sigma2fwhm = 2 * np.sqrt(2 * np.log(2))

TRUE_FALSE_2_TUPLE = [p for p in itertools.product((True, False), repeat=2)]


def test_function():
    g = Gaussian()
    g.centre.value = 1
    g.sigma.value = 2 / sigma2fwhm
    g.A.value = 3 * sqrt2pi * g.sigma.value
    assert_allclose(g.function(2), 1.5)
    assert_allclose(g.function(1), 3)

@pytest.mark.parametrize(("only_current", "binned"), TRUE_FALSE_2_TUPLE)
def test_estimate_parameters_binned(only_current, binned):
    s = Signal1D(np.empty((100,)))
    s.metadata.Signal.binned = binned
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 1
    axis.offset = -20
    g1 = Gaussian(50015.156, 10/sigma2fwhm, 10)
    s.data = g1.function(axis.axis)
    g2 = Gaussian()
    factor = axis.scale if binned else 1
    assert g2.estimate_parameters(s, axis.low_value, axis.high_value,
                                  only_current=only_current)
    assert g2.binned == binned
    assert_allclose(g1.A.value, g2.A.value * factor)
    assert abs(g2.centre.value - g1.centre.value) <= 1e-3
    assert abs(g2.sigma.value - g1.sigma.value) <= 0.1

@pytest.mark.parametrize(("binned"), (True, False))
def test_function_nd(binned):
    s = Signal1D(np.empty((100,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 1
    axis.offset = -20
    g1 = Gaussian(50015.156, 10/sigma2fwhm, 10)
    s.data = g1.function(axis.axis)
    s.metadata.Signal.binned = binned
    s2 = stack([s] * 2)
    g2 = Gaussian()
    factor = axis.scale if binned else 1
    g2.estimate_parameters(s2, axis.low_value, axis.high_value, False)
    assert g2.binned == binned
    assert_allclose(g2.function_nd(axis.axis) * factor, s2.data)

def test_util_fwhm_set():
    g1 = Gaussian()
    g1.fwhm = 1.0
    assert_allclose(g1.sigma.value, 1.0 / sigma2fwhm)

def test_util_fwhm_get():
    g1 = Gaussian()
    g1.sigma.value = 1.0
    assert_allclose(g1.fwhm, 1.0 * sigma2fwhm)

def test_util_fwhm_getset():
    g1 = Gaussian()
    g1.fwhm = 1.0
    assert_allclose(g1.fwhm, 1.0)
