# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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
from distutils.version import LooseVersion

import numpy as np
import pytest
import sympy

from hyperspy.components1d import SkewNormal
from hyperspy.signals import Signal1D
from hyperspy.utils import stack

TRUE_FALSE_2_TUPLE = [p for p in itertools.product((True, False), repeat=2)]

pytestmark = pytest.mark.skipif(LooseVersion(sympy.__version__) <
                                LooseVersion("1.3"),
                                reason="This test requires SymPy >= 1.3")


def test_function():
    g = SkewNormal()
    g.A.value = 2.5
    g.x0.value = 0
    g.scale.value = 1
    g.shape.value = 0
    np.testing.assert_allclose(g.function(0), 1, rtol=3e-3)
    np.testing.assert_allclose(g.function(6), 1.52e-8, rtol=1e-3)
    g.A.value = 5
    g.x0.value = 4
    g.scale.value = 3
    g.shape.value = 2
    np.testing.assert_allclose(g.function(0), 6.28e-3, rtol=1e-3)
    np.testing.assert_allclose(g.function(g.mean), 2.855, rtol=1e-3)


@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("only_current", "binned"), TRUE_FALSE_2_TUPLE)
def test_estimate_parameters_binned(only_current, binned, lazy):
    s = Signal1D(np.empty((300,)))
    s.metadata.Signal.binned = binned
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.2
    axis.offset = -10
    g1 = SkewNormal(A=2, x0=2, scale=10, shape=5)
    s.data = g1.function(axis.axis)
    if lazy:
        s = s.as_lazy()
    g2 = SkewNormal()
    factor = axis.scale if binned else 1
    assert g2.estimate_parameters(s, axis.low_value, axis.high_value,
                                  only_current=only_current)
    assert g2.binned == binned
    np.testing.assert_allclose(g1.A.value, g2.A.value * factor)
    assert abs(g2.x0.value - g1.x0.value) <= 0.002
    assert abs(g2.shape.value - g1.shape.value) <= 0.01
    assert abs(g2.scale.value - g1.scale.value) <= 0.01


@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("binned"), (True, False))
def test_function_nd(binned, lazy):
    s = Signal1D(np.empty((300,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.2
    axis.offset = -10
    g1 = SkewNormal(A=2, x0=2, scale=10, shape=5)
    s.data = g1.function(axis.axis)
    s.metadata.Signal.binned = binned
    s2 = stack([s] * 2)
    if lazy:
        s2 = s2.as_lazy()
    g2 = SkewNormal()
    factor = axis.scale if binned else 1
    assert g2.estimate_parameters(s2, axis.low_value, axis.high_value, False)
    assert g2.binned == binned
    np.testing.assert_allclose(g2.function_nd(axis.axis) * factor, s2.data, 0.06)


def test_util_mean_get():
    g1 = SkewNormal()
    g1.shape.value = 0
    g1.scale.value = 1
    g1.x0.value = 1
    np.testing.assert_allclose(g1.mean, 1.0)


def test_util_variance_get():
    g1 = SkewNormal()
    g1.shape.value = 0
    g1.scale.value = 2
    g1.x0.value = 1
    np.testing.assert_allclose(g1.variance, 4.0)


def test_util_skewness_get():
    g1 = SkewNormal()
    g1.shape.value = 30
    g1.scale.value = 1
    g1.x0.value = 1
    np.testing.assert_allclose(g1.skewness, 0.99, rtol=1e-3)


def test_util_mode_get():
    g1 = SkewNormal()
    g1.shape.value = 0
    g1.scale.value = 1
    g1.x0.value = 1
    np.testing.assert_allclose(g1.mode, 1.0)
