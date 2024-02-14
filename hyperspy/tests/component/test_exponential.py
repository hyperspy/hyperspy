# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

import itertools

import numpy as np
import pytest

from hyperspy.components1d import Exponential
from hyperspy.signals import Signal1D
from hyperspy.utils import stack

TRUE_FALSE_2_TUPLE = [p for p in itertools.product((True, False), repeat=2)]


def test_function():
    g = Exponential()
    g.A.value = 10000.0
    g.tau.value = 200.0

    test_value = 200.0
    test_result = g.A.value * np.exp(-test_value / g.tau.value)
    np.testing.assert_allclose(g.function(0.0), g.A.value)
    np.testing.assert_allclose(g.function(test_value), test_result)


@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("uniform"), (True, False))
@pytest.mark.parametrize(("only_current", "binned"), TRUE_FALSE_2_TUPLE)
def test_estimate_parameters_binned(only_current, binned, lazy, uniform):
    s = Signal1D(np.empty((100,)))
    s.axes_manager.signal_axes[0].is_binned = binned
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.2
    axis.offset = 15.0
    g1 = Exponential(A=10005.7, tau=214.3)
    s.data = g1.function(axis.axis)
    if not uniform:
        axis.convert_to_non_uniform_axis()
    if lazy:
        s = s.as_lazy()
    g2 = Exponential()
    if binned and uniform:
        factor = axis.scale
    elif binned:
        factor = np.gradient(axis.axis)
    else:
        factor = 1
    assert g2.estimate_parameters(
        s, axis.low_value, axis.high_value, only_current=only_current
    )
    assert g2._axes_manager[-1].is_binned == binned
    np.testing.assert_allclose(g1.A.value, g2.A.value * factor, rtol=0.05)
    np.testing.assert_allclose(g1.tau.value, g2.tau.value)


@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("binned"), (True, False))
def test_function_nd(binned, lazy):
    s = Signal1D(np.empty((100,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.2
    axis.offset = 15

    g1 = Exponential(A=10005.7, tau=214.3)
    s.data = g1.function(axis.axis)
    s.axes_manager.signal_axes[0].is_binned = binned

    s2 = stack([s] * 2)
    if lazy:
        s2 = s2.as_lazy()
    g2 = Exponential()
    factor = axis.scale if binned else 1.0
    g2.estimate_parameters(s2, axis.low_value, axis.high_value, False)

    assert g2._axes_manager[-1].is_binned == binned
    np.testing.assert_allclose(g2.function_nd(axis.axis) * factor, s2.data, rtol=0.05)


class TestLinearExponential:
    def setup_method(self, method):
        self.E = Exponential()

    def test_properties(self):
        assert self.E.A._linear
        assert not self.E.tau._linear
