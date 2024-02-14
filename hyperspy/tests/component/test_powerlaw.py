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

from hyperspy.components1d import PowerLaw
from hyperspy.signals import Signal1D
from hyperspy.utils import stack

TRUE_FALSE_2_TUPLE = [p for p in itertools.product((True, False), repeat=2)]


def test_function():
    g = PowerLaw()
    g.A.value = 1
    g.r.value = 2
    g.origin.value = 3
    assert g.function(2) == 1
    assert g.function(1) == 0.25


def test_linear_override():
    g = PowerLaw()
    for para in g.parameters:
        if para is g.A:
            assert para._linear
        else:
            assert not para._linear


@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("only_current", "binned"), TRUE_FALSE_2_TUPLE)
def test_estimate_parameters_binned(only_current, binned, lazy):
    s = Signal1D(np.empty((100,)))
    s.axes_manager.signal_axes[0].is_binned = binned
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g1 = PowerLaw(50015.156, 1.2)
    s.data = g1.function(axis.axis)
    if lazy:
        s = s.as_lazy()
    g2 = PowerLaw()
    factor = axis.scale if binned else 1
    assert g2.estimate_parameters(
        s, axis.low_value, axis.high_value, only_current=only_current
    )
    assert g2._axes_manager[-1].is_binned == binned
    # error of the estimate function is rather large, esp. when binned=FALSE
    np.testing.assert_allclose(g1.A.value, g2.A.value * factor, rtol=0.05)
    assert abs(g2.r.value - g1.r.value) <= 2e-2


@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("binned"), (True, False))
def test_function_nd(binned, lazy):
    s = Signal1D(np.empty((100,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g1 = PowerLaw(50015.156, 1.2)
    s.data = g1.function(axis.axis)
    s.axes_manager.signal_axes[0].is_binned = binned
    s2 = stack([s] * 2)
    if lazy:
        s = s.as_lazy()
    g2 = PowerLaw()
    factor = axis.scale if binned else 1
    g2.estimate_parameters(s2, axis.low_value, axis.high_value, False)
    assert g2._axes_manager[-1].is_binned == binned
    np.testing.assert_allclose(g2.function_nd(axis.axis) * factor, s2.data, rtol=0.05)
