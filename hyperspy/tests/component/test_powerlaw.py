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

@pytest.mark.parametrize(("only_current", "binned"), TRUE_FALSE_2_TUPLE)
def test_estimate_parameters_binned(only_current, binned):
    s = Signal1D(np.empty((100,)))
    s.metadata.Signal.binned = binned
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g1 = PowerLaw(50015.156, 1.2)
    s.data = g1.function(axis.axis)
    g2 = PowerLaw()
    factor = axis.scale if binned else 1
    assert g2.estimate_parameters(s, axis.low_value, axis.high_value,
                                  only_current=only_current)
    assert g2.binned == binned
    # error of the estimate function is rather large, esp. when binned=FALSE
    np.testing.assert_allclose(g1.A.value, g2.A.value * factor, rtol=0.05)
    assert abs(g2.r.value - g1.r.value) <= 2e-2

@pytest.mark.parametrize(("binned"), (True, False))
def test_function_nd(binned):
    s = Signal1D(np.empty((100,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g1 = PowerLaw(50015.156, 1.2)
    s.data = g1.function(axis.axis)
    s.metadata.Signal.binned = binned
    s2 = stack([s] * 2)
    g2 = PowerLaw()
    factor = axis.scale if binned else 1
    g2.estimate_parameters(s2, axis.low_value, axis.high_value, False)
    assert g2.binned == binned
    np.testing.assert_allclose(g2.function_nd(axis.axis) * factor, s2.data, rtol=0.05)
