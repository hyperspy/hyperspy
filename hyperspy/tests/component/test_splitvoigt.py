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

from hyperspy.components1d import SplitVoigt
from hyperspy.signals import Signal1D
from hyperspy.utils import stack

TRUE_FALSE_2_TUPLE = [p for p in itertools.product((True, False), repeat=2)]

def test_function():
    g = SplitVoigt()
    g.A.value = 2.5
    g.centre.value = 0
    g.sigma1.value = 2
    g.sigma2.value = 2
    assert_allclose(g.function(0), 0.49867785, rtol=1e-3)
    assert_allclose(g.function(6), 0.00553981, rtol=1e-3)


@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("only_current", "binned"), TRUE_FALSE_2_TUPLE)
def test_estimate_parameters_binned(only_current, binned, lazy):
    s = Signal1D(np.empty((100,)))
    s.metadata.Signal.binned = binned
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 1
    axis.offset = -20
    g1 = SplitVoigt(A=20001.0, centre=10.0, sigma1=3.0,sigma2=3.0)
    s.data = g1.function(axis.axis)
    if lazy:
        s = s.as_lazy()
    g2 = SplitVoigt()
    factor = axis.scale if binned else 1
    assert g2.estimate_parameters(s, axis.low_value, axis.high_value,
                                  only_current=only_current)
    assert g2.binned == binned
    assert_allclose(g1.A.value, g2.A.value * factor,rtol=0.2)
    assert abs(g2.centre.value - g1.centre.value) <= 0.1
    assert abs(g2.sigma1.value - g1.sigma1.value) <= 0.1
    assert abs(g2.sigma2.value - g1.sigma2.value) <= 0.1

