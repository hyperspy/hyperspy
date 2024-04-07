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

import numpy as np
import pytest

from hyperspy._signals.signal1d import Signal1D
from hyperspy.components1d import Doniach

sqrt2pi = np.sqrt(2 * np.pi)
sigma2fwhm = 2 * np.sqrt(2 * np.log(2))


def test_function():
    g = Doniach()
    g.centre.value = 1
    g.sigma.value = 2 / sigma2fwhm
    g.A.value = 3 * sqrt2pi * g.sigma.value
    g.alpha.value = 1.0e-7
    np.testing.assert_allclose(g.function(2), 3.151281311482424)
    np.testing.assert_allclose(g.function(1), 7.519884823893001)


@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("uniform"), (True, False))
@pytest.mark.parametrize(("only_current"), (True, False))
@pytest.mark.parametrize(("binned"), (True, False))
def test_estimate_parameters_binned(only_current, binned, lazy, uniform):
    s = Signal1D(np.empty((200,)))
    s.axes_manager.signal_axes[0].is_binned = binned
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.05
    axis.offset = -5
    g1 = Doniach(centre=1, A=5, sigma=1, alpha=0.5)
    s.data = g1.function(axis.axis)
    if not uniform:
        axis.convert_to_non_uniform_axis()
    if lazy:
        s = s.as_lazy()
    g2 = Doniach()
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
    np.testing.assert_allclose(g2.sigma.value, 2.331764, 0.01)
    np.testing.assert_allclose(g1.A.value, g2.A.value * factor, 0.3)
    np.testing.assert_allclose(g2.centre.value, -0.4791825)
