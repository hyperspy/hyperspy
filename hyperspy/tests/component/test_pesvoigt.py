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
import pytest

#Legacy test, to be removed in v2.0
from hyperspy.components1d import PESVoigt, Voigt
from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.signals import Signal1D

TRUE_FALSE_2_TUPLE = [p for p in itertools.product((True, False), repeat=2)]


def test_function():
    g = PESVoigt()
    g.area.value = 5
    g.FWHM.value = 0.5
    g.gamma.value = 0.2
    g.centre.value = 1
    np.testing.assert_allclose(g.function(0), 0.35380168)
    np.testing.assert_allclose(g.function(1), 5.06863535)


@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("only_current", "binned"), TRUE_FALSE_2_TUPLE)
def test_estimate_parameters_binned(only_current, binned, lazy):
    s = Signal1D(np.empty((200,)))
    s.metadata.Signal.binned = binned
    axis = s.axes_manager.signal_axes[0]
    axis.scale = .05
    axis.offset = -5
    g1 = PESVoigt()
    g1.centre.value = 1
    g1.area.value = 5.
    g1.gamma.value = 0.001
    g1.FWHM.value = 0.5
    s.data = g1.function(axis.axis)
    if lazy:
        s = s.as_lazy()
    g2 = PESVoigt()
    factor = axis.scale if binned else 1
    assert g2.estimate_parameters(s, axis.low_value, axis.high_value,
                                  only_current=only_current)
    assert g2.binned == binned
    np.testing.assert_allclose(g2.FWHM.value, 1, 0.5)
    np.testing.assert_allclose(g1.area.value, g2.area.value * factor, 0.04)
    np.testing.assert_allclose(g2.centre.value, 1, 1e-3)


def test_legacy():
    """Legacy test, to be removed in v2.0."""
    with pytest.warns(
        VisibleDeprecationWarning,
        match="API of the `Voigt` component will change",
    ):
        g = Voigt(legacy=True)
        g.area.value = 5
        g.FWHM.value = 0.5
        g.gamma.value = 0.2
        g.centre.value = 1
        np.testing.assert_allclose(g.function(0), 0.35380168)
        np.testing.assert_allclose(g.function(1), 5.06863535)
