# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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

from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.signals import Signal1D

from exspy.components import PESVoigt


TRUE_FALSE_2_TUPLE = [p for p in itertools.product((True, False), repeat=2)]


def test_function():
    g = PESVoigt()
    g.area.value = 5
    g.FWHM.value = 0.5
    g.gamma.value = 0.2
    g.centre.value = 1
    np.testing.assert_allclose(g.function(0), 0.35380168)
    np.testing.assert_allclose(g.function(1), 5.06863535)
    assert g._position is g.centre

def test_function_resolution():
    g = PESVoigt()
    g.area.value = 5
    g.FWHM.value = 0.5
    g.gamma.value = 0.2
    g.centre.value = 1
    g.resolution.value = 0.7
    np.testing.assert_allclose(g.function(0), 0.5335443)
    np.testing.assert_allclose(g.function(1), 3.70472923)
    assert g._position is g.centre

def test_function_spinorbit():
    g = PESVoigt()
    g.area.value = 5
    g.FWHM.value = 0.5
    g.gamma.value = 0.2
    g.centre.value = 1
    g.spin_orbit_splitting=True
    spin_orbit_branching_ratio=0.4
    spin_orbit_splitting_energy=0.72
    np.testing.assert_allclose(g.function(0), 1.553312)
    np.testing.assert_allclose(g.function(1), 5.612734)
    assert g._position is g.centre

def test_function_shirleybackground():
    g = PESVoigt()
    g.area.value = 5
    g.FWHM.value = 0.5
    g.gamma.value = 0.2
    g.centre.value = 1
    g.shirley_background.value = 1.5
    g.shirley_background.active = True
    np.testing.assert_allclose(g.function(0), 0.35380168)
    np.testing.assert_allclose(g.function(1), 5.06863535)
    assert g._position is g.centre

@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("uniform"), (True, False))
@pytest.mark.parametrize(("mapnone"), (True, False))
@pytest.mark.parametrize(("only_current", "binned"), TRUE_FALSE_2_TUPLE)
def test_estimate_parameters_binned(only_current, binned, lazy, uniform, mapnone):
    s = Signal1D(np.empty((200,)))
    s.axes_manager.signal_axes[0].is_binned = binned
    axis = s.axes_manager.signal_axes[0]
    axis.scale = .05
    axis.offset = -5
    g1 = PESVoigt()
    g1.centre.value = 1
    g1.area.value = 5.
    g1.gamma.value = 0.001
    g1.FWHM.value = 0.5
    s.data = g1.function(axis.axis)
    if not uniform:
        axis.convert_to_non_uniform_axis()
    if lazy:
        s = s.as_lazy()
    g2 = PESVoigt()
    if binned and uniform:
        factor = axis.scale
    elif binned:
        factor = np.gradient(axis.axis)
    else:
        factor = 1
    if mapnone:
        g2.area.map = None
    assert g2.estimate_parameters(s, axis.low_value, axis.high_value,
                                  only_current=only_current)
    assert g2._axes_manager[-1].is_binned == binned
    np.testing.assert_allclose(g2.FWHM.value, 1, 0.5)
    np.testing.assert_allclose(g1.area.value, g2.area.value * factor, 0.04)
    np.testing.assert_allclose(g2.centre.value, 1, 1e-3)
    

