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

from hyperspy.components1d import Lorentzian
from hyperspy.signals import Signal1D
from hyperspy.utils import stack

TRUE_FALSE_2_TUPLE = [p for p in itertools.product((True, False), repeat=2)]


def test_function():
    g = Lorentzian()
    g.A.value = 1.5 * np.pi
    g.gamma.value = 1
    g.centre.value = 2
    np.testing.assert_allclose(g.function(2), 1.5)
    np.testing.assert_allclose(g.function(4), 0.3)


@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("uniform"), (True, False))
@pytest.mark.parametrize(("only_current", "binned"), TRUE_FALSE_2_TUPLE)
def test_estimate_parameters_binned(only_current, binned, lazy, uniform):
    s = Signal1D(np.empty((250,)))
    s.axes_manager.signal_axes[0].is_binned = binned
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.2
    axis.offset = -15
    g1 = Lorentzian(52342, 2, 10)
    s.data = g1.function(axis.axis)
    if not uniform:
        axis.convert_to_non_uniform_axis()
    if lazy:
        s = s.as_lazy()
    g2 = Lorentzian()
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
    np.testing.assert_allclose(g1.A.value, g2.A.value * factor, 0.1)
    assert abs(g2.centre.value - g1.centre.value) <= 0.2
    assert abs(g2.gamma.value - g1.gamma.value) <= 0.1


@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("binned"), (True, False))
def test_function_nd(binned, lazy):
    s = Signal1D(np.empty((250,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.2
    axis.offset = -15
    g1 = Lorentzian(52342, 2, 10)
    s.data = g1.function(axis.axis)
    s.axes_manager.signal_axes[0].is_binned = binned
    s2 = stack([s] * 2)
    if lazy:
        s2 = s2.as_lazy()
    g2 = Lorentzian()
    factor = axis.scale if binned else 1
    g2.estimate_parameters(s2, axis.low_value, axis.high_value, False)
    assert g2._axes_manager[-1].is_binned == binned
    np.testing.assert_allclose(g2.function_nd(axis.axis) * factor, s2.data, 0.16)


def test_util_gamma_getset():
    g1 = Lorentzian()
    g1.gamma.value = 3.0
    np.testing.assert_allclose(g1.gamma.value, 3.0)


def test_util_fwhm_set():
    g1 = Lorentzian()
    g1.fwhm = 1.0
    np.testing.assert_allclose(g1.gamma.value, 0.5)


def test_util_fwhm_get():
    g1 = Lorentzian()
    g1.gamma.value = 2.0
    np.testing.assert_allclose(g1.fwhm, 4.0)


def test_util_fwhm_getset():
    g1 = Lorentzian()
    g1.fwhm = 4.0
    np.testing.assert_allclose(g1.fwhm, 4.0)


def test_util_height_set():
    g1 = Lorentzian()
    g1.gamma.value = 4.0
    g1.height = 2.0 / np.pi
    np.testing.assert_allclose(g1.A.value, 8)


def test_util_height_get():
    g1 = Lorentzian()
    g1.gamma.value = 3.0
    g1.A.value = np.pi * 1.5
    np.testing.assert_allclose(g1.height, 0.5)


def test_util_height_getset():
    g1 = Lorentzian()
    g1.height = 4.0
    np.testing.assert_allclose(g1.height, 4.0)
