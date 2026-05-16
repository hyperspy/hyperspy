# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

import hyperspy.api as hs

sqrt2pi = np.sqrt(2 * np.pi)
sigma2fwhm = 2 * np.sqrt(2 * np.log(2))

TRUE_FALSE_2_TUPLE = [p for p in itertools.product((True, False), repeat=2)]


def test_function():
    g = hs.model.components1D.Gaussian()
    g.centre.value = 1
    g.sigma.value = 2 / sigma2fwhm
    g.A.value = 3 * sqrt2pi * g.sigma.value
    np.testing.assert_allclose(g.function(2), 1.5)
    np.testing.assert_allclose(g.function(1), 3)


@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("uniform"), (True, False))
@pytest.mark.parametrize(("only_current", "binned"), TRUE_FALSE_2_TUPLE)
def test_estimate_parameters_binned(only_current, binned, lazy, uniform):
    s = hs.signals.Signal1D(np.empty((100,)))
    s.axes_manager.signal_axes[0].is_binned = binned
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 1
    axis.offset = -20
    g1 = hs.model.components1D.Gaussian(50015.156, 10 / sigma2fwhm, 10)
    s.data = g1.function(axis.axis)
    if not uniform:
        axis.convert_to_non_uniform_axis()
    if lazy:
        s = s.as_lazy()
    g2 = hs.model.components1D.Gaussian()
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
    np.testing.assert_allclose(g1.A.value, g2.A.value * factor)
    assert abs(g2.centre.value - g1.centre.value) <= 1e-3
    assert abs(g2.sigma.value - g1.sigma.value) <= 0.1


def test_estimate_parameters_negative_scale():
    s = hs.signals.Signal1D(np.empty((100,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = -1
    axis.offset = 100
    g1 = hs.model.components1D.Gaussian(50015.156, 15 / sigma2fwhm, 50)
    s.data = g1.function(axis.axis)

    g2 = hs.model.components1D.Gaussian()
    with pytest.raises(ValueError):
        g2.estimate_parameters(s, 40, 60)
    assert g2.estimate_parameters(s, 90, 10)
    np.testing.assert_allclose(g1.A.value, g2.A.value)
    assert abs(g2.centre.value - g1.centre.value) <= 1e-3
    assert abs(g2.sigma.value - g1.sigma.value) <= 0.1


@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("binned"), (True, False))
def test_function_nd(binned, lazy):
    s = hs.signals.Signal1D(np.empty((100,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 1
    axis.offset = -20
    g1 = hs.model.components1D.Gaussian(50015.156, 10 / sigma2fwhm, 10)
    s.data = g1.function(axis.axis)
    s.axes_manager.signal_axes[0].is_binned = binned
    s2 = hs.stack([s] * 2)
    if lazy:
        s2 = s2.as_lazy()
    g2 = hs.model.components1D.Gaussian()
    factor = axis.scale if binned else 1
    g2.estimate_parameters(s2, axis.low_value, axis.high_value, False)
    assert g2._axes_manager[-1].is_binned == binned
    np.testing.assert_allclose(g2.function_nd(axis.axis) * factor, s2.data)


def test_util_fwhm_set():
    g1 = hs.model.components1D.Gaussian()
    g1.fwhm = 1.0
    np.testing.assert_allclose(g1.sigma.value, 1.0 / sigma2fwhm)


def test_util_fwhm_get():
    g1 = hs.model.components1D.Gaussian()
    g1.sigma.value = 1.0
    np.testing.assert_allclose(g1.fwhm, 1.0 * sigma2fwhm)


def test_util_fwhm_getset():
    g1 = hs.model.components1D.Gaussian()
    g1.fwhm = 1.0
    np.testing.assert_allclose(g1.fwhm, 1.0)


def test_util_height_set():
    g1 = hs.model.components1D.Gaussian()
    g1.sigma.value = 3.0
    g1.height = 2.0 / sqrt2pi
    np.testing.assert_allclose(g1.A.value, 6)


def test_util_height_get():
    g1 = hs.model.components1D.Gaussian()
    g1.sigma.value = 4.0
    g1.A.value = sqrt2pi * 8
    np.testing.assert_allclose(g1.height, 2)


def test_util_height_getset():
    g1 = hs.model.components1D.Gaussian()
    g1.height = 4.0
    np.testing.assert_allclose(g1.height, 4.0)


class TestGaussian:
    def setup_method(self, method):
        s = hs.signals.Signal1D(np.zeros(1024))
        s.axes_manager[0].offset = -5
        s.axes_manager[0].scale = 0.01
        m = s.create_model()
        m.append(hs.model.components1D.Gaussian())
        m[0].sigma.value = 0.5
        m[0].centre.value = 1
        m[0].A.value = 2
        self.m = m

    @pytest.mark.parametrize(("only_current", "binned"), TRUE_FALSE_2_TUPLE)
    def test_estimate_parameters_binned(self, only_current, binned):
        self.m.signal.axes_manager[-1].is_binned = binned
        self.m.assign_current_values_to_all()
        s = self.m.as_signal()
        assert s.axes_manager[-1].is_binned == binned
        g = hs.model.components1D.Gaussian()
        g.estimate_parameters(s, None, None, only_current=only_current)
        assert g._axes_manager[-1].is_binned == binned
        np.testing.assert_allclose(g.sigma.value, 0.5)
        np.testing.assert_allclose(g.A.value, 2)
        np.testing.assert_allclose(g.centre.value, 1)

    @pytest.mark.parametrize("binned", (True, False))
    def test_function_nd(self, binned):
        self.m.signal.axes_manager[-1].is_binned = binned
        self.m.assign_current_values_to_all()
        s = self.m.as_signal()
        s2 = hs.stack([s] * 2)
        g = hs.model.components1D.Gaussian()
        g.estimate_parameters(s2, None, None, only_current=False)
        assert g._axes_manager[-1].is_binned == binned
        axis = s.axes_manager.signal_axes[0]
        factor = axis.scale if binned else 1
        np.testing.assert_allclose(g.function_nd(axis.axis) * factor, s2.data)
