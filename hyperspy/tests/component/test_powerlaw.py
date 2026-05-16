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

TRUE_FALSE_2_TUPLE = [p for p in itertools.product((True, False), repeat=2)]


def test_function():
    g = hs.model.components1D.PowerLaw()
    g.A.value = 1
    g.r.value = 2
    g.origin.value = 3
    assert g.function(2) == 1
    assert g.function(1) == 0.25


def test_linear_override():
    g = hs.model.components1D.PowerLaw()
    for para in g.parameters:
        if para is g.A:
            assert para._linear
        else:
            assert not para._linear


@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("only_current", "binned"), TRUE_FALSE_2_TUPLE)
def test_estimate_parameters_binned(only_current, binned, lazy):
    s = hs.signals.Signal1D(np.empty((100,)))
    s.axes_manager.signal_axes[0].is_binned = binned
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g1 = hs.model.components1D.PowerLaw(50015.156, 1.2)
    s.data = g1.function(axis.axis)
    if lazy:
        s = s.as_lazy()
    g2 = hs.model.components1D.PowerLaw()
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
    s = hs.signals.Signal1D(np.empty((100,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g1 = hs.model.components1D.PowerLaw(50015.156, 1.2)
    s.data = g1.function(axis.axis)
    s.axes_manager.signal_axes[0].is_binned = binned
    s2 = hs.stack([s] * 2)
    if lazy:
        s = s.as_lazy()
    g2 = hs.model.components1D.PowerLaw()
    factor = axis.scale if binned else 1
    g2.estimate_parameters(s2, axis.low_value, axis.high_value, False)
    assert g2._axes_manager[-1].is_binned == binned
    np.testing.assert_allclose(g2.function_nd(axis.axis) * factor, s2.data, rtol=0.05)


class TestPowerLaw:
    def setup_method(self, method):
        s = hs.signals.Signal1D(np.zeros(1024))
        s.axes_manager[0].offset = 100
        s.axes_manager[0].scale = 0.01
        m = s.create_model()
        m.append(hs.model.components1D.PowerLaw())
        m[0].A.value = 1000
        m[0].r.value = 4
        self.m = m
        self.s = s

    @pytest.mark.parametrize(("only_current", "binned"), TRUE_FALSE_2_TUPLE)
    def test_estimate_parameters(self, only_current, binned):
        self.m.signal.axes_manager[-1].is_binned = binned
        self.m.assign_current_values_to_all()
        s = self.m.as_signal()
        assert s.axes_manager[-1].is_binned == binned
        g = hs.model.components1D.PowerLaw()
        g.estimate_parameters(s, None, None, only_current=only_current)
        assert g._axes_manager[-1].is_binned == binned
        A_value = 1008.4913 if binned else 1006.4378
        r_value = 4.001768 if binned else 4.001752
        np.testing.assert_allclose(g.A.value, A_value)
        np.testing.assert_allclose(g.r.value, r_value)

        if only_current:
            A_value, r_value = 0, 0
        # Test that it all works when calling it with a different signal
        s2 = hs.stack((s, s))
        g.estimate_parameters(s2, None, None, only_current=only_current)
        assert g._axes_manager[-1].is_binned == binned
        np.testing.assert_allclose(g.A.map["values"][1], A_value)
        np.testing.assert_allclose(g.r.map["values"][1], r_value)

    def test_missing_data(self):
        g = hs.model.components1D.PowerLaw()
        self.m.assign_current_values_to_all()
        s = self.m.as_signal()
        s2 = hs.signals.Signal1D(s.data)
        g.estimate_parameters(s2, None, None)

    def test_function_grad_cutoff(self):
        pl = self.m[0]
        pl.left_cutoff.value = 105.0
        axis = self.s.axes_manager[0].axis
        for attr in ["function", "grad_A", "grad_r", "grad_origin"]:
            values = getattr(pl, attr)((axis))
            np.testing.assert_allclose(values[:501], np.zeros((501)))
            assert getattr(pl, attr)((axis))[500] == 0
            getattr(pl, attr)((axis))[502] > 0

    def test_exception_gradient_calculation(self):
        # if this doesn't warn, it means that sympy can compute the gradients
        # and the power law component can be updated.
        with pytest.warns(UserWarning):
            hs.model.components1D.PowerLaw(compute_gradients=True)
