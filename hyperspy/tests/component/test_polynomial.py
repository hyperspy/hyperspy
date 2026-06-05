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


class TestPolynomial:
    def setup_method(self, method):
        s = hs.signals.Signal1D(np.zeros(1024))
        s.axes_manager[0].offset = -5
        s.axes_manager[0].scale = 0.01
        m = s.create_model()
        m.append(hs.model.components1D.Polynomial(order=2))
        coeff_values = (0.5, 2, 3)
        self.m = m
        s_2d = hs.signals.Signal1D(np.arange(1000).reshape(10, 100))
        self.m_2d = s_2d.create_model()
        self.m_2d.append(hs.model.components1D.Polynomial(order=2))
        s_3d = hs.signals.Signal1D(np.arange(1000).reshape(2, 5, 100))
        self.m_3d = s_3d.create_model()
        self.m_3d.append(hs.model.components1D.Polynomial(order=2))
        data = 50 * np.ones(100)
        s_offset = hs.signals.Signal1D(data)
        self.m_offset = s_offset.create_model()

        # if same component is pased, axes_managers get mixed up, tests
        # sometimes randomly fail
        for _m in [self.m, self.m_2d, self.m_3d]:
            _m[0].a2.value = coeff_values[0]
            _m[0].a1.value = coeff_values[1]
            _m[0].a0.value = coeff_values[2]

    def test_gradient(self):
        poly = self.m[0]
        np.testing.assert_allclose(poly.a2.grad(np.arange(3)), np.array([0, 1, 4]))
        np.testing.assert_allclose(poly.a1.grad(np.arange(3)), np.array([0, 1, 2]))
        np.testing.assert_allclose(poly.a0.grad(np.arange(3)), np.array([1, 1, 1]))

    def test_fitting(self):
        s_2d = self.m_2d.signal
        s_2d.data += 100 * np.array([np.random.randint(50, size=10)] * 100).T
        m_2d = s_2d.create_model()
        m_2d.append(hs.model.components1D.Polynomial(order=1))
        m_2d.multifit(grad="analytical")
        np.testing.assert_allclose(m_2d.red_chisq.data.sum(), 0.0, atol=1e-7)

    @pytest.mark.parametrize(("order"), (2, 12))
    @pytest.mark.parametrize(("uniform"), (True, False))
    @pytest.mark.parametrize(("mapnone"), (True, False))
    @pytest.mark.parametrize(("only_current", "binned"), TRUE_FALSE_2_TUPLE)
    def test_estimate_parameters(self, only_current, binned, uniform, order, mapnone):
        self.m.signal.axes_manager[-1].is_binned = binned
        self.m.assign_current_values_to_all()
        s = self.m.as_signal()
        s.axes_manager[-1].is_binned = binned
        if not uniform:
            s.axes_manager[-1].convert_to_non_uniform_axis()
        p = hs.model.components1D.Polynomial(order=order)
        if mapnone:
            p.parameters[0].map = None
        p.estimate_parameters(s, None, None, only_current=only_current)
        assert p._axes_manager[-1].is_binned == binned
        assert p._axes_manager[-1].is_uniform == uniform
        np.testing.assert_allclose(p.parameters[2].value, 0.5)
        np.testing.assert_allclose(p.parameters[1].value, 2)
        np.testing.assert_allclose(p.parameters[0].value, 3)

    def test_zero_order(self):
        m = self.m_offset
        with pytest.raises(ValueError):
            m.append(hs.model.components1D.Polynomial(order=0))

    def test_2d_signal(self):
        for p in self.m_2d[0].parameters:
            p.map["values"] = p.value
            p.map["is_set"] = True
        s = self.m_2d.as_signal()

        model = s.create_model()
        p = hs.model.components1D.Polynomial(order=2)
        model.append(p)
        p.estimate_parameters(s, 0, 100, only_current=False)
        np.testing.assert_allclose(p.a2.map["values"], 0.5)
        np.testing.assert_allclose(p.a1.map["values"], 2)
        np.testing.assert_allclose(p.a0.map["values"], 3)

    def test_3d_signal(self):
        for p in self.m_3d[0].parameters:
            p.map["values"] = p.value
            p.map["is_set"] = True
        s = self.m_3d.as_signal()

        model = s.create_model()
        p = hs.model.components1D.Polynomial(order=2)
        model.append(p)
        p.estimate_parameters(s, 0, 100, only_current=False)
        np.testing.assert_allclose(p.a2.map["values"], 0.5)
        np.testing.assert_allclose(p.a1.map["values"], 2)
        np.testing.assert_allclose(p.a0.map["values"], 3)

    def test_function_nd(self):
        self.m.assign_current_values_to_all()
        s = self.m.as_signal()
        s = hs.stack([s] * 2)
        p = hs.model.components1D.Polynomial(order=2)
        p.estimate_parameters(s, None, None, only_current=False)
        axis = s.axes_manager.signal_axes[0]
        np.testing.assert_allclose(p.function_nd(axis.axis), s.data)
