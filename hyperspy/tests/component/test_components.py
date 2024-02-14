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

import inspect
import itertools

import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy import components1d
from hyperspy.component import Component
from hyperspy.models.model1d import Model1D

TRUE_FALSE_2_TUPLE = [p for p in itertools.product((True, False), repeat=2)]


def get_components1d_name_list():
    components1d_name_list = []
    for c_name in dir(components1d):
        obj = getattr(components1d, c_name)
        if inspect.isclass(obj) and issubclass(obj, Component):
            components1d_name_list.append(c_name)
    return components1d_name_list


@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in true_divide:RuntimeWarning"
)
@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in true_divide:RuntimeWarning"
)
@pytest.mark.filterwarnings("ignore:invalid value encountered in cos:RuntimeWarning")
@pytest.mark.parametrize("component_name", get_components1d_name_list())
def test_creation_components1d(component_name):
    s = hs.signals.Signal1D(np.zeros(1024))
    s.axes_manager[0].offset = 100
    s.axes_manager[0].scale = 0.01

    kwargs = {}
    if component_name == "ScalableFixedPattern":
        kwargs["signal1D"] = s
    elif component_name == "Expression":
        kwargs.update({"expression": "a*x+b", "name": "linear"})
    elif component_name == "Bleasdale":
        # This component only works with numexpr.
        pytest.importorskip("numexpr")

    component = getattr(components1d, component_name)(**kwargs)
    component.function(np.arange(1, 100))

    # Do a export/import cycle to check all the components can be re-created.
    m = s.create_model()
    m.append(component)
    model_dict = m.as_dictionary()

    m2 = s.create_model()
    m2._load_dictionary(model_dict)


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


class TestOffset:
    def setup_method(self, method):
        s = hs.signals.Signal1D(np.zeros(10))
        s.axes_manager[0].scale = 0.01
        m = s.create_model()
        m.append(hs.model.components1D.Offset())
        m[0].offset.value = 10
        self.m = m

    @pytest.mark.parametrize(("uniform"), (True, False))
    @pytest.mark.parametrize(("only_current", "binned"), TRUE_FALSE_2_TUPLE)
    def test_estimate_parameters(self, only_current, binned, uniform):
        self.m.signal.axes_manager[-1].is_binned = binned
        s = self.m.as_signal()
        if not uniform:
            s.axes_manager[-1].convert_to_non_uniform_axis()
        assert s.axes_manager[-1].is_binned == binned
        o = hs.model.components1D.Offset()
        o.estimate_parameters(s, None, None, only_current=only_current)
        assert o._axes_manager[-1].is_binned == binned
        assert o._axes_manager[-1].is_uniform == uniform
        np.testing.assert_allclose(o.offset.value, 10)

    @pytest.mark.parametrize(("uniform"), (True, False))
    @pytest.mark.parametrize(("binned"), (True, False))
    def test_function_nd(self, binned, uniform):
        self.m.signal.axes_manager[-1].is_binned = binned
        s = self.m.as_signal()
        s = hs.stack([s] * 2)
        o = hs.model.components1D.Offset()
        o.estimate_parameters(s, None, None, only_current=False)
        assert o._axes_manager[-1].is_binned == binned
        axis = s.axes_manager.signal_axes[0]
        factor = axis.scale if binned else 1
        np.testing.assert_allclose(o.function_nd(axis.axis) * factor, s.data)

    def test_constant_term(self):
        m = self.m
        o = m[0]
        o.offset.free = True
        assert o._constant_term == 0

        o.offset.free = False
        assert o._constant_term == o.offset.value


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
        # This code should run smoothly, any exceptions should trigger failure
        s = self.m_2d.as_signal()
        model = Model1D(s)
        p = hs.model.components1D.Polynomial(order=2)
        model.append(p)
        p.estimate_parameters(s, 0, 100, only_current=False)
        np.testing.assert_allclose(p.a2.map["values"], 0.5)
        np.testing.assert_allclose(p.a1.map["values"], 2)
        np.testing.assert_allclose(p.a0.map["values"], 3)

    def test_3d_signal(self):
        # This code should run smoothly, any exceptions should trigger failure
        s = self.m_3d.as_signal()
        model = Model1D(s)
        p = hs.model.components1D.Polynomial(order=2)
        model.append(p)
        p.estimate_parameters(s, 0, 100, only_current=False)
        np.testing.assert_allclose(p.a2.map["values"], 0.5)
        np.testing.assert_allclose(p.a1.map["values"], 2)
        np.testing.assert_allclose(p.a0.map["values"], 3)

    def test_function_nd(self):
        s = self.m.as_signal()
        s = hs.stack([s] * 2)
        p = hs.model.components1D.Polynomial(order=2)
        p.estimate_parameters(s, None, None, only_current=False)
        axis = s.axes_manager.signal_axes[0]
        np.testing.assert_allclose(p.function_nd(axis.axis), s.data)


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
        s = self.m.as_signal()
        s2 = hs.stack([s] * 2)
        g = hs.model.components1D.Gaussian()
        g.estimate_parameters(s2, None, None, only_current=False)
        assert g._axes_manager[-1].is_binned == binned
        axis = s.axes_manager.signal_axes[0]
        factor = axis.scale if binned else 1
        np.testing.assert_allclose(g.function_nd(axis.axis) * factor, s2.data)


class TestScalableFixedPattern:
    def setup_method(self, method):
        s = hs.signals.Signal1D(np.linspace(0.0, 100.0, 10))
        s1 = hs.signals.Signal1D(np.linspace(0.0, 1.0, 10))
        s.axes_manager[0].scale = 0.1
        s1.axes_manager[0].scale = 0.1
        self.s = s
        self.pattern = s1

    def test_position(self):
        s1 = self.pattern
        fp = hs.model.components1D.ScalableFixedPattern(s1)
        assert fp._position is fp.shift

    def test_both_unbinned(self):
        s = self.s
        s1 = self.pattern
        s.axes_manager[-1].is_binned = False
        s1.axes_manager[-1].is_binned = False
        m = s.create_model()
        fp = hs.model.components1D.ScalableFixedPattern(s1)
        m.append(fp)
        fp.xscale.free = False
        fp.shift.free = False
        m.fit()
        np.testing.assert_allclose(fp.yscale.value, 100)

    @pytest.mark.parametrize(("uniform"), (True, False))
    def test_both_binned(self, uniform):
        s = self.s
        s1 = self.pattern
        s.axes_manager[-1].is_binned = True
        s1.axes_manager[-1].is_binned = True
        if not uniform:
            s.axes_manager[0].convert_to_non_uniform_axis()
            s1.axes_manager[0].convert_to_non_uniform_axis()
        m = s.create_model()
        fp = hs.model.components1D.ScalableFixedPattern(s1)
        m.append(fp)
        fp.xscale.free = False
        fp.shift.free = False
        m.fit()
        np.testing.assert_allclose(fp.yscale.value, 100)

    def test_pattern_unbinned_signal_binned(self):
        s = self.s
        s1 = self.pattern
        s.axes_manager[-1].is_binned = True
        s1.axes_manager[-1].is_binned = False
        m = s.create_model()
        fp = hs.model.components1D.ScalableFixedPattern(s1)
        m.append(fp)
        fp.xscale.free = False
        fp.shift.free = False
        m.fit()
        np.testing.assert_allclose(fp.yscale.value, 1000)

    def test_pattern_binned_signal_unbinned(self):
        s = self.s
        s1 = self.pattern
        s.axes_manager[-1].is_binned = False
        s1.axes_manager[-1].is_binned = True
        m = s.create_model()
        fp = hs.model.components1D.ScalableFixedPattern(s1)
        m.append(fp)
        fp.xscale.free = False
        fp.shift.free = False
        m.fit()
        np.testing.assert_allclose(fp.yscale.value, 10)

    def test_function(self):
        s = self.s
        s1 = self.pattern
        fp = hs.model.components1D.ScalableFixedPattern(s1, interpolate=False)
        m = s.create_model()
        m.append(fp)
        m.fit(grad="analytical")
        x = s.axes_manager[0].axis
        np.testing.assert_allclose(s.data, fp.function(x))
        np.testing.assert_allclose(fp.function(x), fp.function_nd(x))

    def test_function_nd(self):
        s = self.s
        s1 = self.pattern
        fp = hs.model.components1D.ScalableFixedPattern(s1)
        s_multi = hs.stack([s] * 3)
        m = s_multi.create_model()
        m.append(fp)
        fp.yscale.map["values"] = [1.0, 0.5, 1.0]
        fp.xscale.map["values"] = [1.0, 1.0, 0.75]
        results = fp.function_nd(s.axes_manager[0].axis)
        expected = np.array([s1.data * v for v in [1, 0.5, 0.75]])
        np.testing.assert_allclose(results, expected)

    @pytest.mark.parametrize("interpolate", [True, False])
    def test_recreate_component(self, interpolate):
        s = self.s
        s1 = self.pattern
        fp = hs.model.components1D.ScalableFixedPattern(s1, interpolate=interpolate)
        assert fp.yscale._linear
        assert not fp.xscale._linear
        assert not fp.shift._linear

        m = s.create_model()
        m.append(fp)
        model_dict = m.as_dictionary()

        m2 = s.create_model()
        m2._load_dictionary(model_dict)
        assert m2[0].interpolate == interpolate
        np.testing.assert_allclose(m2[0].signal.data, s1.data)
        assert m2[0].yscale._linear
        assert not m2[0].xscale._linear
        assert not m2[0].shift._linear


class TestHeavisideStep:
    def setup_method(self, method):
        self.c = hs.model.components1D.HeavisideStep()

    def test_integer_values(self):
        c = self.c
        np.testing.assert_array_almost_equal(
            c.function(np.array([-1, 0, 2])), np.array([0, 0.5, 1])
        )

    def test_float_values(self):
        c = self.c
        np.testing.assert_array_almost_equal(
            c.function(np.array([-0.5, 0.5, 2])), np.array([0, 1, 1])
        )

    def test_not_sorted(self):
        c = self.c
        np.testing.assert_array_almost_equal(
            c.function(np.array([3, -0.1, 0])), np.array([1, 0, 0.5])
        )

    def test_gradients(self):
        c = self.c
        np.testing.assert_array_almost_equal(
            c.A.grad(np.array([3, -0.1, 0])), np.array([1, 0, 0.5])
        )


#        np.testing.assert_array_almost_equal(c.n.grad(np.array([3, -0.1, 0])),
#                                             np.array([1, 1, 1]))
