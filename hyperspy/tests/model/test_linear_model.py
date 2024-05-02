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

import warnings

import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.component import Component
from hyperspy.components1d import Expression, Gaussian, Offset
from hyperspy.components2d import Gaussian2D
from hyperspy.decorators import lazifyTestClass
from hyperspy.misc.utils import dummy_context_manager
from hyperspy.signals import Signal1D, Signal2D


def test_fit_binned():
    rng = np.random.default_rng(1)
    s = Signal1D(rng.normal(scale=2, size=10000)).get_histogram()
    s.axes_manager[-1].is_binned = True
    g = Gaussian()
    m = s.create_model()
    m.append(g)
    g.sigma.value = 1
    g.centre.value = 0.5
    g.A.value = 1e3
    # model contains free nonlinear parameters
    with pytest.raises(RuntimeError):
        m.fit(optimizer="lstsq")

    g.centre.free = False
    g.sigma.free = False

    m.fit(optimizer="lstsq")
    np.testing.assert_allclose(m[0].A.value, 6132.640632924692, 1)
    np.testing.assert_allclose(m[0].centre.value, 0.5)
    np.testing.assert_allclose(m[0].sigma.value, 1)


@pytest.mark.parametrize("weighted", [False, True])
@lazifyTestClass
class TestMultiFitLinear:
    def setup_method(self):
        rng = np.random.default_rng(1)
        x = rng.random(30)
        shape = rng.random((2, 3, 1))
        X = shape * x
        s = Signal1D(X)
        m = s.create_model()
        self.s, self.m = s, m

    def _post_setup_method(self, weighted):
        """Convenience method to use class parametrize marker"""
        s = self.s
        if weighted:
            s.estimate_poissonian_noise_variance()

    def test_gaussian(self, weighted):
        self._post_setup_method(weighted)
        m = self.m
        L = Gaussian(centre=15.0)
        L.centre.free = L.sigma.free = False
        m.append(L)

        m.fit(optimizer="lstsq")
        single = m.as_signal()
        m.assign_current_values_to_all()
        cm = (
            pytest.warns(UserWarning)
            if weighted and not self.s._lazy
            else dummy_context_manager()
        )
        with cm:
            m.multifit(optimizer="lstsq")
        multi = m.as_signal()

        np.testing.assert_allclose(
            single._get_current_data(), multi._get_current_data()
        )

    def test_map_values_std_isset(self, weighted):
        self._post_setup_method(weighted)
        m = self.m
        L = Gaussian(centre=15.0)
        L.centre.free = L.sigma.free = False
        m.append(L)

        m.multifit()
        nonlinear = L.A.map.copy()

        L.A.map["is_set"] = False
        cm = (
            pytest.warns(UserWarning)
            if weighted and not self.s._lazy
            else dummy_context_manager()
        )
        with cm:
            m.multifit(optimizer="lstsq", calculate_errors=True)
        linear = L.A.map.copy()

        np.testing.assert_allclose(nonlinear["values"], linear["values"])
        np.testing.assert_allclose(nonlinear["std"], linear["std"])
        np.testing.assert_allclose(nonlinear["is_set"], linear["is_set"])

        cm = (
            pytest.warns(UserWarning)
            if weighted and not self.s._lazy
            else dummy_context_manager()
        )
        with cm:
            m.multifit(optimizer="lstsq", calculate_errors=False)
        np.testing.assert_equal(L.A.map["std"], np.nan)

    def test_offset(self, weighted):
        self._post_setup_method(weighted)
        m = self.m
        L = Offset(offset=1.0)
        m.append(L)

        m.fit(optimizer="lstsq")
        single = m.as_signal()
        m.assign_current_values_to_all()
        cm = (
            pytest.warns(UserWarning)
            if weighted and not self.s._lazy
            else dummy_context_manager()
        )
        with cm:
            m.multifit(optimizer="lstsq")
        multi = m.as_signal()
        # compare fits from first pixel
        np.testing.assert_allclose(
            single._get_current_data(), multi._get_current_data()
        )

    def test_channel_switches(self, weighted):
        self._post_setup_method(weighted)
        m = self.m
        m._channel_switches[5:-5] = False
        L = Gaussian(centre=15.0)
        L.centre.free = L.sigma.free = False
        m.append(L)

        m.fit(optimizer="lstsq")
        single = m.as_signal()
        m.assign_current_values_to_all()
        cm = (
            pytest.warns(UserWarning)
            if weighted and not self.s._lazy
            else dummy_context_manager()
        )
        with cm:
            m.multifit(optimizer="lstsq")
        multi = m.as_signal()

        np.testing.assert_allclose(
            single._get_current_data(), multi._get_current_data()
        )

        m.fit()
        single_nonlinear = m.as_signal()
        np.testing.assert_allclose(
            single._get_current_data(), single_nonlinear._get_current_data()
        )

    def test_multifit_ridge(self, weighted):
        pytest.importorskip("sklearn")
        m = self.m
        L = Gaussian(centre=15.0)
        L.set_parameters_not_free(["centre", "sigma"])
        m.append(L)

        if m.signal._lazy:
            with pytest.raises(ValueError):
                m.multifit(optimizer="ridge_regression")
            return
        else:
            m.multifit(optimizer="ridge_regression")


class TestLinearFitting:
    def setup_method(self, method):
        s = hs.signals.Signal1D(np.arange(0, 100) + 20)
        m = s.create_model()
        g1 = hs.model.components1D.Gaussian(A=1000, centre=20, sigma=10)
        g1.sigma.free = False
        g1.centre.free = False
        c = Expression("a*x+b", "line with offset")
        m.extend([g1, c])

        axis = s.axes_manager[-1].axis
        s.data = s.data + g1.function(axis)

        self.s = s
        self.m = m
        self.c = c

    def test_linear_fitting_with_offset(self):
        m = self.m
        c = self.c
        m.fit(optimizer="lstsq")
        expected_values = np.array([1000.0, 1.0, 20.0])
        np.testing.assert_allclose(m.p0, expected_values, rtol=5e-6)

        # Repeat test with offset fixed
        c.b.free = False
        m.fit(optimizer="lstsq")
        np.testing.assert_allclose(m.p0, expected_values[:2], rtol=5e-6)

    def test_fixed_offset_value(self):
        self.m.fit(optimizer="lstsq")
        c = self.c
        c.b.free = False
        constant = self.m._compute_constant_term(component=c)
        np.testing.assert_allclose(constant, c.b.value)

    def test_constant(self):
        self.c.b.value = -5
        self.c.b.free = False
        assert self.c._constant_term == self.c.b.value


@pytest.mark.parametrize("weighted", [False, True])
@lazifyTestClass
class TestFitAlgorithms:
    def setup_method(self, method):
        s = hs.signals.Signal1D(np.arange(1, 100))
        m = s.create_model()
        g1 = hs.model.components1D.Gaussian()
        g1.sigma.free = False
        g1.centre.free = False
        c = Expression("a*x+b", "line with offset")
        m.extend([g1, c])
        self.m = m

    def _post_setup_method(self, weighted):
        """Convenience method to use class parametrize marker"""
        m = self.m
        if weighted:
            variance = np.arange(10, m.signal.data.size - 10, 0.01)
            m.signal.set_noise_variance(Signal1D(variance))
        m.fit()
        self.nonlinear_fit_res = m.as_signal()
        self.nonlinear_fit_std = [p.std for p in m._free_parameters if p.std]

    def test_compare_lstsq(self, weighted):
        self._post_setup_method(weighted)
        m = self.m
        m.fit(optimizer="lstsq")
        lstsq_fit = m.as_signal()
        np.testing.assert_allclose(
            self.nonlinear_fit_res, lstsq_fit._get_current_data(), atol=1e-8
        )
        linear_std = [para.std for para in m._free_parameters if para.std]
        np.testing.assert_allclose(self.nonlinear_fit_std, linear_std, atol=1e-8)

    def test_nonactive_component(self, weighted):
        self._post_setup_method(weighted)
        m = self.m
        m[1].active = False
        m.fit(optimizer="lstsq")
        linear_fit = m.as_signal()
        m.fit()
        nonlinear_fit = m.as_signal()
        np.testing.assert_allclose(
            nonlinear_fit._get_current_data(), linear_fit._get_current_data(), rtol=1e-5
        )

    def test_compare_ridge(self, weighted):
        self._post_setup_method(weighted)
        pytest.importorskip("sklearn")
        m = self.m
        if m.signal._lazy:
            with pytest.raises(ValueError):
                m.fit(optimizer="ridge_regression")
            return
        else:
            m.fit(optimizer="ridge_regression")
        ridge_fit = m.as_signal()
        np.testing.assert_allclose(self.nonlinear_fit_res, ridge_fit.data, rtol=5e-6)
        linear_std = [para.std for para in m._free_parameters if para.std]
        np.testing.assert_allclose(self.nonlinear_fit_std, linear_std, atol=1e-8)


class TestWarningSlowMultifit:
    def setup_method(self, method):
        s = hs.data.two_gaussians().inav[0]
        m = s.create_model()
        g1 = hs.model.components1D.Gaussian(centre=40)
        g2 = hs.model.components1D.Gaussian(centre=55)
        m.extend([g1, g2])

        # make dummy twinning
        g2.centre.twin = g1.centre
        g2.centre.twin_function_expr = "15 + x"
        g2.A.twin = g1.A
        g2.centre.twin_function_expr = "2 * x"

        m.set_parameters_not_free(only_nonlinear=True)

        self.m = m

    def test_active_is_multidimensional_all_active(self):
        m = self.m
        m[0].active_is_multidimensional = True
        m.multifit(optimizer="lstsq")

    def test_active_is_multidimensional(self):
        m = self.m
        component = m[0]
        component.active_is_multidimensional = True
        component._active_array[10] = False
        m[1].active = False
        assert component.active
        with pytest.warns(UserWarning, match="active components that are not active"):
            with pytest.raises(RuntimeError):
                # when we hit the navigation position, where the component
                # is not active
                m.multifit(optimizer="lstsq")
        assert m.signal.axes_manager.indices == (10,)

    def test_set_value_in_non_free_parameter(self):
        m = self.m
        parameter = m[0].centre
        assert parameter.twin is None
        parameter.map["values"][:3] = 50.0
        parameter.map["is_set"][:3] = True
        with pytest.warns(UserWarning, match="model contains non-free parameters"):
            m.multifit(optimizer="lstsq")

    def test_set_value_in_non_free_parameter_twin(self):
        m = self.m
        parameter = m[1].centre
        assert parameter.twin is not None
        parameter.map["values"][:3] = 40.0
        parameter.map["is_set"][:3] = True
        with pytest.warns(UserWarning, match="model contains non-free parameters"):
            m.multifit(optimizer="lstsq")

    def test_set_value_in_free_parameter_twin(self):
        m = self.m
        parameter = m[1].A
        assert parameter.twin is not None
        parameter.map["values"][:3] = 100.0
        parameter.map["is_set"][:3] = True
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m.multifit(optimizer="lstsq")

    def test_rerun_multifit(self):
        # Check that the parameter map values have set consistently at the end
        # of `multifit(optimizer="lstsq")` so that rerunning it doesn't fall
        # back to slow multifit
        m = self.m
        m.multifit(optimizer="lstsq")
        p = m[0].centre
        np.testing.assert_equal(p.map["is_set"], True)
        np.testing.assert_allclose(p.map["values"], p.map["values"][0])
        np.testing.assert_equal(p.map["std"], np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m.multifit(optimizer="lstsq")
        # assert len(record) == 0

    def test_heteroscedastic_variance(self):
        m = self.m
        m.signal.estimate_poissonian_noise_variance()
        with pytest.warns(UserWarning):
            m.multifit(
                optimizer="lstsq", match="noise of the signal is not homoscedastic"
            )


class TestLinearModel2D:
    def setup_method(self, method):
        low, high = -10, 10
        N = 100
        x = y = np.linspace(low, high, N)
        mesh = np.meshgrid(x, y)
        self.mesh, self.x, self.y = mesh, x, y

    @pytest.mark.parametrize("nav2d", [False, True])
    def test_model2D_one_component(self, nav2d):
        mesh, x, y = self.mesh, self.x, self.y
        G1 = Gaussian2D(30, 5.0, 4.0, 0, 0)

        data = G1.function(*mesh)
        s = Signal2D(data)
        s.axes_manager[-2].offset = x[0]
        s.axes_manager[-1].offset = y[0]

        s.axes_manager[-2].scale = x[1] - x[0]
        s.axes_manager[-1].scale = y[1] - y[0]

        if nav2d:
            s = hs.stack([s] * 2)
            s = hs.stack([s] * 3)

        m = s.create_model()
        m.append(G1)

        G1.set_parameters_not_free()
        G1.A.free = True

        m.multifit(optimizer="lstsq", calculate_errors=True)
        diff = s - m.as_signal(show_progressbar=False)
        np.testing.assert_allclose(diff.data, 0.0, atol=1e-7)
        np.testing.assert_allclose(m.p_std[0], 0.0, atol=1e-7)

    @pytest.mark.parametrize("nav2d", [False, True])
    def test_model2D_linear_many_gaussians(self, nav2d):
        mesh, x, y = self.mesh, self.x, self.y
        gausslow, gausshigh = -8, 8
        gauss_step = 8
        X, Y = mesh
        z = np.zeros(X.shape)
        g = Gaussian2D()
        for i in np.arange(gausslow, gausshigh + 1, gauss_step):
            for j in np.arange(gausslow, gausshigh + 1, gauss_step):
                g.centre_x.value = i
                g.centre_y.value = j
                g.A.value = 10
                z += g.function(X, Y)

        s = Signal2D(z)
        s.axes_manager[-2].offset = x[0]
        s.axes_manager[-1].offset = y[0]

        s.axes_manager[-2].scale = x[1] - x[0]
        s.axes_manager[-1].scale = y[1] - y[0]

        if nav2d:
            s = hs.stack([s] * 2)
            s = hs.stack([s] * 3)

        m = s.create_model()
        for i in np.arange(gausslow, gausshigh + 1, gauss_step):
            for j in np.arange(gausslow, gausshigh + 1, gauss_step):
                g = Gaussian2D(centre_x=i, centre_y=j)
                g.set_parameters_not_free()
                g.A.free = True
                m.append(g)

        m.fit(optimizer="lstsq")
        np.testing.assert_allclose(s.data, m.as_signal().data)

    @pytest.mark.parametrize("nav2d", [False, True])
    def test_model2D_polyexpression(self, nav2d):
        poly = "a*x**2 + b*x - c*y**2 + d*y + e"
        P = Expression(poly, "poly")
        P.a.value = 6
        P.b.value = 5
        P.c.value = 4
        P.d.value = 3
        P.e.value = 2

        data = P.function(*self.mesh)
        s = Signal2D(data)

        if nav2d:
            s = hs.stack([s] * 2)
            s = hs.stack([s] * 3)

        m = s.create_model()
        m.append(P)
        m.fit(optimizer="lstsq")
        diff = s - m.as_signal(show_progressbar=False)
        np.testing.assert_allclose(diff.data, 0.0, atol=1e-7)
        np.testing.assert_allclose(m.p_std, 0.0, atol=1e-7)


class TestLinearFitTwins:
    def setup_method(self, method):
        g1 = Gaussian(centre=10)
        g2 = Gaussian(centre=20)
        g3 = Gaussian(centre=30)

        g3.A.twin = g2.A
        g3.A.twin_function_expr = "-0.5*x"
        g2.A.twin = g1.A
        g2.A.twin_function_expr = "-0.5*x"

        g1.A.value = 20
        x = np.linspace(0, 50, 1000)

        y = g1.function(x) + g2.function(x) + g3.function(x)
        s = Signal1D(y)
        s.axes_manager[-1].scale = x[1] - x[0]

        gs = [g1, g2, g3]
        m = s.create_model()
        m.extend(gs)
        self.s, self.m, self.gs = s, m, gs

    def test_without_twins(self):
        gs = self.gs
        m = self.m
        s = self.s
        for g in gs:
            g.sigma.free = False
            g.centre.free = False
            g.A.twin = None

        gs[0].A.value = 1
        m.fit(optimizer="lstsq")

        np.testing.assert_allclose(gs[0].A.value, 20)
        np.testing.assert_allclose(gs[1].A.value, -10)
        np.testing.assert_allclose(gs[2].A.value, 5)
        np.testing.assert_allclose(s.data, m._get_current_data())

    def test_with_twins(self):
        gs = self.gs
        m = self.m
        s = self.s
        for g in gs:
            g.sigma.free = False
            g.centre.free = False

        gs[0].A.value = 1
        m.fit(optimizer="lstsq")

        np.testing.assert_allclose(gs[0].A.value, 20)
        np.testing.assert_allclose(gs[1].A.value, -10)
        np.testing.assert_allclose(gs[2].A.value, 5)
        np.testing.assert_allclose(s.data, m._get_current_data())


def test_compute_constant_term():
    rng = np.random.default_rng(1)
    s = Signal1D(rng.random(10))
    m = s.create_model()
    lin = Expression("a*x + b", name="linear")
    m.append(lin)

    lin.a.value = 2
    lin.b.value = 3
    lin.b.free = False
    np.testing.assert_allclose(m._compute_constant_term(component=lin), 3)


@lazifyTestClass
class TestLinearEdgeCases:
    def setup_method(self, method):
        s = hs.signals.Signal1D(np.arange(100))
        m = s.create_model()
        g1 = Gaussian(centre=10)
        g2 = Gaussian(centre=20)
        m.extend([g1, g2])
        m.set_parameters_not_free(only_nonlinear=True)

        self.m = m

    def test_no_free_parameters(self):
        self.m.set_parameters_not_free()
        with pytest.raises(
            RuntimeError, match="Model does not contain any free components!"
        ):
            self.m.fit(optimizer="lstsq")

    def test_free_nonlinear_parameters(self):
        self.m[1].sigma.free = True
        with pytest.raises(RuntimeError, match=str(self.m[1].sigma)):
            self.m.fit(optimizer="lstsq")

    def test_force_fake_linear_optimizer(self):
        with pytest.raises(ValueError, match="not supported"):
            self.m._linear_fit(optimizer="foo")


class TestTwinnedComponents:
    def setup_method(self):
        s = hs.signals.Signal1D(np.arange(100) + 20)
        m = s.create_model()
        p0 = hs.model.components1D.Polynomial(order=6)
        g1 = Gaussian(A=1000, centre=20, sigma=10)
        g2 = Gaussian(centre=40, A=500, sigma=5)
        g3 = Gaussian(centre=70, A=500, sigma=5)
        g2.A.twin = g1.A
        g2.A.twin_function_expr = "x / 2"

        m.extend([p0, g1, g2, g3])

        axis = s.axes_manager[-1].axis
        s.data = (
            s.data + g1.function(axis) + g2.function(axis) + g3.function(axis)
        ) * 100
        s.change_dtype(np.int64)
        s.add_poissonian_noise()

        m.set_parameters_not_free(only_nonlinear=True)

        self.m = m

    # def test_fixed_chained_twinned_components(self):
    #     # FIXME
    #     m = self.m

    #     m.fit(optimizer="lstsq")
    #     A = m.as_signal()

    #     m[2].A.free = False
    #     m.fit(optimizer="lstsq")
    #     B = m.as_signal()
    #     np.testing.assert_allclose(A.data, B.data, rtol=5E-5)

    def test_fit_fixed_twinned_components_and_std(self):
        m = self.m
        m[1].A.free = False
        m.fit(optimizer="lstsq")
        lstsq_fit = m.as_signal()
        nonlinear_parameters = [p for c in m for p in c.parameters if not p._linear]
        linear_std = [para.std for para in nonlinear_parameters if para.std]

        m.fit()
        nonlinear_fit = m.as_signal()
        nonlinear_std = [para.std for para in nonlinear_parameters if para.std]

        np.testing.assert_allclose(nonlinear_fit.data, lstsq_fit.data, rtol=5e-5)
        np.testing.assert_allclose(nonlinear_std, linear_std)


class MultiLinearCustomComponent(Component):
    def __init__(self, a0=1, a1=1):
        Component.__init__(self, ("a0", "a1"), linear_parameter_list=["a0", "a1"])

        self.a0.value = a0
        self.a1.value = a1

    def function(self, x):
        a0 = self.a0.value
        a1 = self.a1.value
        return a0 + x * a1


class TestCustomComponent:
    def setup_method(self):
        s = hs.signals.Signal1D(np.arange(100))
        m = s.create_model()
        g1 = Gaussian(centre=10)
        g2 = Gaussian(centre=20)
        m.extend([g1, g2])
        m.set_parameters_not_free(only_nonlinear=True)

        self.m = m

    def test_custom_comp_w_two_linear_attributes(self):
        c = MultiLinearCustomComponent()
        self.m.append(c)
        with pytest.raises(AttributeError, match="has more than one free"):
            self.m.fit(optimizer="lstsq")

    def test_custom_comp(self):
        c = MultiLinearCustomComponent()
        c.a0.free = False
        self.m.append(c)
        self.m.fit(optimizer="lstsq")

    def test_compare_custom_comp(self):
        c = MultiLinearCustomComponent()
        c.a0.free = False
        c.a0.value = 0

        self.m.append(c)
        self.m.fit(optimizer="lstsq")
        linear = c.a1.value

        self.m.fit()
        nonlinear = c.a1.value

        np.testing.assert_allclose(linear, nonlinear)


def test_fixed_free_offset():
    s = Signal1D(np.ones(100) * 3)
    m = s.create_model()
    a = Offset(1.0)
    a.offset.free = False
    b = Offset(0.0)
    m.extend((a, b))

    m.fit(optimizer="lstsq")

    np.testing.assert_almost_equal(a.offset.value, 1.0)
    np.testing.assert_almost_equal(b.offset.value, 2.0)


def test_non_uniform_binned():
    s = hs.data.luminescence_signal()
    s.axes_manager[-1].is_binned = True
    m = s.create_model()
    with pytest.raises(ValueError):
        m.fit(optimizer="lstsq")


@pytest.mark.parametrize("navigation_dim", (1, 2, 3))
def test_navigation_shape(navigation_dim):
    rng = np.random.default_rng(1)
    nav_shape = tuple(range(1, navigation_dim + 1))
    s = hs.signals.Signal1D(np.zeros(nav_shape + (200,)))
    g = hs.model.components1D.Gaussian()
    g.sigma.value = 10
    g.centre.value = 100
    g.A.value = 1000
    m = s.create_model()
    m.append(g)
    g.A.map["values"] = rng.integers(low=500, high=1500, size=nav_shape)
    g.A.map["is_set"] = True
    s.data = m.as_signal().data
    s.add_gaussian_noise(0.5)
    m.set_parameters_not_free(only_nonlinear=True)

    g.A.map["values"] = 0
    m.multifit(optimizer="lstsq")

    np.testing.assert_allclose(s, m.as_signal(), atol=2.5)


def test_power_law():
    s = hs.signals.Signal1D(np.zeros(1024))
    s.axes_manager[0].offset = 10
    s.axes_manager[0].scale = 0.01
    m_ref = s.create_model()
    pl_ref = hs.model.components1D.PowerLaw()
    m_ref.append(pl_ref)
    pl_ref.A.value = 100
    pl_ref.r.value = 4

    s = m_ref.as_signal()

    m = s.create_model()
    pl = hs.model.components1D.PowerLaw()
    pl.r.value = pl_ref.r.value
    m.append(pl)
    m.set_parameters_not_free(only_nonlinear=True)
    m.plot()
    m.fit(optimizer="lstsq")

    np.testing.assert_allclose(pl_ref.A.value, pl.A.value)
    np.testing.assert_allclose(pl_ref.r.value, pl.r.value)
    np.testing.assert_allclose(m.as_signal().data, s.data)


def test_lorentzian():
    """
    Lorentzian component has renamed parameters which needs to be taken
    into account in the linear fitting code.
    """
    s = hs.signals.Signal1D(np.zeros(100))
    s.axes_manager[0].offset = 10
    s.axes_manager[0].scale = 0.1
    m_ref = s.create_model()
    l_ref = hs.model.components1D.Lorentzian()
    m_ref.append(l_ref)
    l_ref.A.value = 100
    l_ref.centre.value = 15

    s = m_ref.as_signal()

    m = s.create_model()
    lorentzian = hs.model.components1D.Lorentzian()
    lorentzian.centre.value = l_ref.centre.value
    m.append(lorentzian)
    m.set_parameters_not_free(only_nonlinear=True)
    m.plot()
    m.fit(optimizer="lstsq")

    np.testing.assert_allclose(l_ref.A.value, lorentzian.A.value)
    np.testing.assert_allclose(l_ref.centre.value, lorentzian.centre.value)
    np.testing.assert_allclose(l_ref.gamma.value, lorentzian.gamma.value)
    np.testing.assert_allclose(m.as_signal().data, s.data)


@pytest.mark.parametrize("nav_dim", (0, 1, 2))
def test_expression_multiple_linear_parameter(nav_dim):
    """
    This test checks that linear fitting works
     - single and multidimensional fit (warning raise)
     - multiple free parameters for the same component (different code path)
    """
    s_ref = hs.signals.Signal1D(np.ones(20))
    p_ref = hs.model.components1D.Polynomial(order=2, a0=25, a1=-50, a2=2.5)

    m_ref = s_ref.create_model()
    m_ref.extend([p_ref])
    s = m_ref.as_signal()

    if nav_dim >= 1:
        s = hs.stack([s] * 2)
    if nav_dim == 2:
        s = hs.stack([s] * 3)

    m = s.create_model()
    p = hs.model.components1D.Polynomial(order=2)
    m.append(p)
    m.multifit(optimizer="lstsq")

    np.testing.assert_allclose(p_ref.a0.value, p.a0.value)
    np.testing.assert_allclose(p_ref.a1.value, p.a1.value)
    np.testing.assert_allclose(p_ref.a2.value, p.a2.value)
    np.testing.assert_allclose(m.as_signal().data, s.data)
    if nav_dim >= 1:
        np.testing.assert_allclose(p.a0.map["values"].mean(), p_ref.a0.value)
        np.testing.assert_allclose(p.a1.map["values"].mean(), p_ref.a1.value)
        np.testing.assert_allclose(p.a2.map["values"].mean(), p_ref.a2.value)
