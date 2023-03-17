# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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
from hyperspy.components1d import Gaussian, Expression, Offset
from hyperspy.components2d import Gaussian2D
from hyperspy.datasets.example_signals import EDS_SEM_Spectrum
from hyperspy.decorators import lazifyTestClass
from hyperspy.misc.utils import dummy_context_manager
from hyperspy.signals import Signal1D, Signal2D


def test_fit_binned():
    rng = np.random.default_rng(1)
    s = Signal1D(rng.normal(scale=2, size=10000)).get_histogram()
    s.axes_manager[-1].binned = True
    g = Gaussian()
    m = s.create_model()
    m.append(g)
    g.sigma.value = 1
    g.centre.value = 0.5
    g.A.value = 1e3
    # model contains free nonlinear parameters
    with pytest.raises(RuntimeError):
        m.fit(optimizer='lstsq')


    g.centre.free = False
    g.sigma.free = False

    m.fit(optimizer='lstsq')
    np.testing.assert_allclose(m[0].A.value, 6132.640632924692, 1)
    np.testing.assert_allclose(m[0].centre.value, 0.5)
    np.testing.assert_allclose(m[0].sigma.value, 1)


@pytest.mark.parametrize('weighted', [False, True])
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
        L = Gaussian(centre=15.)
        L.centre.free = L.sigma.free = False
        m.append(L)

        m.fit(optimizer='lstsq')
        single = m.as_signal()
        m.assign_current_values_to_all()
        cm = pytest.warns(UserWarning) if weighted and not self.s._lazy else \
            dummy_context_manager()
        with cm:
            m.multifit(optimizer='lstsq', iterpath='serpentine')
        multi = m.as_signal()

        np.testing.assert_allclose(single(), multi())

    def test_map_values_std_isset(self, weighted):
        self._post_setup_method(weighted)
        m = self.m
        L = Gaussian(centre=15.)
        L.centre.free = L.sigma.free = False
        m.append(L)

        m.multifit(iterpath="serpentine")
        nonlinear = L.A.map.copy()

        L.A.map['is_set'] = False
        cm = pytest.warns(UserWarning) if weighted and not self.s._lazy else \
            dummy_context_manager()
        with cm:
            m.multifit(optimizer='lstsq', calculate_errors=True)
        linear = L.A.map.copy()

        np.testing.assert_allclose(nonlinear['values'], linear['values'])
        np.testing.assert_allclose(nonlinear['std'], linear['std'])
        np.testing.assert_allclose(nonlinear['is_set'], linear['is_set'])

        cm = pytest.warns(UserWarning) if weighted and not self.s._lazy else \
            dummy_context_manager()
        with cm:
            m.multifit(optimizer='lstsq', calculate_errors=False)
        np.testing.assert_equal(L.A.map['std'], np.nan)

    def test_offset(self, weighted):
        self._post_setup_method(weighted)
        m = self.m
        L = Offset(offset=1.)
        m.append(L)

        m.fit(optimizer='lstsq')
        single = m.as_signal()
        m.assign_current_values_to_all()
        cm = pytest.warns(UserWarning) if weighted and not self.s._lazy else \
            dummy_context_manager()
        with cm:
            m.multifit(optimizer='lstsq', iterpath='serpentine')
        multi = m.as_signal()
        # compare fits from first pixel
        np.testing.assert_allclose(single(), multi())

    def test_channel_switches(self, weighted):
        self._post_setup_method(weighted)
        m = self.m
        m.channel_switches[5:-5] = False
        L = Gaussian(centre=15.)
        L.centre.free = L.sigma.free = False
        m.append(L)

        m.fit(optimizer='lstsq')
        single = m.as_signal()
        m.assign_current_values_to_all()
        cm = pytest.warns(UserWarning) if weighted and not self.s._lazy else \
            dummy_context_manager()
        with cm:
           m.multifit(optimizer='lstsq', iterpath='serpentine')
        multi = m.as_signal()

        np.testing.assert_allclose(single(), multi())

        m.fit()
        single_nonlinear = m.as_signal()
        np.testing.assert_allclose(single(), single_nonlinear())

    def test_multifit_ridge(self, weighted):
        pytest.importorskip("sklearn")
        m = self.m
        L = Gaussian(centre=15.)
        L.set_parameters_not_free(['centre', 'sigma'])
        m.append(L)

        if m.signal._lazy:
            with pytest.raises(ValueError):
                m.multifit(optimizer='ridge_regression')
            return
        else:
            m.multifit(optimizer='ridge_regression')


class TestLinearFitting:

    def setup_method(self, method):
        s = EDS_SEM_Spectrum().isig[5.0:15.0]
        m = s.create_model(auto_background=False)
        c = Expression('a*x+b', 'line with offset')
        m.append(c)
        self.s = s
        self.m = m
        self.c = c

    def test_linear_fitting_with_offset(self):
        m = self.m
        c = self.c
        m.fit(optimizer='lstsq')
        expected_values = np.array(
            [933.23065365,
             47822.97407409,
             -5867.61623971,
             56805.50459484]
            )
        np.testing.assert_allclose(m.p0, expected_values, rtol=5E-6)

        # Repeat test with offset fixed
        c.b.free = False
        m.fit(optimizer='lstsq')
        np.testing.assert_allclose(m.p0, expected_values[:3], rtol=5E-6)

    def test_fixed_offset_value(self):
        self.m.fit(optimizer='lstsq')
        c = self.c
        c.b.free = False
        constant = c._compute_constant_term()
        np.testing.assert_allclose(constant, c.b.value)

    def test_constant(self):
        self.c.b.value = -5
        self.c.b.free = False
        assert self.c._constant_term == self.c.b.value


@pytest.mark.parametrize('weighted', [False, True])
@lazifyTestClass
class TestFitAlgorithms:

    def setup_method(self, method):
        s = EDS_SEM_Spectrum().isig[5.0:15.0]
        m = s.create_model(auto_background=False)
        c = Expression('a*x+b', 'line with offset')
        m.append(c)
        self.m = m

    def _post_setup_method(self, weighted):
        """Convenience method to use class parametrize marker"""
        m = self.m
        if weighted:
            variance = np.arange(10, m.signal.data.size-10, 0.01)
            m.signal.set_noise_variance(Signal1D(variance))
        m.fit()
        self.nonlinear_fit_res = m.as_signal()
        self.nonlinear_fit_std = [p.std for p in m._free_parameters if p.std]

    def test_compare_lstsq(self, weighted):
        self._post_setup_method(weighted)
        m = self.m
        m.fit(optimizer='lstsq')
        lstsq_fit = m.as_signal()
        np.testing.assert_allclose(self.nonlinear_fit_res, lstsq_fit(), rtol=5E-6)
        linear_std = [para.std for para in m._free_parameters if para.std]
        np.testing.assert_allclose(self.nonlinear_fit_std, linear_std, rtol=5E-6)

    def test_nonactive_component(self, weighted):
        self._post_setup_method(weighted)
        m = self.m
        m[1].active = False
        m.fit(optimizer='lstsq')
        linear_fit = m.as_signal()
        m.fit()
        nonlinear_fit = m.as_signal()
        np.testing.assert_allclose(nonlinear_fit(), linear_fit(), rtol=1E-5)

    def test_compare_ridge(self, weighted):
        self._post_setup_method(weighted)
        pytest.importorskip("sklearn")
        m = self.m
        if m.signal._lazy:
            with pytest.raises(ValueError):
                m.fit(optimizer='ridge_regression')
            return
        else:
            m.fit(optimizer='ridge_regression')
        ridge_fit = m.as_signal()
        np.testing.assert_allclose(self.nonlinear_fit_res, ridge_fit.data, rtol=5E-6)
        linear_std = [para.std for para in m._free_parameters if para.std]
        np.testing.assert_allclose(self.nonlinear_fit_std, linear_std, rtol=5E-6)


@lazifyTestClass
class TestLinearEELSFitting:

    def setup_method(self, method):
        ll = hs.datasets.artificial_data.get_low_loss_eels_signal()
        cl = hs.datasets.artificial_data.get_core_loss_eels_signal()
        cl.add_elements(('Mn',))
        m = cl.create_model(auto_background=False)
        m[0].onset_energy.value = 673.
        m_convolved = cl.create_model(auto_background=False, ll=ll)
        m_convolved[0].onset_energy.value = 673.
        self.ll, self.cl = ll, cl
        self.m, self.m_convolved = m, m_convolved

    def test_convolved_and_std_error(self):
        m = self.m_convolved
        m.fit(optimizer='lstsq')
        linear = m.as_signal()
        std_linear = m.p_std
        m.fit(optimizer='lm')
        lm = m.as_signal()
        std_lm = m.p_std
        diff = linear - lm
        np.testing.assert_allclose(diff.data.sum(), 0.0, atol=1E-6)
        np.testing.assert_allclose(std_linear, std_lm)

    def test_nonconvolved(self):
        m = self.m
        m.fit(optimizer='lstsq')
        linear = m.as_signal()
        m.fit(optimizer='lm')
        lm = m.as_signal()
        diff = linear - lm
        np.testing.assert_allclose(diff.data.sum(), 0.0, atol=1E-6)


class TestWarningSlowMultifit:

    def setup_method(self, method):
        cl = hs.datasets.artificial_data.get_core_loss_eels_line_scan_signal()
        ll = hs.datasets.artificial_data.get_low_loss_eels_line_scan_signal()
        cl.add_elements(('Mn', 'Fe'))
        m = cl.create_model(ll=ll, auto_background=False, GOS='hydrogenic')
        m.convolved = False
        # make dummy twinning
        m['Fe_L3'].onset_energy.twin = m['Mn_L3'].onset_energy
        offset = m['Fe_L3'].onset_energy.value - m['Mn_L3'].onset_energy.value
        m['Fe_L3'].onset_energy.twin_function_expr = f'{offset} + x'
        m['Fe_L3'].intensity.twin = m['Mn_L3'].intensity
        m['Fe_L3'].intensity.twin_function_expr = '2 * x'

        self.m = m

    def test_convolved(self):
        m = self.m
        m.convolved = True
        with pytest.warns(UserWarning,
                          match="convolution is not supported"):
            m.multifit(optimizer='lstsq')

    def test_active_is_multidimensional_all_active(self):
        m = self.m
        m[0].active_is_multidimensional = True
        m.multifit(optimizer='lstsq')

    def test_active_is_multidimensional(self):
        m = self.m
        component = m[0]
        component.active_is_multidimensional = True
        component._active_array[-2] = False
        assert component.active
        with pytest.warns(UserWarning,
                          match="active components that are not active"):
            with pytest.raises(RuntimeError):
                # when we hit the navigation position, where the component
                # is not active
                m.multifit(optimizer="lstsq")
        assert m.signal.axes_manager.indices == (10, )

    def test_set_value_in_non_free_parameter(self):
        m = self.m
        parameter = m[0].onset_energy
        assert parameter.twin is None
        parameter.map['values'][:3] = 650.
        parameter.map['is_set'][:3] = True
        with pytest.warns(UserWarning,
                          match="model contains non-free parameters"):
            m.multifit(optimizer="lstsq")

    def test_set_value_in_non_free_parameter_twin(self):
        m = self.m
        parameter = m[1].onset_energy
        assert parameter.twin is not None
        parameter.map['values'][:3] = 660.
        parameter.map['is_set'][:3] = True
        with pytest.warns(UserWarning,
                          match="model contains non-free parameters"):
            m.multifit(optimizer="lstsq")

    def test_set_value_in_free_parameter_twin(self):
        m = self.m
        parameter = m[1].intensity
        assert parameter.twin is not None
        parameter.map['values'][:3] = 100.
        parameter.map['is_set'][:3] = True
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m.multifit(optimizer="lstsq")

    def test_rerun_multifit(self):
        # Check that the parameter map values have set consistently at the end
        # of `multifit(optimizer="lstsq")` so that rerunning it doesn't fall
        # back to slow multifit
        m = self.m
        m.multifit(optimizer='lstsq')
        p = m[0].onset_energy
        np.testing.assert_equal(p.map["is_set"], True)
        np.testing.assert_allclose(p.map['values'], p.map['values'][0])
        np.testing.assert_equal(p.map['std'], np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m.multifit(optimizer="lstsq")
        # assert len(record) == 0

    def test_heteroscedastic_variance(self):
        m = self.m
        m.signal.estimate_poissonian_noise_variance()
        with pytest.warns(UserWarning):
            m.multifit(optimizer="lstsq",
                       match="noise of the signal is not homoscedastic")


class TestLinearModel2D:

    def setup_method(self, method):
        low, high = -10, 10
        N = 100
        x = y = np.linspace(low, high, N)
        mesh = np.meshgrid(x, y)
        self.mesh, self.x, self.y = mesh, x, y

    @pytest.mark.parametrize('nav2d', [False, True])
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
            s = hs.stack([s]*2)
            s = hs.stack([s]*3)

        m = s.create_model()
        m.append(G1)

        G1.set_parameters_not_free()
        G1.A.free = True

        m.multifit(optimizer='lstsq', calculate_errors=True)
        diff = (s - m.as_signal(show_progressbar=False))
        np.testing.assert_allclose(diff.data, 0.0, atol=1E-7)
        np.testing.assert_allclose(m.p_std[0], 0.0, atol=1E-7)

    @pytest.mark.parametrize('nav2d', [False, True])
    def test_model2D_linear_many_gaussians(self, nav2d):
        mesh, x, y = self.mesh, self.x, self.y
        gausslow, gausshigh = -8, 8
        gauss_step = 8
        X, Y = mesh
        z = np.zeros(X.shape)
        g = Gaussian2D()
        for i in np.arange(gausslow, gausshigh+1, gauss_step):
            for j in np.arange(gausslow, gausshigh+1, gauss_step):
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
            s = hs.stack([s]*2)
            s = hs.stack([s]*3)

        m = s.create_model()
        for i in np.arange(gausslow, gausshigh+1, gauss_step):
            for j in np.arange(gausslow, gausshigh+1, gauss_step):
                g = Gaussian2D(centre_x=i, centre_y=j)
                g.set_parameters_not_free()
                g.A.free = True
                m.append(g)

        m.fit(optimizer='lstsq')
        np.testing.assert_allclose(s.data, m.as_signal().data)

    @pytest.mark.parametrize('nav2d', [False, True])
    def test_model2D_polyexpression(self, nav2d):
        poly = "a*x**2 + b*x - c*y**2 + d*y + e"
        P = Expression(poly, 'poly')
        P.a.value = 6
        P.b.value = 5
        P.c.value = 4
        P.d.value = 3
        P.e.value = 2

        data = P.function(*self.mesh)
        s = Signal2D(data)

        if nav2d:
            s = hs.stack([s]*2)
            s = hs.stack([s]*3)

        m = s.create_model()
        m.append(P)
        m.fit(optimizer='lstsq')
        diff = (s - m.as_signal(show_progressbar=False))
        np.testing.assert_allclose(diff.data, 0.0, atol=1E-7)
        np.testing.assert_allclose(m.p_std, 0.0, atol=1E-7)


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
        m.fit(optimizer='lstsq')

        np.testing.assert_allclose(gs[0].A.value, 20)
        np.testing.assert_allclose(gs[1].A.value, -10)
        np.testing.assert_allclose(gs[2].A.value, 5)
        np.testing.assert_allclose(s.data,  m())

    def test_with_twins(self):
        gs = self.gs
        m = self.m
        s = self.s
        for g in gs:
            g.sigma.free = False
            g.centre.free = False

        gs[0].A.value = 1
        m.fit(optimizer='lstsq')

        np.testing.assert_allclose(gs[0].A.value, 20)
        np.testing.assert_allclose(gs[1].A.value, -10)
        np.testing.assert_allclose(gs[2].A.value, 5)
        np.testing.assert_allclose(s.data, m())


def test_compute_constant_term():
    rng = np.random.default_rng(1)
    s = Signal1D(rng.random(10))
    m = s.create_model()
    lin = Expression("a*x + b", name='linear')
    m.append(lin)

    lin.a.value = 2
    lin.b.value = 3
    lin.b.free = False
    np.testing.assert_allclose(lin._compute_constant_term(), 3)


@lazifyTestClass
class TestLinearEdgeCases:
    def setup_method(self, method):
        s = EDS_SEM_Spectrum().isig[5.0:15.0]
        m = s.create_model(auto_background=False)
        self.m = m

    def test_no_free_parameters(self):
        self.m.set_parameters_not_free()
        with pytest.raises(RuntimeError,
                           match="Model does not contain any free components!"):
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
        m = EDS_SEM_Spectrum().create_model()
        m2 = EDS_SEM_Spectrum().isig[5.:15.].create_model()
        self.m = m
        self.m2 = m2

    def test_fixed_chained_twinned_components(self):
        m = self.m
        m.fit(optimizer="lstsq")
        A = m.as_signal()

        m[4].A.free = False
        m.fit(optimizer="lstsq")
        B = m.as_signal()
        np.testing.assert_allclose(A.data, B.data, rtol=5E-5)

    def test_fit_fixed_twinned_components_and_std(self):
        m = self.m2
        m[1].A.free = False
        m.fit(optimizer='lstsq')
        lstsq_fit = m.as_signal()
        nonlinear_parameters = [p for c in m for p in c.parameters
                                if not p._linear]
        linear_std = [para.std for para in nonlinear_parameters if para.std]

        m.fit()
        nonlinear_fit = m.as_signal()
        nonlinear_std = [para.std for para in nonlinear_parameters if para.std]

        np.testing.assert_allclose(nonlinear_fit.data, lstsq_fit.data)
        np.testing.assert_allclose(nonlinear_std, linear_std)


class MultiLinearCustomComponent(Component):

    def __init__(self, a0=1, a1=1):
        Component.__init__(
            self, ('a0', 'a1'), linear_parameter_list=['a0', 'a1']
            )

        self.a0.value = a0
        self.a1.value = a1

    def function(self, x):
        a0 = self.a0.value
        a1 = self.a1.value
        return a0 + x * a1


class TestCustomComponent:

    def setup_method(self):
        self.m = EDS_SEM_Spectrum().create_model(auto_background=False)

    def test_custom_comp_w_two_linear_attributes(self):
        c = MultiLinearCustomComponent()
        self.m.append(c)
        with pytest.raises(AttributeError, match="has more than one free"):
            self.m.fit(optimizer='lstsq')

    def test_custom_comp(self):
        c = MultiLinearCustomComponent()
        c.a0.free = False
        self.m.append(c)
        self.m.fit(optimizer='lstsq')

    def test_compare_custom_comp(self):
        c = MultiLinearCustomComponent()
        c.a0.free = False
        c.a0.value = 0

        self.m.append(c)
        self.m.fit(optimizer='lstsq')
        linear = c.a1.value

        self.m.fit()
        nonlinear = c.a1.value

        np.testing.assert_allclose(linear, nonlinear)


def test_fixed_free_offset():
    s = Signal1D(np.ones(100)*3)
    m = s.create_model()
    a = Offset(1.)
    a.offset.free = False
    b = Offset(0.)
    m.extend((a, b))

    m.fit(optimizer="lstsq")

    np.testing.assert_almost_equal(a.offset.value, 1.)
    np.testing.assert_almost_equal(b.offset.value, 2.)


def test_non_uniform_binned():
    s = hs.datasets.artificial_data.get_luminescence_signal()
    s.axes_manager[-1].is_binned = True
    m = s.create_model()
    with pytest.raises(ValueError):
        m.fit(optimizer="lstsq")


def test_navigation_shape_signal1D():
    rng = np.random.default_rng(1)
    s = hs.signals.Signal1D(np.zeros((2, 3, 200)))
    g = hs.model.components1D.Gaussian()
    g.sigma.value = 10
    g.centre.value = 100
    g.A.value = 1000
    m = s.create_model()
    m.append(g)
    g.A.map['values'] = rng.integers(low=500, high=1500, size=(2, 3))
    g.A.map['is_set'] = True
    s.data = m.as_signal().data
    s.add_gaussian_noise(0.5)
    m.set_parameters_not_free(only_nonlinear=True)

    g.A.map['values'] = 0
    m.multifit(optimizer='lstsq')

    np.testing.assert_allclose(s, m.as_signal(), atol=2)


def test_navigation_shape_signal2D():
    rng = np.random.default_rng(10)
    s = hs.signals.Signal1D(np.zeros((2, 3, 200)))
    g = hs.model.components1D.Gaussian()
    g.sigma.value = 10
    g.centre.value = 100
    g.A.value = 1000
    m = s.create_model()
    m.append(g)
    g.A.map['values'] = rng.integers(low=500, high=1500, size=(2, 3))
    g.A.map['is_set'] = True
    s.data = m.as_signal().data
    s.add_gaussian_noise(0.5)
    m.set_parameters_not_free(only_nonlinear=True)

    g.A.map['values'] = 0
    m.multifit(optimizer='lstsq')

    np.testing.assert_allclose(s, m.as_signal(), atol=2)


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
    m.fit(optimizer='lstsq')

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
    l = hs.model.components1D.Lorentzian()
    l.centre.value = l_ref.centre.value
    m.append(l)
    m.set_parameters_not_free(only_nonlinear=True)
    m.plot()
    m.fit(optimizer='lstsq')

    np.testing.assert_allclose(l_ref.A.value, l.A.value)
    np.testing.assert_allclose(l_ref.centre.value, l.centre.value)
    np.testing.assert_allclose(l_ref.gamma.value, l.gamma.value)
    np.testing.assert_allclose(m.as_signal().data, s.data)


@pytest.mark.parametrize('multiple_free_parameters', (True, False))
@pytest.mark.parametrize('nav_dim', (0, 1, 2))
def test_expression_convolved(nav_dim, multiple_free_parameters):
    s_ref = hs.signals.Signal1D(np.ones(100))

    # Create signal to convolve
    to_convolve_component = hs.model.components1D.Gaussian(A=100, sigma=5, centre=10)
    to_convolve = hs.signals.Signal1D(to_convolve_component.function(np.arange(100)))
    to_convolve.axes_manager[-1].offset = -to_convolve_component.centre.value

    # Create reference signal from model with convolution
    l_ref = hs.model.components1D.Lorentzian(A=100, centre=20, gamma=4)
    m_ref = s_ref.create_model()
    m_ref.append(l_ref)
    m_ref.low_loss = to_convolve
    s = m_ref.as_signal()

    if nav_dim >= 1:
        s = hs.stack([s]*2)
        to_convolve = hs.stack([to_convolve]*2)
    if nav_dim == 2:
        s = hs.stack([s]*3)
        to_convolve = hs.stack([to_convolve]*3)

    m = s.create_model()
    l = hs.model.components1D.Lorentzian(centre=20, gamma=4)
    m.append(l)
    assert not m.convolved
    m.low_loss = to_convolve
    assert m.convolved
    m.set_parameters_not_free(only_nonlinear=True)
    with pytest.warns(UserWarning):
        m.multifit(optimizer='lstsq')

    np.testing.assert_allclose(l_ref.A.value, l.A.value)
    np.testing.assert_allclose(l_ref.centre.value, l.centre.value)
    np.testing.assert_allclose(l_ref.gamma.value, l.gamma.value)
    np.testing.assert_allclose(m.as_signal().data, s.data)
    if nav_dim in (1, 2):
        np.testing.assert_allclose(l.A.map['values'].mean(), l_ref.A.value)
        np.testing.assert_allclose(l.centre.map['values'].mean(), l_ref.centre.value)
        np.testing.assert_allclose(l.gamma.map['values'].mean(), l_ref.gamma.value)


@pytest.mark.parametrize("nav_dim", (0, 1, 2))
@pytest.mark.parametrize("convolve", (True, False))
def test_expression_multiple_linear_parameter(nav_dim, convolve):
    """
    This test checks that linear fitting works with convolution with
     - single and multidimensional fit (warning raise)
     - multiple free parameters for the same component (different code path)
    """
    s_ref = hs.signals.Signal1D(np.ones(20))
    p_ref = hs.model.components1D.Polynomial(order=2, a0=25, a1=-50, a2=2.5,
                                             legacy=False)

    # Create signal to convolve
    to_convolve_component = hs.model.components1D.Gaussian(A=100, sigma=5, centre=10)
    to_convolve = hs.signals.Signal1D(to_convolve_component.function(np.arange(1000)))
    to_convolve.axes_manager[-1].offset = -to_convolve_component.centre.value

    m_ref = s_ref.create_model()
    m_ref.extend([p_ref])
    if convolve:
        m_ref.low_loss = to_convolve
    s = m_ref.as_signal()

    if nav_dim >= 1:
        s = hs.stack([s]*2)
        if convolve:
            to_convolve = hs.stack([to_convolve]*2)
    if nav_dim == 2:
        s = hs.stack([s]*3)
        if convolve:
            to_convolve = hs.stack([to_convolve]*3)

    m = s.create_model()
    p = hs.model.components1D.Polynomial(order=2, legacy=False)
    m.append(p)
    assert not m.convolved
    if convolve:
        m.low_loss = to_convolve
        with pytest.warns(UserWarning):
            m.multifit(optimizer='lstsq')
    else:
        m.multifit(optimizer='lstsq')

    np.testing.assert_allclose(p_ref.a0.value, p.a0.value)
    np.testing.assert_allclose(p_ref.a1.value, p.a1.value)
    np.testing.assert_allclose(p_ref.a2.value, p.a2.value)
    np.testing.assert_allclose(m.as_signal().data, s.data)
    if nav_dim >= 1:
        np.testing.assert_allclose(p.a0.map['values'].mean(), p_ref.a0.value)
        np.testing.assert_allclose(p.a1.map['values'].mean(), p_ref.a1.value)
        np.testing.assert_allclose(p.a2.map['values'].mean(), p_ref.a2.value)


@pytest.mark.parametrize('nav_dim', (0, 1, 2))
def test_multiple_linear_parameters_convolution(nav_dim):
    s_ref = hs.signals.Signal1D(np.ones(1000))

    # Create signal to convolve
    to_convolve_component = hs.model.components1D.Gaussian(A=1000, sigma=50, centre=100)
    to_convolve = hs.signals.Signal1D(to_convolve_component.function(np.arange(1000)))
    to_convolve.axes_manager[-1].offset = -to_convolve_component.centre.value

    l_ref1 = hs.model.components1D.Lorentzian(A=100, centre=200, gamma=10)
    l_ref2 = hs.model.components1D.Lorentzian(A=100, centre=600, gamma=20)

    m_ref = s_ref.create_model()
    m_ref.extend([l_ref1, l_ref2])
    m_ref.low_loss = to_convolve
    s = m_ref.as_signal()

    if nav_dim >= 1:
        s = hs.stack([s]*2)
        to_convolve = hs.stack([to_convolve]*2)
    if nav_dim == 2:
        s = hs.stack([s]*3)
        to_convolve = hs.stack([to_convolve]*3)

    m = s.create_model()
    l1 = hs.model.components1D.Lorentzian(centre=200, gamma=10)
    l2 = hs.model.components1D.Lorentzian(centre=600, gamma=20)
    m.extend([l1, l2])
    assert not m.convolved
    m.low_loss = to_convolve
    assert m.convolved
    m.set_parameters_not_free(only_nonlinear=True)
    with pytest.warns(UserWarning):
        m.multifit(optimizer='lstsq')

    np.testing.assert_allclose(l_ref1.A.value, l1.A.value)
    np.testing.assert_allclose(l_ref1.centre.value, l1.centre.value)
    np.testing.assert_allclose(l_ref1.gamma.value, l1.gamma.value)
    np.testing.assert_allclose(l_ref2.A.value, l2.A.value)
    np.testing.assert_allclose(l_ref2.centre.value, l2.centre.value)
    np.testing.assert_allclose(l_ref2.gamma.value, l2.gamma.value)
    np.testing.assert_allclose(m.as_signal().data, s.data)
    if nav_dim >= 1:
        np.testing.assert_allclose(l1.A.map['values'].mean(), l_ref1.A.value)
        np.testing.assert_allclose(l1.centre.map['values'].mean(), l_ref1.centre.value)
        np.testing.assert_allclose(l1.gamma.map['values'].mean(), l_ref1.gamma.value)
        np.testing.assert_allclose(l2.A.map['values'].mean(), l_ref2.A.value)
        np.testing.assert_allclose(l2.centre.map['values'].mean(), l_ref2.centre.value)
        np.testing.assert_allclose(l2.gamma.map['values'].mean(), l_ref2.gamma.value)
