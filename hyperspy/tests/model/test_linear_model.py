import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D

from hyperspy.component import Component
from hyperspy.components1d import Gaussian, Expression, Offset
from hyperspy.components2d import Gaussian2D

from hyperspy.datasets.example_signals import EDS_SEM_Spectrum
from hyperspy.datasets.artificial_data import get_low_loss_eels_signal
from hyperspy.datasets.artificial_data import get_core_loss_eels_signal
from hyperspy.misc.model_tools import get_top_parent_twin
from hyperspy.decorators import lazifyTestClass
from hyperspy.misc.model_tools import (
    parameter_map_values_all_identical,
    all_set_non_free_para_have_identical_values
    )


def test_fit_binned():
    np.random.seed(1)
    s = Signal1D(
        np.random.normal(
            scale=2,
            size=10000)).get_histogram()
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


@lazifyTestClass
class TestMultiFitLinear:

    def setup_method(self):
        np.random.seed(1)
        x = np.random.random(30)
        shape = np.random.random((2,3,1))
        X = shape*x
        self.s = Signal1D(X)
        self.m = self.s.create_model()

    def test_gaussian(self):
        m = self.m
        L = Gaussian(centre=15.)
        L.centre.free = L.sigma.free = False
        m.append(L)

        m.fit(optimizer='lstsq')
        single = m.as_signal()
        m.assign_current_values_to_all()
        m.multifit(optimizer='lstsq', iterpath='serpentine')
        multi = m.as_signal()

        np.testing.assert_almost_equal(
            single.inav[0,0].data, multi.inav[0,0].data)

    def test_map_values_std_isset(self):
        m = self.m
        L = Gaussian(centre=15.)
        L.centre.free = L.sigma.free = False
        m.append(L)

        m.multifit(iterpath="serpentine")
        nonlinear = L.A.map.copy()

        L.A.map['is_set'] = False
        m.multifit(optimizer='lstsq', calculate_errors=True)
        linear = L.A.map.copy()

        np.testing.assert_almost_equal(nonlinear['values'], linear['values'])
        np.testing.assert_almost_equal(nonlinear['std'], linear['std'])
        np.testing.assert_almost_equal(nonlinear['is_set'], linear['is_set'])

        m.multifit(optimizer='lstsq', calculate_errors=False)
        np.testing.assert_equal(L.A.map['std'], np.nan)

    def test_offset(self):
        m = self.m
        L = Offset(offset=1.)
        m.append(L)

        m.fit(optimizer='lstsq')
        single = m.as_signal()
        m.assign_current_values_to_all()
        m.multifit(optimizer='lstsq', iterpath='serpentine')
        multi = m.as_signal()
        # compare fits from first pixel
        np.testing.assert_almost_equal(
            single.inav[0,0].data, multi.inav[0,0].data)

    def test_channel_switches(self):
        m = self.m
        m.channel_switches[5:-5] = False
        L = Gaussian(centre=15.)
        L.centre.free = L.sigma.free = False
        m.append(L)

        m.fit(optimizer='lstsq')
        single = m.as_signal()
        m.assign_current_values_to_all()
        m.multifit(optimizer='lstsq', iterpath='serpentine')
        multi = m.as_signal()

        np.testing.assert_almost_equal(
            single.inav[0,0].data, multi.inav[0,0].data)

        m.fit()
        single_nonlinear = m.as_signal()
        np.testing.assert_almost_equal(
            single.inav[0,0].data, single_nonlinear.inav[0,0].data)

class TestLinearFitting:

    def setup_method(self, method):
        self.s = EDS_SEM_Spectrum().isig[5.0:15.0]
        self.m = self.s.create_model(auto_background=False)
        self.c = Expression('a*x+b', 'line with offset')
        self.m.append(self.c)

    def test_linear_fitting_with_offset(self):
        m = self.m
        m.fit(optimizer='lstsq')
        np.testing.assert_allclose(
            m.p0,
            np.array([933.2343071493418,
                      47822.98004150301,
                      -5867.611808815612,
                      56805.518919752234])
            )

        # Repeat test with offset fixed
        self.c.b.free = False
        m.fit(optimizer='lstsq')
        np.testing.assert_allclose(
            m.p0,
            np.array([933.2343071496773, 47822.98004150315, -5867.611808815624])
            )

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


@lazifyTestClass
class TestFitAlgorithms:

    def setup_method(self, method):
        s = EDS_SEM_Spectrum().isig[5.0:15.0]
        m = s.create_model(auto_background=False)
        c = Expression('a*x+b', 'line with offset')
        m.append(c)
        m.fit()
        self.nonlinear_fit_res = m.as_signal()
        self.nonlinear_fit_std = [p.std for p in m._free_parameters if p.std]
        self.m = m

    def test_compare_lstsq(self):
        m = self.m
        m.fit(optimizer='lstsq')
        lstsq_fit = m.as_signal()
        np.testing.assert_allclose(self.nonlinear_fit_res, lstsq_fit.data, rtol=2E-6)
        linear_std = [para.std for para in m._free_parameters if para.std]
        np.testing.assert_allclose(self.nonlinear_fit_std, linear_std)

    def test_nonactive_component(self):
        m = self.m
        m[1].active = False
        m.fit(optimizer='lstsq')
        linear_fit = m.as_signal()
        m.fit()
        nonlinear_fit = m.as_signal()
        np.testing.assert_allclose(nonlinear_fit.data, linear_fit.data, rtol=5E-6)

    def test_compare_ridge(self):
        pytest.importorskip("sklearn")
        m = self.m
        if m.signal._lazy:
            with pytest.raises(ValueError):
                m.fit(optimizer='ridge_regression')
            return
        else:
            m.fit(optimizer='ridge_regression')
        ridge_fit = m.as_signal()
        np.testing.assert_allclose(self.nonlinear_fit_res, ridge_fit.data, rtol=2E-6)
        linear_std = [para.std for para in m._free_parameters if para.std]
        np.testing.assert_allclose(self.nonlinear_fit_std, linear_std)


@lazifyTestClass
class TestLinearEELSFitting:

    def setup_method(self, method):
        self.ll = get_low_loss_eels_signal()
        self.cl = get_core_loss_eels_signal()
        self.cl.add_elements(('Mn',))
        self.m = self.cl.create_model(auto_background=False)
        self.m[0].onset_energy.value = 673.
        self.m_convolved = self.cl.create_model(auto_background=False, ll=self.ll)
        self.m_convolved[0].onset_energy.value = 673.

    def test_convolved_and_std_error(self):
        m = self.m_convolved
        m.fit(optimizer='lstsq')
        linear = m.as_signal()
        std_linear = m.p_std
        m.fit(optimizer='lm')
        lm = m.as_signal()
        std_lm = m.p_std
        diff = linear - lm
        np.testing.assert_almost_equal(diff.data.sum(), 0.0, decimal=2)
        np.testing.assert_almost_equal(std_linear, std_lm, decimal=5)

    def test_nonconvolved(self):
        m = self.m
        m.fit(optimizer='lstsq')
        linear = m.as_signal()
        m.fit(optimizer='lm')
        lm = m.as_signal()
        diff = linear - lm
        np.testing.assert_almost_equal(diff.data.sum(), 0.0, decimal=2)


class TestLinearModel2D:

    def setup_method(self, method):
        low, high = -10, 10
        N = 100
        self.x = self.y = np.linspace(low, high, N)
        self.mesh = np.meshgrid(self.x, self.y)

    def test_model2D_one_component(self):
        G1 = Gaussian2D(30, 5.0, 4.0, 0, 0)

        data = G1.function(*self.mesh)
        s = Signal2D(data)
        s.axes_manager[-2].offset = self.x[0]
        s.axes_manager[-1].offset = self.y[0]

        s.axes_manager[-2].scale = self.x[1] - self.x[0]
        s.axes_manager[-1].scale = self.y[1] - self.y[0]

        m = s.create_model()
        m.append(G1)

        G1.set_parameters_not_free()
        G1.A.free = True

        m.fit(optimizer='lstsq')
        diff = (s - m.as_signal(show_progressbar=False))
        np.testing.assert_almost_equal(diff.data.sum(), 0.0)
        np.testing.assert_almost_equal(m.p_std[0], 0.0)

    def test_model2D_linear_many_gaussians(self):
        gausslow, gausshigh = -8, 8
        gauss_step = 8
        X, Y = self.mesh
        z = np.zeros(X.shape)
        g = Gaussian2D()
        for i in np.arange(gausslow, gausshigh+1, gauss_step):
            for j in np.arange(gausslow, gausshigh+1, gauss_step):
                g.centre_x.value = i
                g.centre_y.value = j
                g.A.value = 10
                z += g.function(X, Y)

        s = Signal2D(z)
        s.axes_manager[-2].offset = self.x[0]
        s.axes_manager[-1].offset = self.y[0]

        s.axes_manager[-2].scale = self.x[1] - self.x[0]
        s.axes_manager[-1].scale = self.y[1] - self.y[0]

        m = s.create_model()
        for i in np.arange(gausslow, gausshigh+1, gauss_step):
            for j in np.arange(gausslow, gausshigh+1, gauss_step):
                g = Gaussian2D(centre_x = i, centre_y=j)
                g.set_parameters_not_free()
                g.A.free = True
                m.append(g)

        m.fit(optimizer='lstsq')
        np.testing.assert_array_almost_equal(s.data, m.as_signal().data)

    def test_model2D_polyexpression(self):
        poly = "a*x**2 + b*x - c*y**2 + d*y + e"
        P = Expression(poly, 'poly')
        P.a.value = 6
        P.b.value = 5
        P.c.value = 4
        P.d.value = 3
        P.e.value = 2

        data = P.function(*self.mesh)
        s = Signal2D(data)

        m = s.create_model()
        m.append(P)
        m.fit(optimizer='lstsq')
        diff = (s - m.as_signal(show_progressbar=False))
        np.testing.assert_almost_equal(diff.data.sum(), 0.0, decimal=2)
        np.testing.assert_almost_equal(m.p_std, 0.0, decimal=2)


class TestLinearFitTwins:

    def setup_method(self, method):
        from hyperspy._components.gaussian import Gaussian
        from hyperspy._signals.signal1d import Signal1D
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

    def test_get_parent_twin(self):
        assert get_top_parent_twin(self.gs[2].A) is self.gs[0].A
        assert get_top_parent_twin(self.gs[1].A) is self.gs[0].A
        assert get_top_parent_twin(self.gs[0].A) is self.gs[0].A

    def test_top_parent_twins_are_active(self):
        assert self.gs[0]._top_parent_twins_are_active
        assert self.gs[1]._top_parent_twins_are_active
        assert self.gs[2]._top_parent_twins_are_active

        self.gs[2].active = False
        assert self.gs[0]._top_parent_twins_are_active
        assert self.gs[1]._top_parent_twins_are_active
        assert not self.gs[2]._top_parent_twins_are_active

        self.gs[0].active = False
        assert not self.gs[0]._top_parent_twins_are_active
        assert not self.gs[1]._top_parent_twins_are_active
        assert not self.gs[2]._top_parent_twins_are_active

    def test_without_twins(self):
        for g in self.gs:
            g.sigma.free = False
            g.centre.free = False
            g.A.twin = None

        self.gs[0].A.value = 1
        self.m.fit(optimizer='lstsq')

        np.testing.assert_almost_equal(self.gs[0].A.value, 20)
        np.testing.assert_almost_equal(self.gs[1].A.value, -10)
        np.testing.assert_almost_equal(self.gs[2].A.value, 5)
        np.testing.assert_array_almost_equal((self.s - self.m.as_signal()).data, 0)

    def test_with_twins(self):
        for g in self.gs:
            g.sigma.free = False
            g.centre.free = False

        self.gs[0].A.value = 1
        self.m.fit(optimizer='lstsq')

        np.testing.assert_almost_equal(self.gs[0].A.value, 20)
        np.testing.assert_almost_equal(self.gs[1].A.value, -10)
        np.testing.assert_almost_equal(self.gs[2].A.value, 5)
        np.testing.assert_array_almost_equal((self.s - self.m.as_signal()).data, 0)


class TestCompute:

    def setup_method(self):
        self.s = Signal1D(np.random.random(10))
        m = self.s.create_model()
        self.lin = Expression("a*x + b", name='linear')
        m.append(self.lin)

    def test_compute_component_zero(self):
        np.testing.assert_array_almost_equal(self.lin._compute_component(), 0)

    def test_compute_component(self):
        self.lin.a.value = 2
        self.lin.b.value = 3
        np.testing.assert_array_almost_equal(self.lin._compute_component(), 2*np.arange(self.s.axes_manager[-1].size) + 3)

    def test_compute_constant(self):
        self.lin.a.value = 2
        self.lin.b.value = 3
        self.lin.b.free = False
        np.testing.assert_array_almost_equal(self.lin._compute_constant_term(), 3)


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

    def test_append_linear_comp_not_in_model(self):
        with pytest.raises(AssertionError):
            self.m._compute_component(component=Gaussian())


@lazifyTestClass
class TestLinearMultiFitEdgeCases:

    def setup_method(self, method):
        nav = Signal2D(np.random.random((2,2)))
        s = EDS_SEM_Spectrum().isig[5.0:15.0] * nav.T
        self.m = s.create_model()

    def test_set_value_in_non_free_parameter(self):
        self.m[1].sigma.map['values'][0,0] = 2.
        self.m[1].sigma.map['is_set'][0,0] = True
        with pytest.warns(UserWarning, match="model contains non-free parameters"):
            self.m.multifit(optimizer="lstsq")


class TestLinearModelTools:

    def setup_method(self):
        nav = Signal2D(np.random.random((2,2)))
        s = EDS_SEM_Spectrum() * nav.T
        self.m = s.create_model()

    def test_parameter_map_values_all_identical(self):
        para = self.m[0].a1
        assert parameter_map_values_all_identical(para)
        para.map['values'][0,0] = 2
        assert not parameter_map_values_all_identical(para)


    def test_all_set_non_free_para_have_identical_values(self):
        assert all_set_non_free_para_have_identical_values(self.m)

        para1 = self.m[0].a1

        # Value varies, but is_set is False
        para1.map['values'][0,0] = 2
        is_identical, para_list =  all_set_non_free_para_have_identical_values(self.m)
        assert is_identical is True
        assert not para_list

        # Same, but para is not free
        para1.free = False
        is_identical, para_list =  all_set_non_free_para_have_identical_values(self.m)
        assert is_identical is True
        assert not para_list

        # para is_set is now True, no longer identical
        para1.map['is_set'][0,0] = True
        is_identical, para_list = all_set_non_free_para_have_identical_values(self.m)
        assert is_identical is False
        assert para1 in para_list and len(para_list) == 1

        # all is_set are True, but the values vary, no not identical either
        para1.map['is_set'] = True
        is_identical, para_list = all_set_non_free_para_have_identical_values(self.m)
        assert is_identical is False
        assert para1 in para_list and len(para_list) == 1


class TestTwinnedComponents:

    def setup_method(self):
        self.m = EDS_SEM_Spectrum().create_model()
        self.m2 = EDS_SEM_Spectrum().isig[5.:15.].create_model()

    def test_fixed_chained_twinned_components(self):
        m = self.m
        m.fit(optimizer="lstsq")
        A = m.as_signal()

        m[4].A.free = False
        m.fit(optimizer="lstsq")
        B = m.as_signal()
        np.testing.assert_array_almost_equal(A.data, B.data)

    def test_fit_fixed_twinned_components_and_std(self):
        m = self.m2
        m[1].A.free = False
        m.fit(optimizer='lstsq')
        lstsq_fit = m.as_signal()
        linear_std = [para.std for para in self.m.nonlinear_parameters if para.std]

        m.fit()
        nonlinear_fit = m.as_signal()
        nonlinear_std = [para.std for para in self.m.nonlinear_parameters if para.std]

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
            with pytest.warns(UserWarning, match="not based on Expression"):
                self.m.fit(optimizer='lstsq')

    def test_custom_comp_warning(self):
        c = MultiLinearCustomComponent()
        c.a0.free = False
        self.m.append(c)
        with pytest.warns(UserWarning, match="contains custom components"):
            self.m.fit(optimizer='lstsq')

    def test_compare_custom_comp(self):
        c = MultiLinearCustomComponent()
        c.a0.free = False
        c.a0.value = 0

        self.m.append(c)
        with pytest.warns(UserWarning, match="contains custom components"):
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
