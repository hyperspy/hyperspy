from unittest import mock

import numpy as np
import pytest

from hyperspy.misc.eels.eelsdb import eelsdb
from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D
from hyperspy._components.gaussian import Gaussian
from hyperspy._components.gaussian2d import Gaussian2D
from hyperspy._components.power_law import PowerLaw
from hyperspy._components.expression import Expression
from hyperspy._components.offset import Offset

from hyperspy.datasets.example_signals import EDS_SEM_Spectrum
from hyperspy.datasets.artificial_data import get_low_loss_eels_signal
from hyperspy.datasets.artificial_data import get_core_loss_eels_signal
from hyperspy.misc.utils import slugify
from hyperspy.decorators import lazifyTestClass


@lazifyTestClass
class TestModelFitBinned:

    def setup_method(self, method):
        np.random.seed(1)
        s = Signal1D(
            np.random.normal(
                scale=2,
                size=10000)).get_histogram()
        s.metadata.Signal.binned = True
        g = Gaussian()
        m = s.create_model()
        m.append(g)
        g.sigma.value = 1
        g.centre.value = 0.5
        g.A.value = 1e3
        self.m = m

    def test_model_is_not_linear(self):
        """
        Model is not currently linear as Gaussian sigma and centre parameters
        are free
        """
        assert not self.m._check_all_active_components_are_linear()

    def test_fit_lsq_linear(self):
        self.m[0].sigma.free = False
        self.m[0].centre.free = False
        self.m.fit(fitter="linear")
        np.testing.assert_allclose(self.m[0].A.value, 6132.640632924692, 1)
        np.testing.assert_allclose(self.m[0].centre.value, 0.5)
        np.testing.assert_allclose(self.m[0].sigma.value, 1)


@lazifyTestClass
class TestMultifit:

    def setup_method(self, method):
        s = Signal1D(np.zeros((2, 200)))
        s.axes_manager[-1].offset = 1
        s.data[:] = 2 * s.axes_manager[-1].axis ** (-3)

        m = s.create_model()
        m.append(PowerLaw())
        m[0].A.value = 2
        m[0].r.value = 2
        m.store_current_values()
        m.axes_manager.indices = (1,)
        m[0].r.value = 100
        m[0].A.value = 2
        m.store_current_values()
        m[0].A.free = False
        self.m = m
        m.axes_manager.indices = (0,)
        m[0].A.value = 100

    def test_bounded_lsq_linear(self):
        m = self.m
        m[0].A.free = True
        m[0].r.free = False

        m.signal.data *= 2.
        m[0].A.value = 2.
        m[0].r.value = 3.
        m[0].r.assign_current_value_to_all()
        m.multifit(fitter='linear', bounded=True, show_progressbar=None)
        np.testing.assert_array_almost_equal(self.m[0].A.map['values'],
                                             [4., 4.])

class TestLinearFitting:
    def setup_method(self, method):
        self.s = EDS_SEM_Spectrum().isig[5.0:15.0]
        self.m = self.s.create_model(auto_background=False)
        self.c = Expression('a*x+b', 'line with offset')
        self.m.append(self.c)

    def test_linear_fitting_with_offset(self):
        m = self.m
        m.fit('linear')
        linear = m.as_signal()
        np.testing.assert_allclose(m.p0, np.array(
            [933.234307, 47822.980041, -5867.611809, 56805.51892]))

        m.fit('leastsq')
        leastsq = m.as_signal()
        diff = (leastsq - linear)
        np.testing.assert_almost_equal(diff.data.sum(), 0.0, decimal=4)

    def test_fixed_offset_value(self):
        c = self.c
        m = self.m
        constant = c._compute_constant_term()
        assert (constant - c.b.value*np.ones(m.axis.axis.shape)).sum() == 0

    def test_3rd_order_polynomial(self):
        m = self.m
        m.remove(self.c)
        c2 = Expression('a*x**3+b*x**2+c*x+d', '3rd Poly')
        c2.a.value = 0
        c2.b.value = 0
        c2.a.free = False
        c2.b.free = False
        m.append(c2)
        m.fit('linear')
        diff = (self.s - m.as_signal())
        np.testing.assert_almost_equal(diff.data.sum(), 0)


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
        m.fit('linear')
        linear = m.as_signal()
        std_linear = m.p_std
        m.fit('leastsq')
        leastsq = m.as_signal()
        std_leastsq = m.p_std
        diff = linear - leastsq
        np.testing.assert_almost_equal(diff.data.sum(), 0.0, decimal=2)
        np.testing.assert_almost_equal(std_linear, std_leastsq, decimal=5)

    def test_nonconvolved(self):
        m = self.m
        m.fit('linear')
        linear = m.as_signal()
        m.fit('leastsq')
        leastsq = m.as_signal()
        diff = linear - leastsq
        np.testing.assert_almost_equal(diff.data.sum(), 0.0, decimal=2)

    # The following twos tests need a model based on cl spectrum with more 
    # components. Does not work with hydrogenic GOS as there Mn_L3 == Mn_L2

    # def test_chained_twins(self):
    #     m = self.m
    #     m[2].parameters[0].twin = m[1].parameters[0]
    #     m[1].parameters[0].twin = m[0].parameters[0]
    #     m.fit('linear')
    #     linear = m.as_signal()
    #     m.fit('leastsq')
    #     leastsq = m.as_signal()
    #     diff = linear - leastsq
    #     np.testing.assert_almost_equal(diff.data.sum(), 0.0, decimal=2)

    # def test_fit_fix_fit(self):
    #     'Fit with twinned components after the top parent twin becomes fixed'
    #     m = self.m_convolved
    #     m.append(Offset()) # Need random free component in the mix as well
    #     m.fit('linear')
    #     data1 = m.as_signal().data
    #     m[2].parameters[0].twin = m[1].parameters[0]
    #     m[1].parameters[0].twin = m[0].parameters[0]
    #     m[0].set_parameters_not_free()
    #     m.fit('linear')
    #     data2 = m.as_signal().data
    #     np.testing.assert_almost_equal(data1, data2)

class TestLinearModel2D:
    def setup_method(self, method):
        self.X, self.Y = np.arange(0,100), np.arange(0,50)
        self.mesh = np.meshgrid(self.X,self.Y)

    def test_model2D_one_component(self):
        G1 = Gaussian2D(30, 5.0, 4.0, 50, 25)

        data = G1.function(*self.mesh)
        s = Signal2D(data)
        m = s.create_model()
        m.append(G1)
        G1.sigma_x.value = 5
        G1.sigma_y.value = 4
        G1.centre_x.value = 50
        G1.centre_y.value = 25

        for para in G1.free_parameters[1:]:
            para.free = False     

        m.fit('linear')
        diff = (s - m.as_signal(show_progressbar=False))
        np.testing.assert_almost_equal(diff.data.sum(), 0.0, decimal=2)
        assert m.p_std[0] == 0.

    def test_model2D_two_components(self):
        G1 = Gaussian2D(30, 5.0, 4.0, 50, 25)
        G2 = Gaussian2D(10, 5.0, 4.0, 60, 30)

        data = G1.function(*self.mesh) + G2.function(*self.mesh)
        s = Signal2D(data)
        m = s.create_model()
        m.append(G1)
        m.append(G2)

        G1.sigma_x.value = 5
        G1.sigma_y.value = 4
        G1.centre_x.value = 50
        G1.centre_y.value = 25

        for para in G1.free_parameters[1:]:
            para.free = False     
        
        G2.sigma_x.value = 5
        G2.sigma_y.value = 4
        G2.centre_x.value = 60
        G2.centre_y.value = 30

        for para in G2.free_parameters[1:]:
            para.free = False    

        m.fit('linear')
        diff = (s - m.as_signal(show_progressbar=False))
        np.testing.assert_almost_equal(diff.data.sum(), 0.0, decimal=2)
        np.testing.assert_almost_equal(m.p_std, [0.0, 0.0], decimal=2)

    def test_model2D_polyexpression(self):
        poly = "a*x**2 + b*x - c*y**2 + d*y + e"
        P = Expression(poly, 'poly')
        P.a.value = 6
        P.b.value = 5
        P.c.value = 4
        P.d.value = 3
        P.e.value = 2
        
        data = P.function(*self.mesh)# + G2.function(*mesh)
        s = Signal2D(data)

        m = s.create_model()
        m.append(P)
        m.fit('linear')
        diff = (s - m.as_signal(show_progressbar=False))
        np.testing.assert_almost_equal(diff.data.sum(), 0.0, decimal=2)
        np.testing.assert_almost_equal(m.p_std, 0.0, decimal=2)
