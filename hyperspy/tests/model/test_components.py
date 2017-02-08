import numpy as np

from numpy.testing import assert_allclose

import hyperspy.api as hs
from hyperspy.models.model1d import Model1D
from hyperspy.misc.test_utils import ignore_warning


class TestPowerLaw:

    def setup_method(self, method):
        s = hs.signals.Signal1D(np.zeros(1024))
        s.axes_manager[0].offset = 100
        s.axes_manager[0].scale = 0.01
        m = s.create_model()
        m.append(hs.model.components1D.PowerLaw())
        m[0].A.value = 10
        m[0].r.value = 4
        self.m = m

    def test_estimate_parameters_binned_only_current(self):
        self.m.signal.metadata.Signal.binned = True
        s = self.m.as_signal(show_progressbar=None, parallel=False)
        s.metadata.Signal.binned = True
        g = hs.model.components1D.PowerLaw()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        assert_allclose(g.A.value, 10.084913947965161)
        assert_allclose(g.r.value, 4.0017676988807409)

    def test_estimate_parameters_unbinned_only_current(self):
        self.m.signal.metadata.Signal.binned = False
        s = self.m.as_signal(show_progressbar=None, parallel=False)
        s.metadata.Signal.binned = False
        g = hs.model.components1D.PowerLaw()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        assert_allclose(g.A.value, 10.064378823244837)
        assert_allclose(g.r.value, 4.0017522876514304)

    def test_estimate_parameters_binned(self):
        self.m.signal.metadata.Signal.binned = True
        s = self.m.as_signal(show_progressbar=None, parallel=False)
        s.metadata.Signal.binned = True
        g = hs.model.components1D.PowerLaw()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=False)
        assert_allclose(g.A.value, 10.084913947965161)
        assert_allclose(g.r.value, 4.0017676988807409)

    def test_estimate_parameters_unbinned(self):
        self.m.signal.metadata.Signal.binned = False
        s = self.m.as_signal(show_progressbar=None, parallel=False)
        s.metadata.Signal.binned = False
        g = hs.model.components1D.PowerLaw()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=False)
        assert_allclose(g.A.value, 10.064378823244837)
        assert_allclose(g.r.value, 4.0017522876514304)
        # Test that it all works when calling it with a different signal
        s2 = hs.stack((s, s))
        g.estimate_parameters(s2,
                              None,
                              None,
                              only_current=False)
        assert_allclose(g.A.map["values"][1], 10.064378823244837)
        assert_allclose(g.r.map["values"][0], 4.0017522876514304)


class TestOffset:

    def setup_method(self, method):
        s = hs.signals.Signal1D(np.zeros(10))
        s.axes_manager[0].scale = 0.01
        m = s.create_model()
        m.append(hs.model.components1D.Offset())
        m[0].offset.value = 10
        self.m = m

    def test_estimate_parameters_binned(self):
        self.m.signal.metadata.Signal.binned = True
        s = self.m.as_signal(show_progressbar=None, parallel=False)
        s.metadata.Signal.binned = True
        g = hs.model.components1D.Offset()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        assert_allclose(g.offset.value, 10)

    def test_estimate_parameters_unbinned(self):
        self.m.signal.metadata.Signal.binned = False
        s = self.m.as_signal(show_progressbar=None, parallel=False)
        s.metadata.Signal.binned = False
        g = hs.model.components1D.Offset()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        assert_allclose(g.offset.value, 10)


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
        # if same component is pased, axes_managers get mixed up, tests
        # sometimes randomly fail
        for _m in [self.m, self.m_2d, self.m_3d]:
            _m[0].coefficients.value = coeff_values

    def test_gradient(self):
        c = self.m[0]
        np.testing.assert_array_almost_equal(c.grad_coefficients(1),
                                             np.array([[6, ], [4.5], [3.5]]))
        assert c.grad_coefficients(np.arange(10)).shape == (3, 10)

    def test_estimate_parameters_binned(self):
        self.m.signal.metadata.Signal.binned = True
        s = self.m.as_signal(show_progressbar=None, parallel=False)
        s.metadata.Signal.binned = True
        g = hs.model.components1D.Polynomial(order=2)
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        assert_allclose(g.coefficients.value[0], 0.5)
        assert_allclose(g.coefficients.value[1], 2)
        assert_allclose(g.coefficients.value[2], 3)

    def test_estimate_parameters_unbinned(self):
        self.m.signal.metadata.Signal.binned = False
        s = self.m.as_signal(show_progressbar=None, parallel=False)
        s.metadata.Signal.binned = False
        g = hs.model.components1D.Polynomial(order=2)
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        assert_allclose(g.coefficients.value[0], 0.5)
        assert_allclose(g.coefficients.value[1], 2)
        assert_allclose(g.coefficients.value[2], 3)

    def test_2d_signal(self):
        # This code should run smoothly, any exceptions should trigger failure
        s = self.m_2d.as_signal(show_progressbar=None, parallel=False)
        model = Model1D(s)
        p = hs.model.components1D.Polynomial(order=2)
        model.append(p)
        p.estimate_parameters(s, 0, 100, only_current=False)
        np.testing.assert_allclose(p.coefficients.map['values'],
                                   np.tile([0.5, 2, 3], (10, 1)))

    def test_3d_signal(self):
        # This code should run smoothly, any exceptions should trigger failure
        s = self.m_3d.as_signal(show_progressbar=None, parallel=False)
        model = Model1D(s)
        p = hs.model.components1D.Polynomial(order=2)
        model.append(p)
        p.estimate_parameters(s, 0, 100, only_current=False)
        np.testing.assert_allclose(p.coefficients.map['values'],
                                   np.tile([0.5, 2, 3], (2, 5, 1)))


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

    def test_estimate_parameters_binned(self):
        self.m.signal.metadata.Signal.binned = True
        s = self.m.as_signal(show_progressbar=None, parallel=False)
        s.metadata.Signal.binned = True
        g = hs.model.components1D.Gaussian()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        assert_allclose(g.sigma.value, 0.5)
        assert_allclose(g.A.value, 2)
        assert_allclose(g.centre.value, 1)

    def test_estimate_parameters_unbinned(self):
        self.m.signal.metadata.Signal.binned = False
        s = self.m.as_signal(show_progressbar=None, parallel=False)
        s.metadata.Signal.binned = False
        g = hs.model.components1D.Gaussian()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        assert_allclose(g.sigma.value, 0.5)
        assert_allclose(g.A.value, 2)
        assert_allclose(g.centre.value, 1)


class TestExpression:

    def setup_method(self, method):
        self.g = hs.model.components1D.Expression(
            expression="height * exp(-(x - x0) ** 2 * 4 * log(2)/ fwhm ** 2)",
            name="Gaussian",
            position="x0",
            height=1,
            fwhm=1,
            x0=0,
            module="numpy")

    def test_name(self):
        assert self.g.name == "Gaussian"

    def test_position(self):
        assert self.g._position is self.g.x0

    def test_f(self):
        assert self.g.function(0) == 1

    def test_grad_height(self):
        assert_allclose(
            self.g.grad_height(2),
            1.5258789062500007e-05)

    def test_grad_x0(self):
        assert_allclose(
            self.g.grad_x0(2),
            0.00016922538587889289)

    def test_grad_fwhm(self):
        assert_allclose(
            self.g.grad_fwhm(2),
            0.00033845077175778578)


def test_expression_substitution():
    expr = 'A / B; A = x+2; B= x-c'
    comp = hs.model.components1D.Expression(expr, name='testcomp',
                                            autodoc=True,
                                            c=2)
    assert ''.join(p.name for p in comp.parameters) == 'c'
    assert comp.function(1) == -3

class TestScalableFixedPattern:

    def setup_method(self, method):
        s = hs.signals.Signal1D(np.linspace(0., 100., 10))
        s1 = hs.signals.Signal1D(np.linspace(0., 1., 10))
        s.axes_manager[0].scale = 0.1
        s1.axes_manager[0].scale = 0.1
        self.s = s
        self.pattern = s1

    def test_both_unbinned(self):
        s = self.s
        s1 = self.pattern
        s.metadata.Signal.binned = False
        s1.metadata.Signal.binned = False
        m = s.create_model()
        fp = hs.model.components1D.ScalableFixedPattern(s1)
        m.append(fp)
        with ignore_warning(message="invalid value encountered in sqrt",
                            category=RuntimeWarning):
            m.fit()
        assert abs(fp.yscale.value - 100) <= 0.1

    def test_both_binned(self):
        s = self.s
        s1 = self.pattern
        s.metadata.Signal.binned = True
        s1.metadata.Signal.binned = True
        m = s.create_model()
        fp = hs.model.components1D.ScalableFixedPattern(s1)
        m.append(fp)
        with ignore_warning(message="invalid value encountered in sqrt",
                            category=RuntimeWarning):
            m.fit()
        assert abs(fp.yscale.value - 100) <= 0.1

    def test_pattern_unbinned_signal_binned(self):
        s = self.s
        s1 = self.pattern
        s.metadata.Signal.binned = True
        s1.metadata.Signal.binned = False
        m = s.create_model()
        fp = hs.model.components1D.ScalableFixedPattern(s1)
        m.append(fp)
        with ignore_warning(message="invalid value encountered in sqrt",
                            category=RuntimeWarning):
            m.fit()
        assert abs(fp.yscale.value - 1000) <= 1

    def test_pattern_binned_signal_unbinned(self):
        s = self.s
        s1 = self.pattern
        s.metadata.Signal.binned = False
        s1.metadata.Signal.binned = True
        m = s.create_model()
        fp = hs.model.components1D.ScalableFixedPattern(s1)
        m.append(fp)
        with ignore_warning(message="invalid value encountered in sqrt",
                            category=RuntimeWarning):
            m.fit()
        assert abs(fp.yscale.value - 10) <= .1


class TestHeavisideStep:

    def setup_method(self, method):
        self.c = hs.model.components1D.HeavisideStep()

    def test_integer_values(self):
        c = self.c
        np.testing.assert_array_almost_equal(c.function([-1, 0, 2]),
                                             [0, 0.5, 1])

    def test_float_values(self):
        c = self.c
        np.testing.assert_array_almost_equal(c.function([-0.5, 0.5, 2]),
                                             [0, 1, 1])

    def test_not_sorted(self):
        c = self.c
        np.testing.assert_array_almost_equal(c.function([3, -0.1, 0]),
                                             [1, 0, 0.5])

    def test_gradients(self):
        c = self.c
        np.testing.assert_array_almost_equal(c.A.grad([3, -0.1, 0]),
                                             [1, 1, 1])
        np.testing.assert_array_almost_equal(c.n.grad([3, -0.1, 0]),
                                             [1, 0, 0.5])
