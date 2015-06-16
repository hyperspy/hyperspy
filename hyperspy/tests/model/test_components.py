import numpy as np
import nose.tools

import hyperspy.hspy as hs


class TestPowerLaw:

    def setUp(self):
        s = hs.signals.Spectrum(np.empty((1024)))
        s.axes_manager[0].offset = 100
        s.axes_manager[0].scale = 0.01
        m = hs.create_model(s)
        m.append(hs.components.PowerLaw())
        m[0].A.value = 10
        m[0].r.value = 4
        self.m = m

    def test_estimate_parameters_binned_only_current(self):
        self.m.spectrum.metadata.Signal.binned = True
        s = self.m.as_signal()
        s.metadata.Signal.binned = True
        g = hs.components.PowerLaw()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nose.tools.assert_almost_equal(g.A.value, 10.084913947965161)
        nose.tools.assert_almost_equal(g.r.value, 4.0017676988807409)

    def test_estimate_parameters_unbinned_only_current(self):
        self.m.spectrum.metadata.Signal.binned = False
        s = self.m.as_signal()
        s.metadata.Signal.binned = False
        g = hs.components.PowerLaw()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nose.tools.assert_almost_equal(g.A.value, 10.064378823244837)
        nose.tools.assert_almost_equal(g.r.value, 4.0017522876514304)

    def test_estimate_parameters_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        s = self.m.as_signal()
        s.metadata.Signal.binned = True
        g = hs.components.PowerLaw()
        g._axes_manager = self.m.spectrum.axes_manager
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=False)
        nose.tools.assert_almost_equal(g.A.value, 10.084913947965161)
        nose.tools.assert_almost_equal(g.r.value, 4.0017676988807409)

    def test_estimate_parameters_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        s = self.m.as_signal()
        s.metadata.Signal.binned = False
        g = hs.components.PowerLaw()
        g._axes_manager = self.m.spectrum.axes_manager
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=False)
        nose.tools.assert_almost_equal(g.A.value, 10.064378823244837)
        nose.tools.assert_almost_equal(g.r.value, 4.0017522876514304)


class TestOffset:

    def setUp(self):
        s = hs.signals.Spectrum(np.empty((10)))
        s.axes_manager[0].scale = 0.01
        m = hs.create_model(s)
        m.append(hs.components.Offset())
        m[0].offset.value = 10
        self.m = m

    def test_estimate_parameters_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        s = self.m.as_signal()
        s.metadata.Signal.binned = True
        g = hs.components.Offset()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nose.tools.assert_almost_equal(g.offset.value, 10)

    def test_estimate_parameters_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        s = self.m.as_signal()
        s.metadata.Signal.binned = False
        g = hs.components.Offset()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nose.tools.assert_almost_equal(g.offset.value, 10)


class TestiPolynomial:

    def setUp(self):
        s = hs.signals.Spectrum(np.empty((1024)))
        s.axes_manager[0].offset = -5
        s.axes_manager[0].scale = 0.01
        m = hs.create_model(s)
        m.append(hs.components.Polynomial(order=2))
        m[0].coefficients.value = (0.5, 2, 3)
        self.m = m

    def test_estimate_parameters_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        s = self.m.as_signal()
        s.metadata.Signal.binned = True
        g = hs.components.Polynomial(order=2)
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nose.tools.assert_almost_equal(g.coefficients.value[0], 0.5)
        nose.tools.assert_almost_equal(g.coefficients.value[1], 2)
        nose.tools.assert_almost_equal(g.coefficients.value[2], 3)

    def test_estimate_parameters_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        s = self.m.as_signal()
        s.metadata.Signal.binned = False
        g = hs.components.Polynomial(order=2)
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nose.tools.assert_almost_equal(g.coefficients.value[0], 0.5)
        nose.tools.assert_almost_equal(g.coefficients.value[1], 2)
        nose.tools.assert_almost_equal(g.coefficients.value[2], 3)


class TestGaussian:

    def setUp(self):
        s = hs.signals.Spectrum(np.empty((1024)))
        s.axes_manager[0].offset = -5
        s.axes_manager[0].scale = 0.01
        m = hs.create_model(s)
        m.append(hs.components.Gaussian())
        m[0].sigma.value = 0.5
        m[0].centre.value = 1
        m[0].A.value = 2
        self.m = m

    def test_estimate_parameters_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        s = self.m.as_signal()
        s.metadata.Signal.binned = True
        g = hs.components.Gaussian()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nose.tools.assert_almost_equal(g.sigma.value, 0.5)
        nose.tools.assert_almost_equal(g.A.value, 2)
        nose.tools.assert_almost_equal(g.centre.value, 1)

    def test_estimate_parameters_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        s = self.m.as_signal()
        s.metadata.Signal.binned = False
        g = hs.components.Gaussian()
        g.estimate_parameters(s,
                              None,
                              None,
                              only_current=True)
        nose.tools.assert_almost_equal(g.sigma.value, 0.5)
        nose.tools.assert_almost_equal(g.A.value, 2)
        nose.tools.assert_almost_equal(g.centre.value, 1)


class TestExpression:

    def setUp(self):
        self.g = hs.components.Expression(
            expression="height * exp(-(x - x0) ** 2 * 4 * log(2)/ fwhm ** 2)",
            name="Gaussian",
            position="x0",
            height=1,
            fwhm=1,
            x0=0,
            module="numpy")

    def test_name(self):
        nose.tools.assert_equal(self.g.name, "Gaussian")

    def test_position(self):
        nose.tools.assert_is(self.g._position, self.g.x0)

    def test_f(self):
        nose.tools.assert_equal(self.g.function(0), 1)

    def test_grad_height(self):
        nose.tools.assert_almost_equal(
            self.g.grad_height(2),
            1.5258789062500007e-05)

    def test_grad_x0(self):
        nose.tools.assert_almost_equal(
            self.g.grad_x0(2),
            0.00016922538587889289)

    def test_grad_fwhm(self):
        nose.tools.assert_almost_equal(
            self.g.grad_fwhm(2),
            0.00033845077175778578)
