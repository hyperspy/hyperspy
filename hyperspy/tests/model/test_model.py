import numpy as np
import nose.tools

from hyperspy.hspy import *


class TestModelFitBinned:
    def setUp(self):
        np.random.seed(1)
        s = signals.Spectrum(np.random.normal(scale=2, size=10000)).get_histogram()
        s.metadata.Signal.binned = True
        g = components.Gaussian()
        m = create_model(s)
        m.append(g)
        g.sigma.value = 1
        g.centre.value = 0.5
        g.A.value = 1e3
        self.m = m

    def test_fit_fmin_leastsq(self):
        self.m.fit(fitter="fmin", method="ls")
        nose.tools.assert_almost_equal(self.m[0].A.value, 9976.14519369)
        nose.tools.assert_almost_equal(self.m[0].centre.value, -0.110610743285)
        nose.tools.assert_almost_equal(self.m[0].sigma.value, 1.98380705455)

    def test_fit_fmin_ml(self):
        self.m.fit(fitter="fmin", method="ml")
        nose.tools.assert_almost_equal(self.m[0].A.value, 10001.3962086)
        nose.tools.assert_almost_equal(self.m[0].centre.value, -0.104151139427)
        nose.tools.assert_almost_equal(self.m[0].sigma.value, 2.00053642434)

    def test_fit_leastsq(self):
        self.m.fit(fitter="leastsq")
        nose.tools.assert_almost_equal(self.m[0].A.value, 9976.14526082)
        nose.tools.assert_almost_equal(self.m[0].centre.value, -0.110610727064)
        nose.tools.assert_almost_equal(self.m[0].sigma.value, 1.98380707571)

    def test_fit_mpfit(self):
        self.m.fit(fitter="mpfit")
        nose.tools.assert_almost_equal(self.m[0].A.value, 9976.14526286)
        nose.tools.assert_almost_equal(self.m[0].centre.value, -0.110610718444)
        nose.tools.assert_almost_equal(self.m[0].sigma.value, 1.98380707614)

    def test_fit_odr(self):
        self.m.fit(fitter="odr")
        nose.tools.assert_almost_equal(self.m[0].A.value, 9976.14531979)
        nose.tools.assert_almost_equal(self.m[0].centre.value, -0.110610724054)
        nose.tools.assert_almost_equal(self.m[0].sigma.value, 1.98380709939)

    def test_fit_leastsq_grad(self):
        self.m.fit(fitter="leastsq", grad=True)
        nose.tools.assert_almost_equal(self.m[0].A.value, 9976.14526084)
        nose.tools.assert_almost_equal(self.m[0].centre.value, -0.11061073306)
        nose.tools.assert_almost_equal(self.m[0].sigma.value, 1.98380707552)

    def test_fit_mpfit_grad(self):
        self.m.fit(fitter="mpfit", grad=True)
        nose.tools.assert_almost_equal(self.m[0].A.value, 9976.14526084)
        nose.tools.assert_almost_equal(self.m[0].centre.value, -0.11061073306)
        nose.tools.assert_almost_equal(self.m[0].sigma.value, 1.98380707552)

    def test_fit_odr_grad(self):
        self.m.fit(fitter="odr", grad=True)
        nose.tools.assert_almost_equal(self.m[0].A.value, 9976.14531979)
        nose.tools.assert_almost_equal(self.m[0].centre.value, -0.110610724054)
        nose.tools.assert_almost_equal(self.m[0].sigma.value, 1.98380709939)

    def test_fit_bounded(self):
        self.m[0].centre.bmin = 0.5
        self.m[0].bounded = True
        self.m.fit(fitter="mpfit", bounded=True)
        nose.tools.assert_almost_equal(self.m[0].A.value, 9991.65422046)
        nose.tools.assert_almost_equal(self.m[0].centre.value, 0.5)
        nose.tools.assert_almost_equal(self.m[0].sigma.value, 2.08398236966)

class TestModelWeighted:
    def setUp(self):
        np.random.seed(1)
        s = signals.SpectrumSimulation(np.arange(10, 100, 0.1))
        s.axes_manager[0].scale = 0.1
        s.axes_manager[0].offset = 10
        s.add_poissonian_noise()
        m = create_model(s)
        m.append(components.Polynomial(1))
        self.m = m

    def test_fit_leastsq_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        self.m.fit(method="leastsq", weights=np.arange(10, 100, 0.01))
        for result, expected in zip(self.m[0].coefficients.value,
                                    (9.9171648232330245, 1.6013075526198008)):
            nose.tools.assert_almost_equal(result, expected, places=5)

    def test_fit_odr_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        self.m.fit(method="odr", weights=np.arange(10, 100, 0.01))
        for result, expected in zip(self.m[0].coefficients.value,
                                    (9.9171647904033939, 1.6013075460711306)):
            nose.tools.assert_almost_equal(result, expected, places=5)

    def test_fit_mpfit_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        self.m.fit(method="mpfit", weights=np.arange(10, 100, 0.01))
        for result, expected in zip(self.m[0].coefficients.value,
                                    (9.9171648167658937, 1.6013069135078979)):
            nose.tools.assert_almost_equal(result, expected, places=5)

    def test_fit_fmin_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        self.m.fit(fitter="fmin", method="ls", weights=np.arange(10, 100, 0.01))
        for result, expected in zip(self.m[0].coefficients.value,
                                    (9.9171652269521182, 1.6012882249456402)):
            nose.tools.assert_almost_equal(result, expected, places=5)

    def test_fit_leastsq_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        self.m.fit(method="leastsq", weights=np.arange(10, 100, 0.01))
        for result, expected in zip(self.m[0].coefficients.value,
                                    (0.991716479620117, 0.16013091522594181)):
            nose.tools.assert_almost_equal(result, expected, places=5)

    def test_fit_odr_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        self.m.fit(method="odr", weights=np.arange(10, 100, 0.01))
        for result, expected in zip(self.m[0].coefficients.value,
                                    (0.99171647904033933, 0.16013075460711151)):
            nose.tools.assert_almost_equal(result, expected, places=5)

    def test_fit_mpfit_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        self.m.fit(method="mpfit", weights=np.arange(10, 100, 0.01))
        for result, expected in zip(self.m[0].coefficients.value,
                                    (0.99171648027761228, 0.16013071797342349)):
            nose.tools.assert_almost_equal(result, expected, places=5)

    def test_fit_fmin_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        self.m.fit(fitter="fmin", method="ls", weights=np.arange(10, 100, 0.01))
        for result, expected in zip(self.m[0].coefficients.value,
                                    (0.99171625667646135, 0.16012922226345222)):
            nose.tools.assert_almost_equal(result, expected, places=5)
