import numpy as np
import nose.tools

from hyperspy.hspy import *


class TestModelFitBinned:

    def setUp(self):
        np.random.seed(1)
        s = signals.Spectrum(
            np.random.normal(
                scale=2,
                size=10000)).get_histogram()
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
        nose.tools.assert_almost_equal(self.m[0].A.value, 9976.14526082, 1)
        nose.tools.assert_almost_equal(self.m[0].centre.value, -0.110610727064)
        nose.tools.assert_almost_equal(self.m[0].sigma.value, 1.98380707571, 5)

    def test_fit_mpfit(self):
        self.m.fit(fitter="mpfit")
        nose.tools.assert_almost_equal(self.m[0].A.value, 9976.14526286, 5)
        nose.tools.assert_almost_equal(self.m[0].centre.value, -0.110610718444)
        nose.tools.assert_almost_equal(self.m[0].sigma.value, 1.98380707614)

    def test_fit_odr(self):
        self.m.fit(fitter="odr")
        nose.tools.assert_almost_equal(self.m[0].A.value, 9976.14531979, 3)
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
        nose.tools.assert_almost_equal(self.m[0].A.value, 9976.14531979, 3)
        nose.tools.assert_almost_equal(self.m[0].centre.value, -0.110610724054)
        nose.tools.assert_almost_equal(self.m[0].sigma.value, 1.98380709939)

    def test_fit_bounded(self):
        self.m[0].centre.bmin = 0.5
        self.m[0].bounded = True
        self.m.fit(fitter="mpfit", bounded=True)
        nose.tools.assert_almost_equal(self.m[0].A.value, 9991.65422046, 5)
        nose.tools.assert_almost_equal(self.m[0].centre.value, 0.5)
        nose.tools.assert_almost_equal(self.m[0].sigma.value, 2.08398236966)

    @nose.tools.raises(ValueError)
    def test_wrong_method(self):
        self.m.fit(method="dummy")


class TestModelWeighted:

    def setUp(self):
        np.random.seed(1)
        s = signals.SpectrumSimulation(np.arange(10, 100, 0.1))
        s.metadata.set_item("Signal.Noise_properties.variance",
                            signals.Spectrum(np.arange(10, 100, 0.01)))
        s.axes_manager[0].scale = 0.1
        s.axes_manager[0].offset = 10
        s.add_poissonian_noise()
        m = create_model(s)
        m.append(components.Polynomial(1))
        self.m = m

    def test_fit_leastsq_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        self.m.fit(fitter="leastsq", method="ls")
        for result, expected in zip(self.m[0].coefficients.value,
                                    (9.9165596693502778, 1.6628238107916631)):
            nose.tools.assert_almost_equal(result, expected, places=5)

    def test_fit_odr_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        self.m.fit(fitter="odr", method="ls")
        for result, expected in zip(self.m[0].coefficients.value,
                                    (9.9165596548961972, 1.6628247412317521)):
            nose.tools.assert_almost_equal(result, expected, places=5)

    def test_fit_mpfit_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        self.m.fit(fitter="mpfit", method="ls")
        for result, expected in zip(self.m[0].coefficients.value,
                                    (9.9165596607108739, 1.6628243846485873)):
            nose.tools.assert_almost_equal(result, expected, places=5)

    def test_fit_fmin_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        self.m.fit(
            fitter="fmin",
            method="ls",
            weights=np.arange(
                10,
                100,
                0.01))
        for result, expected in zip(self.m[0].coefficients.value,
                                    (9.9137288425667442, 1.8446013472266145)):
            nose.tools.assert_almost_equal(result, expected, places=5)

    def test_fit_leastsq_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        self.m.fit(fitter="leastsq", method="ls")
        for result, expected in zip(self.m[0].coefficients.value,
                                    (0.99165596391487121, 0.16628254242532492)):
            nose.tools.assert_almost_equal(result, expected, places=5)

    def test_fit_odr_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        self.m.fit(fitter="odr", method="ls")
        for result, expected in zip(self.m[0].coefficients.value,
                                    (0.99165596548961943, 0.16628247412317315)):
            nose.tools.assert_almost_equal(result, expected, places=5)

    def test_fit_mpfit_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        self.m.fit(fitter="mpfit", method="ls")
        for result, expected in zip(self.m[0].coefficients.value,
                                    (0.99165596295068958, 0.16628257462820528)):
            nose.tools.assert_almost_equal(result, expected, places=5)

    def test_fit_fmin_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        self.m.fit(
            fitter="fmin",
            method="ls",
            weights=np.arange(
                10,
                100,
                0.01))
        for result, expected in zip(self.m[0].coefficients.value,
                                    (0.99136169230026261, 0.18483060534056939)):
            nose.tools.assert_almost_equal(result, expected, places=5)

    def test_chisq(self):
        self.m.spectrum.metadata.Signal.binned = True
        self.m.fit(fitter="leastsq", method="ls")
        nose.tools.assert_almost_equal(self.m.chisq.data, 3029.16949561)

    def test_red_chisq(self):
        self.m.fit(fitter="leastsq", method="ls")
        nose.tools.assert_almost_equal(self.m.red_chisq.data, 3.37700055)


class TestModelScalarVariance:

    def setUp(self):
        s = signals.SpectrumSimulation(np.ones(100))
        m = create_model(s)
        m.append(components.Offset())
        self.s = s
        self.m = m

    def test_std1_chisq(self):
        std = 1
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.metadata.set_item("Signal.Noise_properties.variance", std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        nose.tools.assert_almost_equals(self.m.chisq.data, 78.35015229)

    def test_std10_chisq(self):
        std = 10
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.metadata.set_item("Signal.Noise_properties.variance", std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        nose.tools.assert_almost_equals(self.m.chisq.data, 78.35015229)

    def test_std1_red_chisq(self):
        std = 1
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.metadata.set_item("Signal.Noise_properties.variance", std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        nose.tools.assert_almost_equals(self.m.red_chisq.data, 0.79949135)

    def test_std10_red_chisq(self):
        std = 10
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.metadata.set_item("Signal.Noise_properties.variance", std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        nose.tools.assert_almost_equals(self.m.red_chisq.data, 0.79949135)

    def test_std1_red_chisq(self):
        std = 1
        self.m.set_signal_range(10, 50)
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.metadata.set_item("Signal.Noise_properties.variance", std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        nose.tools.assert_almost_equals(self.m.red_chisq.data, 0.86206965)


class TestModelSignalVariance:

    def setUp(self):
        variance = signals.SpectrumSimulation(
            np.arange(
                100, 300).reshape(
                (2, 100)))
        s = variance.deepcopy()
        np.random.seed(1)
        std = 10
        s.add_gaussian_noise(std)
        s.add_poissonian_noise()
        s.metadata.set_item("Signal.Noise_properties.variance",
                            variance + std ** 2)
        m = create_model(s)
        m.append(components.Polynomial(order=1))
        self.s = s
        self.m = m

    def test_std1_red_chisq(self):
        self.m.multifit(fitter="leastsq", method="ls")
        nose.tools.assert_almost_equals(self.m.red_chisq.data[0],
                                        0.79693355673230915)
        nose.tools.assert_almost_equals(self.m.red_chisq.data[1],
                                        0.91453032901427167)
