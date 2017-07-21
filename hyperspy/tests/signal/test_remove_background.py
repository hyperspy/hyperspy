import numpy as np

from hyperspy import signals
from hyperspy import components1d
from hyperspy.decorators import lazifyTestClass


@lazifyTestClass
class TestRemoveBackground1DGaussian:

    def setup_method(self, method):
        gaussian = components1d.Gaussian()
        gaussian.A.value = 10
        gaussian.centre.value = 10
        gaussian.sigma.value = 1
        self.signal = signals.Signal1D(
            gaussian.function(np.arange(0, 20, 0.01)))
        self.signal.axes_manager[0].scale = 0.01
        self.signal.metadata.Signal.binned = False

    def test_background_remove_gaussian(self):
        s1 = self.signal.remove_background(
            signal_range=(None, None),
            background_type='Gaussian',
            show_progressbar=None)
        assert np.allclose(s1.data, np.zeros(len(s1.data)))

    def test_background_remove_gaussian_full_fit(self):
        s1 = self.signal.remove_background(
            signal_range=(None, None),
            background_type='Gaussian',
            fast=False)
        assert np.allclose(s1.data, np.zeros(len(s1.data)))


@lazifyTestClass
class TestRemoveBackground1DPowerLaw:

    def setup_method(self, method):
        pl = components1d.PowerLaw()
        pl.A.value = 1e10
        pl.r.value = 3
        self.signal = signals.Signal1D(
            pl.function(np.arange(100, 200)))
        self.signal.axes_manager[0].offset = 100
        self.signal.metadata.Signal.binned = False

        self.signal_noisy = self.signal.deepcopy()
        self.signal_noisy.add_gaussian_noise(1)

    def test_background_remove_pl(self):
        s1 = self.signal.remove_background(
            signal_range=(None, None),
            background_type='PowerLaw',
            show_progressbar=None)
        assert np.allclose(s1.data, np.zeros(len(s1.data)), atol=60)

    def test_background_remove_pl_zero(self):
        s1 = self.signal_noisy.remove_background(
            signal_range=(110.0, 190.0),
            background_type='PowerLaw',
            zero_fill=True,
            show_progressbar=None)
        assert np.allclose(s1.sum(-1).data, np.array([3787]), atol=200)
        assert np.allclose(s1.data[:10], np.zeros(10), atol=0.5)

    def test_background_remove_pl_int(self):
        self.signal.change_dtype("int")
        s1 = self.signal.remove_background(
            signal_range=(None, None),
            background_type='PowerLaw',
            show_progressbar=None)
        assert np.allclose(s1.data, np.zeros(len(s1.data)), atol=60)

    def test_background_remove_pl_int_zero(self):
        self.signal_noisy.change_dtype("int")
        s1 = self.signal_noisy.remove_background(
            signal_range=(110.0, 190.0),
            background_type='PowerLaw',
            zero_fill=True,
            show_progressbar=None)
        assert np.allclose(s1.sum(-1).data, np.array([3787]), atol=200)
        assert np.allclose(s1.data[:10], np.zeros(10), atol=0.5)
