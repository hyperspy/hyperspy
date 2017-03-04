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

    def test_background_remove_pl(self):
        s1 = self.signal.remove_background(
            signal_range=(None, None),
            background_type='PowerLaw',
            show_progressbar=None)
        assert np.allclose(s1.data, np.zeros(len(s1.data)), atol=60)

    def test_background_remove_pl_int(self):
        self.signal.change_dtype("int")
        s1 = self.signal.remove_background(
            signal_range=(None, None),
            background_type='PowerLaw',
            show_progressbar=None)
        assert np.allclose(s1.data, np.zeros(len(s1.data)), atol=60)
