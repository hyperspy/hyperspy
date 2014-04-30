import numpy as np
from nose.tools import (
    assert_true,)

from hyperspy import signals
from hyperspy import components


class TestRemoveBackground1D:

    def setUp(self):
        gaussian = components.Gaussian()
        gaussian.A.value = 10
        gaussian.centre.value = 10
        gaussian.sigma.value = 1
        self.signal = signals.Spectrum(
            gaussian.function(np.arange(0, 20, 0.01)))
        self.signal.axes_manager[0].scale = 0.01
        self.signal.metadata.Signal.binned = False

    def test_background_remove_gaussian(self):
        s1 = self.signal.remove_background(
            signal_range=(None, None),
            background_type='Gaussian')
        assert_true(np.allclose(s1.data, np.zeros(len(s1.data))))
