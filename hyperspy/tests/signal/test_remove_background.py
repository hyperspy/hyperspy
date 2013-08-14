import numpy as np
import copy
from nose.tools import (
    assert_true,
    assert_equal,
    assert_not_equal,
    raises)

from hyperspy.signal import Signal
from hyperspy import signals
from hyperspy import components

class Test2D:
    def setUp(self):
        self.xaxis = np.arange(0,10000,0.01) 

        self.gaussian = components.Gaussian()
        self.gaussian.A.value = 10000
        self.gaussian.centre.value = 5000
        self.gaussian.sigma.value = 1000

    def test_background_remove_gaussian(self):
        gaussian = copy.deepcopy(self.gaussian)

        signal = signals.Signal(gaussian.function(self.xaxis))
        s1 = signal.remove_background(
                signal_range=(200000,800000), 
                background_type='Gaussian')
        self.s1 = s1
        assert_true(
                np.allclose(
                    s1.data,
                    np.zeros(len(s1.data)),
                    atol=0.00001*self.gaussian.A.value))
