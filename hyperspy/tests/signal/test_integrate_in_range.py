from nose.tools import assert_true, assert_equal, raises
import numpy as np

from hyperspy.signals import Signal 
from hyperspy.components import Gaussian

class Test1D():
    def setUp(self):
        gaussian = Gaussian() 
        gaussian.A.value = 20
        gaussian.sigma.value = 10
        gaussian.centre.value = 100
        self.signal = Signal(gaussian.function(np.arange(0,1000,0.01)))
    
    def test_integrate_in_range(self):
        integrated_signal = self.signal.integrate_in_range(signal_range=(0,100000))
        assert_true(
                np.allclose(
                    integrated_signal.data,
                    2000,
                    rtol=0.0000001))
                    
