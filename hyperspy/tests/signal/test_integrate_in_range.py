
import numpy as np

from hyperspy._signals.signal1d import Signal1D
from hyperspy.components1d import Gaussian
from hyperspy.decorators import lazifyTestClass


@lazifyTestClass
class Test1D:

    def setup_method(self, method):
        gaussian = Gaussian()
        gaussian.A.value = 20
        gaussian.sigma.value = 10
        gaussian.centre.value = 50
        self.signal = Signal1D(gaussian.function(np.arange(0, 100, 0.01)))
        self.signal.axes_manager[0].scale = 0.01

    def test_integrate_in_range(self):
        integrated_signal = self.signal.integrate_in_range(signal_range=(None,
                                                                         None))
        assert np.allclose(integrated_signal.data, 20,)
