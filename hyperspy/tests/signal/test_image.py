
import numpy as np

from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D
from hyperspy.decorators import lazifyTestClass


@lazifyTestClass
class Test2D:

    def setup_method(self, method):
        self.im = Signal2D(np.random.random((2, 3)))

    def test_to_signal1D(self):
        s = self.im.to_signal1D()
        assert isinstance(s, Signal1D)
        assert s.data.shape == self.im.data.T.shape
        if not s._lazy:
            assert s.data.flags["C_CONTIGUOUS"]


@lazifyTestClass
class Test3D:

    def setup_method(self, method):
        self.im = Signal2D(np.random.random((2, 3, 4)))

    def test_to_signal1D(self):
        s = self.im.to_signal1D()
        assert isinstance(s, Signal1D)
        assert s.data.shape == (3, 4, 2)
        if not s._lazy:
            assert s.data.flags["C_CONTIGUOUS"]


@lazifyTestClass
class Test4D:

    def setup_method(self, method):
        self.s = Signal2D(np.random.random((2, 3, 4, 5)))

    def test_to_image(self):
        s = self.s.to_signal1D()
        assert isinstance(s, Signal1D)
        assert s.data.shape == (3, 4, 5, 2)
        if not s._lazy:
            assert s.data.flags["C_CONTIGUOUS"]
