import nose.tools as nt
import numpy as np

from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D


class Test2D:

    def setUp(self):
        self.im = Signal2D(np.random.random((2, 3)))

    def test_to_signal1D(self):
        s = self.im.to_signal1D()
        nt.assert_true(isinstance(s, Signal1D))
        nt.assert_equal(s.data.shape, self.im.data.T.shape)
        nt.assert_true(s.data.flags["C_CONTIGUOUS"])


class Test3D:

    def setUp(self):
        self.im = Signal2D(np.random.random((2, 3, 4)))

    def test_to_signal1D(self):
        s = self.im.to_signal1D()
        nt.assert_true(isinstance(s, Signal1D))
        nt.assert_equal(s.data.shape, (3, 4, 2))
        nt.assert_true(s.data.flags["C_CONTIGUOUS"])


class Test4D:

    def setUp(self):
        self.s = Signal2D(np.random.random((2, 3, 4, 5)))

    def test_to_image(self):
        s = self.s.to_signal1D()
        nt.assert_true(isinstance(s, Signal1D))
        nt.assert_equal(s.data.shape, (3, 4, 5, 2))
        nt.assert_true(s.data.flags["C_CONTIGUOUS"])
