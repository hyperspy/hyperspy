import nose.tools as nt
import numpy as np

from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D
from hyperspy.decorators import lazifyTestClass


@lazifyTestClass
class Test2D:

    def setUp(self):
        self.s = Signal1D(np.random.random((2, 3)))

    def test_to_signal2D(self):
        im = self.s.to_signal2D()
        nt.assert_true(isinstance(im, Signal2D))
        nt.assert_equal(im.data.shape, self.s.data.T.shape)
        if not im._lazy:
            nt.assert_true(im.data.flags["C_CONTIGUOUS"])


@lazifyTestClass
class Test3D:

    def setUp(self):
        self.s = Signal1D(np.random.random((2, 3, 4)))

    def test_to_signal2D(self):
        im = self.s.to_signal2D()
        nt.assert_true(isinstance(im, Signal2D))
        nt.assert_equal(im.data.shape, (4, 2, 3))
        if not im._lazy:
            nt.assert_true(im.data.flags["C_CONTIGUOUS"])


@lazifyTestClass
class Test4D:

    def setUp(self):
        self.s = Signal1D(np.random.random((2, 3, 4, 5)))

    def test_to_signal2D(self):
        im = self.s.to_signal2D()
        nt.assert_true(isinstance(im, Signal2D))
        nt.assert_equal(im.data.shape, (5, 2, 3, 4))
        if not im._lazy:
            nt.assert_true(im.data.flags["C_CONTIGUOUS"])
