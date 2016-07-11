import nose.tools as nt
import numpy as np

from hyperspy.signals import BaseSignal
from hyperspy import signals
from hyperspy.exceptions import DataDimensionError


class Test1d:

    def setUp(self):
        self.s = BaseSignal(np.arange(2))

    @nt.raises(DataDimensionError)
    def test_as_signal2D(self):
        nt.assert_true((self.s.data == self.s.as_signal2D((0, 1)).data).all())

    def test_as_signal1D(self):
        nt.assert_true((self.s.data == self.s.as_signal1D(0).data).all())

    def test_set_EELS(self):
        s = self.s.as_signal1D(0)
        s.set_signal_type("EELS")
        nt.assert_equal(s.metadata.Signal.signal_type, "EELS")
        nt.assert_is_instance(s, signals.EELSSpectrum)


class Test2d:

    def setUp(self):
        self.s = BaseSignal(np.random.random((2, 3)))  # (|3, 2)

    def test_as_signal2D_T(self):
        nt.assert_equal(
            self.s.data.T.shape, self.s.as_signal2D((1, 0)).data.shape)

    def test_as_signal2D(self):
        nt.assert_equal(
            self.s.data.shape, self.s.as_signal2D((0, 1)).data.shape)

    def test_as_signal1D_T(self):
        nt.assert_equal(
            self.s.data.T.shape, self.s.as_signal1D(1).data.shape)

    def test_as_signal1D(self):
        nt.assert_equal(
            self.s.data.shape, self.s.as_signal1D(0).data.shape)

    def test_s2EELS2im2s(self):
        s = self.s.as_signal1D(0)
        s.set_signal_type("EELS")
        im = s.as_signal2D((1, 0))
        nt.assert_equal(im.metadata.Signal.signal_type, "EELS")
        s = im.as_signal1D(0)
        nt.assert_equal(s.metadata.Signal.signal_type, "EELS")
        nt.assert_is_instance(s, signals.EELSSpectrum)


class Test3d:

    def setUp(self):
        self.s = BaseSignal(np.random.random((2, 3, 4)))  # (|4, 3, 2)

    def test_as_signal2D_contigous(self):
        nt.assert_true(self.s.as_signal2D((0, 1)).data.flags['C_CONTIGUOUS'])

    def test_as_signal2D_1(self):
        nt.assert_equal(
            self.s.as_signal2D((0, 1)).data.shape, (2, 3, 4))  # (2| 4, 3)

    def test_as_signal2D_2(self):
        nt.assert_equal(
            self.s.as_signal2D((1, 0)).data.shape, (2, 4, 3))  # (2| 3, 4)

    def test_as_signal2D_3(self):
        nt.assert_equal(
            self.s.as_signal2D((1, 2)).data.shape, (4, 2, 3))  # (4| 3, 2)

    def test_as_signal1D_contigous(self):
        nt.assert_true(self.s.as_signal1D(0).data.flags['C_CONTIGUOUS'])

    def test_as_signal1D_0(self):
        nt.assert_equal(
            self.s.as_signal1D(0).data.shape, (2, 3, 4))  # (3, 2| 4)

    def test_as_signal1D_1(self):
        nt.assert_equal(
            self.s.as_signal1D(1).data.shape, (2, 4, 3))  # (4, 2| 3)

    def test_as_signal1D_2(self):
        nt.assert_equal(
            self.s.as_signal1D(2).data.shape, (3, 4, 2))  # (4, 3| 2)

    def test_remove_axis(self):
        im = self.s.as_signal2D((-2, -1))
        im._remove_axis(-1)
        nt.assert_is_instance(im, signals.Signal1D)
