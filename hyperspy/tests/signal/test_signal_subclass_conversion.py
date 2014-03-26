from nose.tools import assert_true, assert_equal, raises
import numpy as np

from hyperspy.signal import Signal
from hyperspy import signals
from hyperspy.exceptions import DataDimensionError


class Test1d():

    def setUp(self):
        self.s = Signal(np.arange(2))

    @raises(DataDimensionError)
    def test_as_image(self):
        assert_true((self.s.data == self.s.as_image((0, 1)).data).all())

    def test_as_spectrum(self):
        assert_true((self.s.data == self.s.as_spectrum(0).data).all())

    def test_set_EELS(self):
        s = self.s.as_spectrum(0)
        s.set_signal_type("EELS")
        assert_equal(s.metadata.Signal.signal_type, "EELS")
        assert_true(isinstance(s, signals.EELSSpectrum))


class Test2d():

    def setUp(self):
        self.s = Signal(np.random.random((2, 3)))

    def test_as_image_T(self):
        assert_true(
            self.s.data.T.shape == self.s.as_image((0, 1)).data.shape)

    def test_as_image(self):
        assert_true(
            self.s.data.shape == self.s.as_image((1, 0)).data.shape)

    def test_as_spectrum_T(self):
        assert_true(
            self.s.data.T.shape == self.s.as_spectrum(0).data.shape)

    def test_as_spectrum(self):
        assert_true(
            self.s.data.shape == self.s.as_spectrum(1).data.shape)

    def test_s2EELS2im2s(self):
        s = self.s.as_spectrum(0)
        s.set_signal_type("EELS")
        im = s.as_image((1, 0))
        assert_equal(im.metadata.Signal.signal_type, "EELS")
        s = im.as_spectrum((0))
        assert_equal(s.metadata.Signal.signal_type, "EELS")
        assert_true(isinstance(s, signals.EELSSpectrum))


class Test3d():

    def setUp(self):
        self.s = Signal(np.random.random((2, 3, 4)))

    def test_as_image_contigous(self):
        assert_true(self.s.as_image((0, 1)).data.flags['C_CONTIGUOUS'])

    def test_as_image_1(self):
        assert_equal(
            self.s.as_image((0, 1)).data.shape, (4, 2, 3))

    def test_as_image_2(self):
        assert_equal(
            self.s.as_image((1, 0)).data.shape, (4, 3, 2))

    def test_as_image_3(self):
        assert_equal(
            self.s.as_image((1, 2)).data.shape, (3, 4, 2))

    def test_as_spectrum_contigous(self):
        assert_true(self.s.as_spectrum(0).data.flags['C_CONTIGUOUS'])

    def test_as_spectrum_0(self):
        assert_equal(
            self.s.as_spectrum(0).data.shape, (2, 4, 3))

    def test_as_spectrum_1(self):
        assert_equal(
            self.s.as_spectrum(1).data.shape, (3, 4, 2))

    def test_as_spectrum_2(self):
        assert_equal(
            self.s.as_spectrum(1).data.shape, (3, 4, 2))

    def test_as_spectrum_3(self):
        assert_equal(
            self.s.as_spectrum(2).data.shape, (2, 3, 4))

    def test_remove_axis(self):
        im = self.s.as_image((-2, -1))
        im._remove_axis(-1)
        assert_true(isinstance(im, signals.Spectrum))
