from nose.tools import assert_true, assert_equal, raises
import numpy as np

from hyperspy.signals import Spectrum, Image


class Test2D():

    def setUp(self):
        self.im = Image(np.random.random((2, 3)))

    def test_to_image(self):
        s = self.im.to_spectrum()
        assert_true(isinstance(s, Spectrum))
        assert_equal(s.data.shape, self.im.data.T.shape)
        assert_true(s.data.flags["C_CONTIGUOUS"])


class Test3D():

    def setUp(self):
        self.im = Image(np.random.random((2, 3, 4)))

    def test_to_image(self):
        s = self.im.to_spectrum()
        assert_true(isinstance(s, Spectrum))
        assert_equal(s.data.shape, (3, 4, 2))
        assert_true(s.data.flags["C_CONTIGUOUS"])


class Test4D():

    def setUp(self):
        self.s = Image(np.random.random((2, 3, 4, 5)))

    def test_to_image(self):
        s = self.s.to_spectrum()
        assert_true(isinstance(s, Spectrum))
        assert_equal(s.data.shape, (3, 4, 5, 2))
        assert_true(s.data.flags["C_CONTIGUOUS"])
