from nose.tools import assert_true, assert_equal, raises
import numpy as np

from hyperspy.signals import Spectrum, Image


class Test2D():

    def setUp(self):
        self.s = Spectrum(np.random.random((2, 3)))

    def test_to_image(self):
        im = self.s.to_image()
        assert_true(isinstance(im, Image))
        assert_equal(im.data.shape, self.s.data.T.shape)
        assert_true(im.data.flags["C_CONTIGUOUS"])


class Test3D():

    def setUp(self):
        self.s = Spectrum(np.random.random((2, 3, 4)))

    def test_to_image(self):
        im = self.s.to_image()
        assert_true(isinstance(im, Image))
        assert_equal(im.data.shape, (4, 2, 3))
        assert_true(im.data.flags["C_CONTIGUOUS"])


class Test4D():

    def setUp(self):
        self.s = Spectrum(np.random.random((2, 3, 4, 5)))

    def test_to_image(self):
        im = self.s.to_image()
        assert_true(isinstance(im, Image))
        assert_equal(im.data.shape, (5, 2, 3, 4))
        assert_true(im.data.flags["C_CONTIGUOUS"])
