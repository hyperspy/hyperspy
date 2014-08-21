import nose.tools
import numpy as np

from hyperspy.signals import Spectrum, Image


class Test2D():

    def setUp(self):
        self.s = Spectrum(np.random.random((2, 3)))

    def test_to_image(self):
        im = self.s.to_image()
        nose.tools.assert_true(isinstance(im, Image))
        nose.tools.assert_equal(im.data.shape, self.s.data.T.shape)
        nose.tools.assert_true(im.data.flags["C_CONTIGUOUS"])


class Test3D():

    def setUp(self):
        self.s = Spectrum(np.random.random((2, 3, 4)))

    def test_to_image(self):
        im = self.s.to_image()
        nose.tools.assert_true(isinstance(im, Image))
        nose.tools.assert_equal(im.data.shape, (4, 2, 3))
        nose.tools.assert_true(im.data.flags["C_CONTIGUOUS"])


class Test4D():

    def setUp(self):
        self.s = Spectrum(np.random.random((2, 3, 4, 5)))

    def test_to_image(self):
        im = self.s.to_image()
        nose.tools.assert_true(isinstance(im, Image))
        nose.tools.assert_equal(im.data.shape, (5, 2, 3, 4))
        nose.tools.assert_true(im.data.flags["C_CONTIGUOUS"])
