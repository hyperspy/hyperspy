import numpy as np
import nose.tools as nt

from hyperspy.misc import rgb_tools
import hyperspy.api as hs


class TestRGBA8:

    def setUp(self):
        self.s = hs.signals.Signal1D(np.array(
            [[[1, 1, 1, 0],
              [2, 2, 2, 0]],
             [[3, 3, 3, 0],
              [4, 4, 4, 0]]],
            dtype="uint8"))
        self.im = hs.signals.Signal1D(np.array(
            [[(1, 1, 1, 0), (2, 2, 2, 0)],
             [(3, 3, 3, 0), (4, 4, 4, 0)]],
            dtype=rgb_tools.rgba8))

    def test_torgb(self):
        self.s.change_dtype("rgba8")
        np.testing.assert_array_equal(self.s.data, self.im.data)

    def test_touint(self):
        self.im.change_dtype("uint8")
        np.testing.assert_array_equal(self.s.data, self.im.data)
        nt.assert_equal(len(self.im.axes_manager._axes), 3)
        nt.assert_equal(self.im.axes_manager.signal_axes[0].name, "RGB index")

    @nt.raises(AttributeError)
    def test_wrong_bs(self):
        self.s.change_dtype("rgba16")

    @nt.raises(AttributeError)
    def test_wrong_rgb(self):
        self.im.change_dtype("rgb8")


class TestRGBA16:

    def setUp(self):
        self.s = hs.signals.Signal1D(np.array(
            [[[1, 1, 1, 0],
              [2, 2, 2, 0]],
             [[3, 3, 3, 0],
              [4, 4, 4, 0]]],
            dtype="uint16"))
        self.im = hs.signals.Signal1D(np.array(
            [[(1, 1, 1, 0), (2, 2, 2, 0)],
             [(3, 3, 3, 0), (4, 4, 4, 0)]],
            dtype=rgb_tools.rgba16))

    def test_torgb(self):
        self.s.change_dtype("rgba16")
        np.testing.assert_array_equal(self.s.data, self.im.data)

    def test_touint(self):
        self.im.change_dtype("uint16")
        np.testing.assert_array_equal(self.s.data, self.im.data)

    @nt.raises(AttributeError)
    def test_wrong_bs(self):
        self.s.change_dtype("rgba8")

    @nt.raises(AttributeError)
    def test_wrong_rgb(self):
        self.im.change_dtype("rgb16")
