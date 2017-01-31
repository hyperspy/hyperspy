import numpy as np

import pytest

from hyperspy.misc import rgb_tools
import hyperspy.api as hs


class TestRGBA8:

    def setup_method(self, method):
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
        assert len(self.im.axes_manager._axes) == 3
        assert self.im.axes_manager.signal_axes[0].name == "RGB index"

    def test_wrong_bs(self):
        with pytest.raises(AttributeError):
            self.s.change_dtype("rgba16")

    def test_wrong_rgb(self):
        with pytest.raises(AttributeError):
            self.im.change_dtype("rgb8")


class TestRGBA16:

    def setup_method(self, method):
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

    def test_wrong_bs(self):
        with pytest.raises(AttributeError):
            self.s.change_dtype("rgba8")

    def test_wrong_rgb(self):
        with pytest.raises(AttributeError):
            self.im.change_dtype("rgb16")
