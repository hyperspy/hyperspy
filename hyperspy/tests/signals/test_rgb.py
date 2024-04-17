# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import numpy as np
import pytest
from rsciio.utils import rgb_tools

import hyperspy.api as hs


class TestRGBA8:
    def setup_method(self, method):
        self.s = hs.signals.Signal1D(
            np.array(
                [[[1, 1, 1, 0], [2, 2, 2, 0]], [[3, 3, 3, 0], [4, 4, 4, 0]]],
                dtype="uint8",
            )
        )
        self.im = hs.signals.Signal1D(
            np.array(
                [[(1, 1, 1, 0), (2, 2, 2, 0)], [(3, 3, 3, 0), (4, 4, 4, 0)]],
                dtype=rgb_tools.rgba8,
            )
        )

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

    def test_change_dtype_with_plot(self):
        # Check that changing dtype with a plot open
        # doesn't raise any error
        s = self.s
        s.plot()
        assert s._plot.navigator_plot is not None
        s.change_dtype("rgba8")
        assert s._plot.navigator_plot is None
        s.change_dtype("uint8")
        assert s._plot.navigator_plot is not None


class TestRGBA16:
    def setup_method(self, method):
        self.s = hs.signals.Signal1D(
            np.array(
                [[[1, 1, 1, 0], [2, 2, 2, 0]], [[3, 3, 3, 0], [4, 4, 4, 0]]],
                dtype="uint16",
            )
        )
        self.im = hs.signals.Signal1D(
            np.array(
                [[(1, 1, 1, 0), (2, 2, 2, 0)], [(3, 3, 3, 0), (4, 4, 4, 0)]],
                dtype=rgb_tools.rgba16,
            )
        )

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
