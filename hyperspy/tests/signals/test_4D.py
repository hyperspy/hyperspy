# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

from hyperspy.decorators import lazifyTestClass
from hyperspy.signals import Signal1D


@lazifyTestClass
class Test4D:
    def setup_method(self, method):
        s = Signal1D(np.ones((5, 4, 3, 6)))
        for axis, name in zip(
            s.axes_manager._get_axes_in_natural_order(), ["x", "y", "z", "E"]
        ):
            axis.name = name
        self.s = s

    def test_diff_data(self):
        s = self.s
        diff = s.diff(axis=2, order=2)
        diff_data = np.diff(s.data, n=2, axis=0)
        np.testing.assert_array_equal(diff.data, diff_data)

    def test_diff_axis(self):
        s = self.s
        diff = s.diff(axis=2, order=2)
        assert (
            diff.axes_manager[2].offset
            == s.axes_manager[2].offset + s.axes_manager[2].scale
        )

    def test_rollaxis_int(self):
        assert self.s.rollaxis(2, 0).data.shape == (4, 3, 5, 6)

    def test_rollaxis_str(self):
        assert self.s.rollaxis("z", "x").data.shape == (4, 3, 5, 6)

    def test_unfold_spectrum(self):
        self.s.unfold()
        assert self.s.data.shape == (60, 6)

    def test_unfold_spectrum_returns_true(self):
        assert self.s.unfold()

    def test_unfold_spectrum_signal_returns_false(self):
        assert not self.s.unfold_signal_space()

    def test_unfold_image(self):
        im = self.s.to_signal2D()
        im.unfold()
        assert im.data.shape == (30, 12)

    def test_image_signal_unfolded_deepcopy(self):
        im = self.s.to_signal2D()
        im.unfold()
        # The following could fail if the constructor was not taking the fact
        # that the signal is unfolded into account when setting the signal
        # dimension.
        im.deepcopy()

    def test_image_signal_unfolded_false(self):
        im = self.s.to_signal2D()
        assert not im.metadata._HyperSpy.Folding.signal_unfolded

    def test_image_signal_unfolded_true(self):
        im = self.s.to_signal2D()
        im.unfold()
        assert im.metadata._HyperSpy.Folding.signal_unfolded

    def test_image_signal_unfolded_back_to_false(self):
        im = self.s.to_signal2D()
        im.unfold()
        im.fold()
        assert not im.metadata._HyperSpy.Folding.signal_unfolded
