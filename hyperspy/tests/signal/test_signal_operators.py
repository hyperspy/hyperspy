# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from numpy.testing import assert_array_equal

import pytest

from hyperspy import signals
from hyperspy.decorators import lazifyTestClass


@lazifyTestClass
class TestBinaryOperators:

    def setup_method(self, method):
        self.s1 = signals.Signal1D(np.ones((2, 3)))
        self.s2 = signals.Signal1D(np.ones((2, 3)))
        self.s2.data *= 2

    def test_sum_same_shape_signals(self):
        s = self.s1 + self.s2
        assert_array_equal(s.data, self.s1.data * 3)

    def test_sum_in_place_same_shape_signals(self):
        if self.s1._lazy:
            pytest.skip("Inplace not supported by LazySignals")
        s1 = self.s1
        self.s1 += self.s2
        assert_array_equal(self.s1.data, np.ones((2, 3)) * 3)
        assert s1 is self.s1

    def test_sum_same_shape_signals_not_aligned(self):
        s1 = self.s1
        s2 = signals.Signal1D(2 * np.ones((3, 2)))
        if s1._lazy:
            s2 = s2.as_lazy()
        s1.axes_manager._axes[0].navigate = False
        s1.axes_manager._axes[1].navigate = True
        s2.axes_manager._axes[1].navigate = False
        s2.axes_manager._axes[0].navigate = True
        s12 = s1 + s2
        s21 = s2 + s1
        assert_array_equal(s12.data, np.ones((3, 2)) * 3)
        assert_array_equal(s21.data, s12.data)

    def test_sum_in_place_same_shape_signals_not_aligned(self):
        if self.s1._lazy:
            pytest.skip("Inplace not supported by LazySignals")
        s1 = self.s1
        s2 = signals.Signal1D(2 * np.ones((3, 2)))
        s1c = s1
        s2c = s2
        s1.axes_manager._axes[0].navigate = False
        s1.axes_manager._axes[1].navigate = True
        s2.axes_manager._axes[1].navigate = False
        s2.axes_manager._axes[0].navigate = True
        s1 += s2
        assert_array_equal(s1.data, np.ones((3, 2)) * 3)
        s2 += s2
        assert_array_equal(s2.data, np.ones((3, 2)) * 4)
        assert s1 is s1c
        assert s2 is s2c

    def test_sum_wrong_shape(self):
        s1 = self.s1
        s2 = signals.Signal1D(np.ones((3, 3)))
        with pytest.raises(ValueError):
            s1 + s2

    def test_broadcast_missing_sig_and_nav(self):
        s1 = self.s1
        s2 = self.s2.as_signal2D((1, 0))  # (|3, 2)
        s1 = s1.transpose(signal_axes=0)
        s = s1 + s2
        assert_array_equal(s.data, 3 * np.ones((2, 3, 2, 3)))
        assert s.axes_manager.signal_dimension == 2

    def test_broadcast_missing_sig(self):
        s1 = self.s1
        s2 = self.s2
        s1 = s1.transpose(signal_axes=0)
        s2.axes_manager._axes[1].navigate = True
        s2.axes_manager._axes[0].navigate = False  # (3| 2)
        s12 = s1 + s2  # (3, 2| 2)
        s21 = s2 + s1
        assert_array_equal(s12.data, 3 * np.ones((2, 3, 2)))
        assert_array_equal(s21.data, 3 * np.ones((2, 3, 2)))

    def test_broadcast_in_place_missing_sig_wrong(self):
        if self.s1._lazy:
            pytest.skip("Inplace not supported by LazySignals")
        s1 = self.s1
        s2 = self.s2
        s1 = s1.transpose(signal_axes=0)
        s2.axes_manager._axes[1].navigate = True
        s2.axes_manager._axes[0].navigate = False  # (3| 2)
        with pytest.raises(ValueError):
            s1 += s2

    def test_broadcast_in_place(self):
        if self.s1._lazy:
            pytest.skip("Inplace not supported by LazySignals")
        s1 = self.s1
        s1 = s1.transpose(signal_axes=1)
        s2 = signals.BaseSignal(np.ones((4, 2, 4, 3)))
        s2 = s2.transpose(signal_axes=2)
        s2c = s2
        print(s2)
        print(s1)
        s2 += s1
        assert_array_equal(s2.data, 2 * np.ones((4, 2, 4, 3)))
        assert s2 is s2c

    def test_equal_naxes_diff_shape(self):
        s32 = self.s1  # (3| 2)
        s31 = signals.Signal1D(np.ones((1, 3)))
        s12 = signals.Signal1D(np.ones((2, 1)))
        if s32._lazy:
            s31 = s31.as_lazy()
            s12 = s12.as_lazy()
        assert_array_equal((s32 + s31).data, s32.data + 1)
        assert_array_equal((s32 + s12).data, s32.data + 1)


@lazifyTestClass
class TestUnaryOperators:

    def setup_method(self, method):
        self.s1 = signals.Signal1D(np.array((1, -1, 4, -3)))

    def test_minus(self):
        assert_array_equal((-self.s1).data, -self.s1.data)

    def test_plus(self):
        assert_array_equal((+self.s1).data, +self.s1.data)

    def test_invert(self):
        assert_array_equal((~self.s1).data, ~self.s1.data)

    def test_abs(self):
        assert_array_equal(abs(self.s1).data, abs(self.s1.data))
