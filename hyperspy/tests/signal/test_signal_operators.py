# Copyright 2007-2011 The HyperSpy developers
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
from nose.tools import (
    assert_true,
    assert_equal,
    raises)

from hyperspy.signal import Signal


class TestBinaryOperatorsCase1:

    """The signals are not aligned but can be aligned because their
    shapes are as follows:

    +---------+------------------+-------------+
    | Signal  | NavigationShape  | SignalShape |
    +---------+------------------+-------------+
    +---------+------------------+-------------+
    |   s1    |        a         |      b      |
    +---------+------------------+-------------+
    |   s2    |       (0,)       |      a      |
    +---------+------------------+-------------+

    """

    def setUp(self):
        self.s1 = Signal(np.arange(20).reshape(2, 2, 5))
        self.s2 = Signal(np.arange(4).reshape(2, 2))
        self.s2.axes_manager.set_signal_dimension(2)
        self.s3 = Signal(np.arange(4).reshape(1, 2, 2))
        self.s3.axes_manager.set_signal_dimension(2)

    def test_s1_plus_s2(self):
        n = self.s1 + self.s2
        assert_true((n.data ==
                     self.s1.data + self.s2.data[..., np.newaxis]).all())
        for i, size in enumerate(n.data.shape):
            assert_equal(size, n.axes_manager._axes[i].size)

    def test_s2_plus_s1(self):
        n = self.s2 + self.s1
        assert_true((n.data ==
                     self.s1.data + self.s2.data[..., np.newaxis]).all())
        for i, size in enumerate(n.data.shape):
            assert_equal(size, n.axes_manager._axes[i].size)

    @raises(ValueError)
    def test_s3_plus_s1(self):
        self.s3 + self.s1


class TestBinaryOperatorsCase2:

    """The signals are not aligned but can be aligned because their
    shapes are as follows:

    +---------+------------------+-------------+
    | Signal  | NavigationShape  | SignalShape |
    +---------+------------------+-------------+
    +---------+------------------+-------------+
    |   s1    |        a         |      b      |
    +---------+------------------+-------------+
    |   s2    |       (0,)       |      b      |
    +---------+------------------+-------------+

    """

    def setUp(self):
        self.s1 = Signal(np.arange(20).reshape(2, 2, 5))
        self.s2 = Signal(np.arange(5))
        self.s3 = Signal(np.arange(5).reshape(1, 5))

    def test_s1_plus_s2(self):
        n = self.s1 + self.s2
        assert_true((n.data ==
                     self.s1.data + self.s2.data).all())
        for i, size in enumerate(n.data.shape):
            assert_equal(size, n.axes_manager._axes[i].size)

    def test_s2_plus_s1(self):
        n = self.s2 + self.s1
        assert_true((n.data ==
                     self.s1.data + self.s2.data).all())
        for i, size in enumerate(n.data.shape):
            assert_equal(size, n.axes_manager._axes[i].size)

    def test_s3_plus_s1(self):
        n = self.s3 + self.s1
        assert_true((n.data ==
                     self.s1.data + self.s2.data).all())
        for i, size in enumerate(n.data.shape):
            assert_equal(size, n.axes_manager._axes[i].size)


class TestBinaryOperatorsCase3:

    """The signals are not aligned but can be aligned because their
    shapes are as follows:

    +---------+------------------+-------------+
    | Signal  | NavigationShape  | SignalShape |
    +---------+------------------+-------------+
    +---------+------------------+-------------+
    |   s1    |       (0,)       |      a      |
    +---------+------------------+-------------+
    |   s2    |       (0,)       |      b      |
    +---------+------------------+-------------+

    """

    def setUp(self):
        self.s1 = Signal(np.arange(10).reshape(2, 5))
        self.s1.axes_manager.set_signal_dimension(2)
        self.s2 = Signal(np.arange(6))

    def test_s1_times_s2(self):
        n = self.s1 * self.s2
        assert_true((n.data ==
                     self.s1.data[np.newaxis, ...] *
                     self.s2.data[:, np.newaxis, np.newaxis]).all())
        for i, size in enumerate(n.data.shape):
            assert_equal(size, n.axes_manager._axes[i].size)
        assert_equal(n.axes_manager.signal_dimension, 2)

    def test_s2_times_s1(self):
        n = self.s2 * self.s1
        assert_true((n.data ==
                     (self.s2.data[np.newaxis, np.newaxis, :] *
                      self.s1.data[..., np.newaxis])).all())
        for i, size in enumerate(n.data.shape):
            assert_equal(size, n.axes_manager._axes[i].size)
        assert_equal(n.axes_manager.signal_dimension, 1)


class TestBinaryOperatorsCase4:

    """Signal and number.

    """

    def setUp(self):
        self.s1 = Signal(np.arange(6))

    def test_plus_right(self):
        assert_true(((self.s1 + 2).data == self.s1.data + 2).all())

    @raises(TypeError)
    def test_left_right(self):
        assert_true(((2 + self.s1).data == self.s1.data + 2).all())


class TestUnaryOperators:

    def setUp(self):
        self.s1 = Signal(np.array((1, -1, 4, -3)))

    def test_minus(self):
        assert_true(((-self.s1).data == -(self.s1.data)).all())

    def test_plus(self):
        assert_true(((+self.s1).data == +(self.s1.data)).all())

    def test_invert(self):
        assert_true(((~self.s1).data == ~(self.s1.data)).all())

    def test_abs(self):
        assert_true((abs(self.s1).data == abs(self.s1.data)).all())
