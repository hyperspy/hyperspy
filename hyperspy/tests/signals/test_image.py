# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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

from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D
from hyperspy.decorators import lazifyTestClass


@lazifyTestClass
class Test2D:

    def setup_method(self, method):
        self.im = Signal2D(np.random.random((2, 3)))

    def test_to_signal1D(self):
        s = self.im.to_signal1D()
        assert isinstance(s, Signal1D)
        assert s.data.shape == self.im.data.T.shape
        if not s._lazy:
            assert s.data.flags["C_CONTIGUOUS"]


@lazifyTestClass
class Test3D:

    def setup_method(self, method):
        self.im = Signal2D(np.random.random((2, 3, 4)))

    def test_to_signal1D(self):
        s = self.im.to_signal1D()
        assert isinstance(s, Signal1D)
        assert s.data.shape == (3, 4, 2)
        if not s._lazy:
            assert s.data.flags["C_CONTIGUOUS"]


@lazifyTestClass
class Test4D:

    def setup_method(self, method):
        self.s = Signal2D(np.random.random((2, 3, 4, 5)))

    def test_to_image(self):
        s = self.s.to_signal1D()
        assert isinstance(s, Signal1D)
        assert s.data.shape == (3, 4, 5, 2)
        if not s._lazy:
            assert s.data.flags["C_CONTIGUOUS"]
