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
import pytest

from hyperspy import _lazy_signals, signals
from hyperspy.decorators import lazifyTestClass
from hyperspy.exceptions import DataDimensionError
from hyperspy.signals import BaseSignal


@lazifyTestClass
class Test1d:

    def setup_method(self, method):
        self.s = BaseSignal(np.arange(2))

    def test_as_signal2D(self):
        with pytest.raises(DataDimensionError):
            assert (self.s.data == self.s.as_signal2D(
                    (0, 1)).data).all()

    def test_as_signal1D(self):
        assert (self.s.data == self.s.as_signal1D(0).data).all()

    def test_set_EELS(self):
        s = self.s.as_signal1D(0)
        s.set_signal_type("EELS")
        assert s.metadata.Signal.signal_type == "EELS"
        if s._lazy:
            _class = _lazy_signals.LazyEELSSpectrum
        else:
            _class = signals.EELSSpectrum
        assert isinstance(s, _class)


@lazifyTestClass
class Test2d:

    def setup_method(self, method):
        self.s = BaseSignal(np.random.random((2, 3)))  # (|3, 2)

    def test_as_signal2D_T(self):
        assert (
            self.s.data.T.shape == self.s.as_signal2D((1, 0)).data.shape)

    def test_as_signal2D(self):
        assert (
            self.s.data.shape == self.s.as_signal2D((0, 1)).data.shape)

    def test_as_signal1D_T(self):
        assert (
            self.s.data.T.shape == self.s.as_signal1D(1).data.shape)

    def test_as_signal1D(self):
        assert (
            self.s.data.shape == self.s.as_signal1D(0).data.shape)

    def test_s2EELS2im2s(self):
        s = self.s.as_signal1D(0)
        s.set_signal_type("EELS")
        im = s.as_signal2D((1, 0))
        assert im.metadata.Signal.signal_type == "EELS"
        s = im.as_signal1D(0)
        assert s.metadata.Signal.signal_type == "EELS"
        if s._lazy:
            _class = _lazy_signals.LazyEELSSpectrum
        else:
            _class = signals.EELSSpectrum
        assert isinstance(s, _class)


@lazifyTestClass
class Test3d:

    def setup_method(self, method):
        self.s = BaseSignal(np.random.random((2, 3, 4)))  # (|4, 3, 2)

    def test_as_signal2D_contigous(self):
        if self.s._lazy:
            pytest.skip("Dask array flags not supported")
        assert self.s.as_signal2D((0, 1)).data.flags['C_CONTIGUOUS']

    def test_as_signal2D_1(self):
        assert (
            self.s.as_signal2D((0, 1)).data.shape == (2, 3, 4))  # (2| 4, 3)

    def test_as_signal2D_2(self):
        assert (
            self.s.as_signal2D((1, 0)).data.shape == (2, 4, 3))  # (2| 3, 4)

    def test_as_signal2D_3(self):
        assert (
            self.s.as_signal2D((1, 2)).data.shape == (4, 2, 3))  # (4| 3, 2)

    def test_as_signal1D_contigous(self):
        if self.s._lazy:
            pytest.skip("Dask array flags not supported")
        assert self.s.as_signal1D(0).data.flags['C_CONTIGUOUS']

    def test_as_signal1D_0(self):
        assert (
            self.s.as_signal1D(0).data.shape == (2, 3, 4))  # (3, 2| 4)

    def test_as_signal1D_1(self):
        assert (
            self.s.as_signal1D(1).data.shape == (2, 4, 3))  # (4, 2| 3)

    def test_as_signal1D_2(self):
        assert (
            self.s.as_signal1D(2).data.shape == (3, 4, 2))  # (4, 3| 2)

    def test_remove_axis(self):
        im = self.s.as_signal2D((-2, -1))
        im._remove_axis(-1)
        assert isinstance(im, signals.Signal1D)
