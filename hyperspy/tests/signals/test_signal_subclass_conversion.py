# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

import logging

import dask.array as da
import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.decorators import lazifyTestClass
from hyperspy.exceptions import DataDimensionError
from hyperspy.signals import BaseSignal


@lazifyTestClass
class Test1d:
    def setup_method(self, method):
        self.s = hs.signals.BaseSignal(np.arange(2))

    def test_as_signal2D(self):
        with pytest.raises(DataDimensionError):
            assert (self.s.data == self.s.as_signal2D((0, 1)).data).all()

    def test_as_signal1D(self):
        assert (self.s.data == self.s.as_signal1D(0).data).all()


@lazifyTestClass
class Test2d:
    def setup_method(self, method):
        self.s = hs.signals.BaseSignal(np.random.random((2, 3)))  # (|3, 2)

    def test_as_signal2D_T(self):
        assert self.s.data.T.shape == self.s.as_signal2D((1, 0)).data.shape

    def test_as_signal2D(self):
        assert self.s.data.shape == self.s.as_signal2D((0, 1)).data.shape

    def test_as_signal1D_T(self):
        assert self.s.data.T.shape == self.s.as_signal1D(1).data.shape

    def test_as_signal1D(self):
        assert self.s.data.shape == self.s.as_signal1D(0).data.shape


@lazifyTestClass
class Test3d:
    def setup_method(self, method):
        self.s = BaseSignal(np.random.random((2, 3, 4)))  # (|4, 3, 2)

    def test_as_signal2D_contigous(self):
        if self.s._lazy:
            pytest.skip("Dask array flags not supported")
        assert self.s.as_signal2D((0, 1)).data.flags["C_CONTIGUOUS"]

    def test_as_signal2D_1(self):
        assert self.s.as_signal2D((0, 1)).data.shape == (2, 3, 4)  # (2| 4, 3)

    def test_as_signal2D_2(self):
        assert self.s.as_signal2D((1, 0)).data.shape == (2, 4, 3)  # (2| 3, 4)

    def test_as_signal2D_3(self):
        assert self.s.as_signal2D((1, 2)).data.shape == (4, 2, 3)  # (4| 3, 2)

    def test_as_signal1D_contigous(self):
        if self.s._lazy:
            pytest.skip("Dask array flags not supported")
        assert self.s.as_signal1D(0).data.flags["C_CONTIGUOUS"]

    def test_as_signal1D_0(self):
        assert self.s.as_signal1D(0).data.shape == (2, 3, 4)  # (3, 2| 4)

    def test_as_signal1D_1(self):
        assert self.s.as_signal1D(1).data.shape == (2, 4, 3)  # (4, 2| 3)

    def test_as_signal1D_2(self):
        assert self.s.as_signal1D(2).data.shape == (3, 4, 2)  # (4, 3| 2)

    def test_remove_axis(self):
        im = self.s.as_signal2D((-2, -1))
        im._remove_axis(-1)
        assert isinstance(im, hs.signals.Signal1D)


def test_as_lazy_chunks(caplog):
    data = np.ones((100, 100, 2000))
    s = hs.signals.Signal1D(data)

    # chunks is "auto"
    s2 = s.as_lazy()
    assert isinstance(s2.data, da.Array)
    assert s2.data.chunks == ((50, 50), (100,), (2000,))

    s3 = s.as_lazy(chunks="auto")
    assert s3.data.chunks == ((50, 50), (100,), (2000,))

    s4 = s.as_lazy(chunks="dask_auto")
    assert s4.data.chunks == ((100,), (100,), (1677, 323))

    s5 = s.as_lazy(chunks=(25, 25, 500))
    assert s5.data.chunks == ((25,) * 4, (25,) * 4, (500,) * 4)

    # ignore `chunks` since data is already a dask array
    s6 = s5.as_lazy(chunks="dask_auto")
    with caplog.at_level(logging.WARNING):
        assert s6.data.chunks == s5.data.chunks
    assert "Ignoring `chunks` argument" in caplog.text
