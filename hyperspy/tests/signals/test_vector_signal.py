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
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pytest
import dask.array as da

from hyperspy.decorators import lazifyTestClass
from hyperspy.signals import BaseVectorSignal, BaseSignal


class TestVectorSignal:
    @pytest.fixture
    def two_d_vector(self):
        x = np.empty(shape=(4, 3), dtype=object)
        for i in np.ndindex(x.shape):
            x[i] = np.random.random((4, 2))
        s = BaseVectorSignal(x)
        for ax, name in zip(s.axes_manager._axes, "abcd"):
            ax.name = name
        return s

    @pytest.fixture
    def lazy_four_d_vector(self):
        x = da.empty(shape=(4), dtype=object)
        for i in np.ndindex(x.shape):
            x[i] = da.random.random((6, 4))
        s = L(x).T
        s.vector = True
        s.set_signal_type("vector")
        for ax, name in zip(s.axes_manager._axes, "abcd"):
            ax.name = name
        return s

    def test_all_to_vector(self, two_d_vector):
        new = two_d_vector.nav_to_vector(inplace=False)
        assert len(new.axes_manager.signal_axes) ==4
        assert new.data.shape == (1,)

    def test_all_to_vector_axis(self, two_d_vector):
        new = two_d_vector.nav_to_vector(axis=0, inplace=False)
        assert len(new.axes_manager.signal_axes) ==3
        assert new.data.shape == (3,)
        new = two_d_vector.nav_to_vector(axis=1, inplace=False)
        assert len(new.axes_manager.signal_axes) ==3
        assert new.data.shape == (4,)

    def test_to_markers(self,two_d_vector):
        markers = two_d_vector.to_markers()
