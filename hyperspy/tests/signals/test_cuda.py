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

import hyperspy.api as hs

try:
    import cupy as cp
except ImportError:
    pytest.skip("cupy is required", allow_module_level=True)


class TestCupy:
    def setup_method(self, method):
        N = 100
        ndim = 3
        data = cp.arange(N**3).reshape([N]*ndim)
        s = hs.signals.Signal1D(data)
        self.s = s

    def test_call_signal(self):
        s = self.s
        np.testing.assert_allclose(s(), np.arange(100))

    def test_roi(self):
        s = self.s
        roi = hs.roi.CircleROI(40, 60, 15)
        sr = roi(s)
        sr.plot()
        assert isinstance(sr.data, cp.ndarray)
        sr0 = cp.asnumpy(sr.inav[0, 0].data)
        np.testing.assert_allclose(np.nan_to_num(sr0), np.zeros_like(sr0))
        assert sr.isig[0].nansum() == 4.360798E08

