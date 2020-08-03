# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

from hyperspy.drawing.utils import contrast_stretching


class TestImageStretching:

    def setup_method(self, method):
        self.data = np.arange(11).astype("float")
        self.data[-1] = np.nan


    @pytest.mark.parametrize("value", (("0.5th", "99.5th"), (0.045, 8.955)))
    def test_no_nans(self, value):
        data = self.data[:-1]
        bounds = contrast_stretching(data, value[0], value[1])
        assert bounds == (
            np.percentile(data, 0.5),
            np.percentile(data, 99.5))

    @pytest.mark.parametrize("value", (("0.5th", "99.5th"), (0.045, 8.955)))
    def test_nans(self, value):
        data = self.data[:-1]
        bounds = contrast_stretching(self.data, value[0], value[1])
        assert bounds == (
            np.percentile(data, 0.5),
            np.percentile(data, 99.5))

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            contrast_stretching(self.data, "-0.5th", "100.5th")