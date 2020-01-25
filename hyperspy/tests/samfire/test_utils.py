# Copyright 2007-2016 The HyperSpy developers
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


from hyperspy.samfire_utils.strategy import nearest_indices


class TestSamfireUtils:

    def setup_method(self, method):
        self.shape = (10, 10)
        self.radii = (1, 2)

    def test_nearest_indices_ind1(self):
        par, _ = nearest_indices(self.shape, (0, 0), self.radii)
        par_ans = (slice(0, 2, None), slice(0, 3, None))
        assert par == par_ans

    def test_nearest_indices_ind2(self):
        par, _ = nearest_indices(self.shape, (0, 1), self.radii)
        par_ans = (slice(0, 2, None), slice(0, 4, None))
        assert par == par_ans

    def test_nearest_indices_ind3(self):
        par, _ = nearest_indices(self.shape, (0, 2), self.radii)
        par_ans = (slice(0, 2, None), slice(0, 5, None))
        assert par == par_ans

    def test_nearest_indices_ind4(self):
        par, _ = nearest_indices(self.shape, (1, 2), self.radii)
        par_ans = (slice(0, 3, None), slice(0, 5, None))
        assert par == par_ans

    def test_nearest_indices_ind5(self):
        par, _ = nearest_indices(self.shape, (2, 3), self.radii)
        par_ans = (slice(1, 4, None), slice(1, 6, None))
        assert par == par_ans

    def test_nearest_indices_cent1(self):
        _, c = nearest_indices(self.shape, (0, 0), self.radii)
        c_ans = (0, 0)
        assert c == c_ans

    def test_nearest_indices_cent2(self):
        _, c = nearest_indices(self.shape, (0, 1), self.radii)
        c_ans = (0, 1)
        assert c == c_ans

    def test_nearest_indices_cent3(self):
        _, c = nearest_indices(self.shape, (0, 2), self.radii)
        c_ans = (0, 2)
        assert c == c_ans

    def test_nearest_indices_cent4(self):
        _, c = nearest_indices(self.shape, (3, 2), self.radii)
        c_ans = (1, 2)
        assert c == c_ans
