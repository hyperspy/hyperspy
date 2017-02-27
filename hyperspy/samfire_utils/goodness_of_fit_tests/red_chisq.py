# -*- coding: utf-8 -*-
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

from hyperspy.samfire_utils.goodness_of_fit_tests.test_general import goodness_test


class red_chisq_test(goodness_test):

    def __init__(self, tolerance):
        self.name = 'Reduced chi-squared test'
        self.expected = 1.0
        self.tolerance = tolerance

    def test(self, model, ind):
        return np.abs(
            model.red_chisq.data[ind] - self.expected) < self.tolerance

    def map(self, model, mask):
        rc = model.red_chisq.data
        rc = np.where(np.isnan(rc), -np.inf, rc)
        ans = np.abs(rc - self.expected) < self.tolerance
        return np.logical_and(mask, ans)
