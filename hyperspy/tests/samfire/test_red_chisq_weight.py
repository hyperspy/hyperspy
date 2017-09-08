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


from hyperspy.samfire_utils.weights.red_chisq import ReducedChiSquaredWeight
from hyperspy.misc.utils import DictionaryTreeBrowser


class Test_Red_chisq_weight:

    def setup_method(self, method):
        self.w = ReducedChiSquaredWeight()
        artificial_model = DictionaryTreeBrowser()
        artificial_model.add_node('red_chisq.data')
        artificial_model.red_chisq.data = np.arange(35).reshape((5, 7))
        self.w.model = artificial_model

    def test_function(self):
        w = self.w
        ind = (2, 3)
        assert w.function(ind) == 16

    def test_map_noslice(self):
        w = self.w
        mask = np.ones((5, 7), dtype=bool)
        mask[0, 0] = False
        ans = w.map(mask)
        assert np.all(w.model.red_chisq.data[mask] - 1 == ans[mask])
        assert np.isnan(ans[0, 0])
