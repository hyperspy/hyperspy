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


from hyperspy.samfire_utils.segmenters.histogram import HistogramSegmenter
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.external.astroML.histtools import histogram


def compare_two_value_dicts(ans_r, ans):
    test = True
    for k, v in ans_r.items():
        test = test and k in ans
        if test:
            for p, pv in v.items():
                test = test and p in ans[k]
                if test:
                    if isinstance(pv, tuple):
                        assert np.all(pv[0] == ans[k][p][0])
                        assert np.all(pv[1] == ans[k][p][1])
                    else:
                        assert np.all(pv == ans[k][p])
    return test


class TestHistogramSegmenter:

    def setup_method(self, method):
        self.test_dict = {'one': {'A': np.array([10.])},
                          'two': {'centre': np.array([0., 1.]),
                                  'sigma': np.array([-3., 0., 3., 1., 1.5, 2.,
                                                     3.0, 3.05, 3.1, 3.15, 4.,
                                                     4., 5., 17., 30.])}
                          }

        self.test_database = {'one': {'A': None},
                              'two': {'centre': None, 'sigma': None}}

        self.test_database['one']['A'] = np.histogram(
            self.test_dict['one']['A'],
            10)
        self.test_database['two']['centre'] = np.histogram(
            self.test_dict['two']['centre'],
            10)
        self.test_database['two']['sigma'] = histogram(
            self.test_dict['two']['sigma'],
            'blocks')
        self.s = HistogramSegmenter()

    def test_init(self):
        s = self.s
        assert s.database is None
        assert s._min_points == 4
        assert s.bins == 'freedman'

    def test_most_frequent(self):
        s = self.s
        s.database = self.test_database
        freq = s.most_frequent()
        res = {'one': {'A': np.array([10.05])},
               'two': {'centre': np.array([0.05, 0.95]), 'sigma': np.array([0.75])}}
        assert compare_two_value_dicts(res, freq)

    def test_update(self):
        s = self.s
        s.bins = 'blocks'
        s.update(self.test_dict)
        print('required:')
        print(self.test_database)
        print('--------------------------------------\n calculated:')
        print(s.database)
        assert compare_two_value_dicts(s.database, self.test_database)
