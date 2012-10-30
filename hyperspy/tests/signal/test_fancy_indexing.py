# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.


import os

import numpy as np

from nose.tools import assert_true, assert_equal, assert_not_equal
from hyperspy.signal import Signal

class Test1D:
    def setUp(self):
        self.signal = Signal({'data' : np.arange(10)})
        
    def test_std_slice(self):
        s = self.signal[1:-1]
        assert_equal(s.data[0], 1)
        assert_equal(s.data[-1], 8)
        assert_equal(s.axes_manager.axes[0].offset, 1)
        assert_equal(s.axes_manager.axes[0].scale,
                     self.signal.axes_manager.axes[0].scale)

    def test_reverse_slice(self):
        s = self.signal[-1:1:-1]
        assert_equal(s.data[0], 9)
        assert_equal(s.data[-1], 2)
        assert_equal(s.axes_manager.axes[0].offset, 9)
        assert_equal(s.axes_manager.axes[0].scale,
                     self.signal.axes_manager.axes[0].scale*-1)
                     
    def test_step2_slice(self):
        s = self.signal[1:-1:2]
        assert_equal(s.data[0], 1)
        assert_equal(s.data[-1], 7)
        assert_equal(s.axes_manager.axes[0].offset, 1)
        assert_equal(np.sign(s.axes_manager.axes[0].scale),
                     np.sign(self.signal.axes_manager.axes[0].scale))
        assert_equal(s.axes_manager.axes[0].scale,
                     self.signal.axes_manager.axes[0].scale*2.)
                     
    def test_index(self):
        s = self.signal[3]
        assert_equal(s.data, 3)
        
        
