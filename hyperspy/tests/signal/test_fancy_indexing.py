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
from nose.tools import (
    assert_true,
    assert_equal,
    assert_not_equal,
    raises)

from hyperspy.signal import Signal

class Test1D:
    def setUp(self):
        self.signal = Signal({'data' : np.arange(10)})
        self.data = self.signal.data.copy()
        
    def test_std_slice(self):
        s = self.signal[1:-1]
        d = self.data[1:-1]
        assert_true((s.data==d).all())
        assert_equal(s.axes_manager.axes[0].offset, 1)
        assert_equal(s.axes_manager.axes[0].scale,
                     self.signal.axes_manager.axes[0].scale)

    def test_reverse_slice(self):
        s = self.signal[-1:1:-1]
        d = self.data[-1:1:-1]
        assert_true((s.data==d).all())
        assert_equal(s.axes_manager.axes[0].offset, 9)
        assert_equal(s.axes_manager.axes[0].scale,
                     self.signal.axes_manager.axes[0].scale * -1)
                     
    def test_step2_slice(self):
        s = self.signal[1:-1:2]
        d = self.data[1:-1:2]
        assert_true((s.data==d).all())
        assert_equal(s.axes_manager.axes[0].offset, 1)
        assert_equal(np.sign(s.axes_manager.axes[0].scale),
                     np.sign(self.signal.axes_manager.axes[0].scale))
        assert_equal(s.axes_manager.axes[0].scale,
                     self.signal.axes_manager.axes[0].scale*2.)

    def test_index(self):
        s = self.signal[3]
        
    def test_signal_indexer_slice(self):
        s = self.signal.signal_indexer[1:-1]
        d = self.data[1:-1]
        assert_true((s.data==d).all())
        assert_equal(s.axes_manager.axes[0].offset, 1)
        assert_equal(s.axes_manager.axes[0].scale,
                     self.signal.axes_manager.axes[0].scale)

    def test_signal_indexer_reverse_slice(self):
        s = self.signal.signal_indexer[-1:1:-1]
        d = self.data[-1:1:-1]
        assert_true((s.data==d).all())
        assert_equal(s.axes_manager.axes[0].offset, 9)
        assert_equal(s.axes_manager.axes[0].scale,
                     self.signal.axes_manager.axes[0].scale * -1)
                     
    def test_signal_indexer_step2_slice(self):
        s = self.signal.signal_indexer[1:-1:2]
        d = self.data[1:-1:2]
        assert_true((s.data==d).all())
        assert_equal(s.axes_manager.axes[0].offset, 1)
        assert_equal(np.sign(s.axes_manager.axes[0].scale),
                     np.sign(self.signal.axes_manager.axes[0].scale))
        assert_equal(s.axes_manager.axes[0].scale,
                     self.signal.axes_manager.axes[0].scale*2.)

    def test_signal_indexer_index(self):
        s = self.signal.signal_indexer[3]
        assert_equal(s.data, 3)
     
    @raises(IndexError)    
    def test_navigation_indexer_navdim0(self):
        s = self.signal.navigation_indexer[3]
        
        
class Test3D_SignalDim0:
    def setUp(self):
        self.signal = Signal({'data' : np.arange(24).reshape((2,3,4))})
        self.data = self.signal.data.copy()
        self.signal.axes_manager.axes[2].navigate = True
        
    def test_signal_dim0(self):
        s = self.signal
        assert((s[:].data == s.data).all())
        
    def test_signal_indexer_signal_dim0(self):
        s = self.signal
        assert((s.signal_indexer[:].data == s.data).all())
        
    def test_navigation_indexer_signal_dim0(self):
        s = self.signal
        assert((s.navigation_indexer[:].data == s.data).all())
        
class Test3D_Navigate_0_and_1:
    def setUp(self):
        self.signal = Signal({'data' : np.arange(24).reshape((2,3,4))})
        self.data = self.signal.data.copy()
        self.signal.axes_manager.axes[0].navigate = True
        self.signal.axes_manager.axes[1].navigate = True
        self.signal.axes_manager.axes[2].navigate = False
    
    def test_1px_slice(self):
        s = self.signal[1:2]
        d = self.data[:,1:2]
        assert_true((s.data==d).all())
        assert_equal(s.axes_manager.axes[1].offset, 1)
        assert_equal(s.axes_manager.axes[1].size, 1)
        assert_equal(s.axes_manager.axes[1].scale,
                     self.signal.axes_manager.axes[1].scale)
                     
    def test_1px_navigation_indexer_slice(self):
        s = self.signal.navigation_indexer[1:2]
        d = self.data[:,1:2]
        assert_true((s.data==d).all())
        assert_equal(s.axes_manager.axes[1].offset, 1)
        assert_equal(s.axes_manager.axes[1].size, 1)
        assert_equal(s.axes_manager.axes[1].scale,
                     self.signal.axes_manager.axes[1].scale)
                     
    def test_1px_signal_indexer_slice(self):
        s = self.signal.signal_indexer[1:2]
        d = self.data[:,:,1:2]
        assert_true((s.data==d).all())
        assert_equal(s.axes_manager.signal_axes[0].offset, 1)
        assert_equal(s.axes_manager.signal_axes[0].size, 1)
        assert_equal(s.axes_manager.signal_axes[0].scale,
                     self.signal.axes_manager.signal_axes[0].scale)
                     
class Test3D_Navigate_1:
    def setUp(self):
        self.signal = Signal({'data' : np.arange(24).reshape((2,3,4))})
        self.data = self.signal.data.copy()
        self.signal.axes_manager.axes[0].navigate = False
        self.signal.axes_manager.axes[1].navigate = True
        self.signal.axes_manager.axes[2].navigate = False
    
    def test_1px_slice(self):
        s = self.signal[1:2]
        d = self.data[:,1:2]
        assert_true((s.data==d).all())
        assert_equal(s.axes_manager.axes[1].offset, 1)
        assert_equal(s.axes_manager.axes[1].size, 1)
        assert_equal(s.axes_manager.axes[1].scale,
                     self.signal.axes_manager.axes[1].scale)
                     
    def test_1px_navigation_indexer_slice(self):
        s = self.signal.navigation_indexer[1:2]
        d = self.data[:,1:2]
        assert_true((s.data==d).all())
        assert_equal(s.axes_manager.axes[1].offset, 1)
        assert_equal(s.axes_manager.axes[1].size, 1)
        assert_equal(s.axes_manager.axes[1].scale,
                     self.signal.axes_manager.axes[1].scale)
                     
    def test_1px_signal_indexer_slice(self):
        s = self.signal.signal_indexer[1:2]
        d = self.data[:,:,1:2]
        assert_true((s.data==d).all())
        assert_equal(s.axes_manager.signal_axes[1].offset, 1)
        assert_equal(s.axes_manager.signal_axes[1].size, 1)
        assert_equal(s.axes_manager.signal_axes[1].scale,
                     self.signal.axes_manager.signal_axes[1].scale)
                     
class TestFloatArguments:
    def setUp(self):
        self.signal = Signal({'data' : np.arange(10)})
        self.signal.axes_manager[0].scale = 0.5
        self.signal.axes_manager[0].offset = 0.25
        self.data = self.signal.data.copy()
        
    def test_float_start(self):
        s = self.signal[0.75:]
        d = self.data[1:]
        assert_true((s.data==d).all())
        assert_equal(s.axes_manager.axes[0].offset, 0.75)
        assert_equal(s.axes_manager.axes[0].scale,
                     self.signal.axes_manager.axes[0].scale)

    def test_float_end(self):
        s = self.signal[:4.75]
        d = self.data[:-1]
        assert_true((s.data==d).all())
        assert_equal(s.axes_manager.axes[0].offset, 0.25)
        assert_equal(s.axes_manager.axes[0].scale,
                     self.signal.axes_manager.axes[0].scale)
                     
    def test_float_both(self):
        s = self.signal[0.75:4.75]
        d = self.data[1:-1]
        assert_true((s.data==d).all())
        assert_equal(s.axes_manager.axes[0].offset, 0.75)
        assert_equal(s.axes_manager.axes[0].scale,
                     self.signal.axes_manager.axes[0].scale)

    

