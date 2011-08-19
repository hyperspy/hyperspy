# -*- coding: utf-8 -*-
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

from hyperspy.signal import Signal
import numpy as np

def three_d_image():
    s = Signal({'data' : np.random.random((16,32,32))})
    s.axes_manager.axes[1].slice_bool = True
    s.plot()

def four_d_image():
    s = Signal({'data' : np.random.random((16,16,32,32))})
    s.axes_manager.axes[2].slice_bool = True
    s.plot()

def line_spectrum():
    s = Signal({'data' : np.random.random((100,1024))})
    s.plot()

def spectrum_image():
    s = Signal({'data' : np.random.random((64,64,1024))})
    s.plot()


