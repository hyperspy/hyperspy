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


import numpy as np

from hyperspy.component import Component

class Offset(Component):
    """Component to add a constant value in the y-axis
    
    f(x) = k + x
    
    +------------+-----------+
    | Parameter  | Attribute |
    +------------+-----------+
    +------------+-----------+
    |     k      |  offset   |
    +------------+-----------+

    """

    def __init__( self, offset = 0. ):
        Component.__init__(self, ('offset',))
        self.offset.free = True
        self.offset.value = offset

        self.isbackground = True
        self.convolved = False

        # Gradients
        self.offset.grad = self.grad_offset
        
    def function(self, x):
        return np.ones((len(x))) * self.offset.value
    def grad_offset(self, x):
        return np.ones((len(x)))
        
    def estimate_parameters(self, signal, x1, x2, only_current = False):
        """Estimate the parameters by the two area method

        Parameters
        ----------
        signal : Signal instance
        x1 : float
            Defines the left limit of the spectral range to use for the 
            estimation.
        x2 : float
            Defines the right limit of the spectral range to use for the 
            estimation.
            
        only_current : bool
            If False estimates the parameters for the full dataset.
            
        Returns
        -------
        bool
            
        """
        axis = signal.axes_manager.signal_axes[0]
        energy2index = axis.value2index
        i1 = energy2index(x1)
        i2 = energy2index(x2)
        
        if only_current is True:
            self.offset.value = signal()[i1:i2].mean()
            return True
        else:
            if self.A.map is None:
                self.create_arrays(signal.axes_manager.navigation_shape)
            dc = signal.data
            gi = [slice(None),] * len(dc.shape)
            gi[axis.index_in_array] = slice(i1,i2)
            self.offset.map['values'][:] = dc[gi].mean(axis.index_in_array)
            self.offset.map['is_set'][:] = True
            return True
