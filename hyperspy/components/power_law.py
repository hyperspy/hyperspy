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

import math

import numpy as np

from hyperspy.component import Component

class PowerLaw(Component):
    """Power law component
    
    f(x) = A*(x-x0)^-r
    
    +------------+-----------+
    | Parameter  | Attribute |
    +------------+-----------+
    +------------+-----------+
    |     A      |     A     |
    +------------+-----------+
    |     r      |     r     |
    +------------+-----------+
    |    x0      |  origin   |
    +------------+-----------+

    The left_cutoff parameter can be used to set a lower threshold from which
    the component will return 0. 
    
    
    """

    def __init__(self, A=10e5, r=3.,origin = 0.):
        Component.__init__(self, ('A', 'r', 'origin'))
        self.A.value = A
        self.r.value = r
        self.origin.value = origin
        self.origin.free = False
        self.left_cutoff = 0.

        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None
        self.r.bmin = 1.
        self.r.bmax = 5.

        self.isbackground = True
        self.convolved = False

    def function(self, x):
        return np.where(x > self.left_cutoff, self.A.value * 
        (x - self.origin.value)**(-self.r.value), 0)
    def grad_A(self, x):
        return self.function(x) / self.A.value
    def grad_r(self,x):
        return np.where(x > self.left_cutoff, -self.A.value * 
        np.log(x - self.origin.value) * 
        (x - self.origin.value)**(-self.r.value),0 )
    def grad_origin(self,x):
        return np.where( x > self.left_cutoff , self.r.value * 
        (x - self.origin.value)**(-self.r.value - 1) * self.A.value, 0)
        
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
        if (energy2index(x2) - i1) % 2 == 0:
            i2 = energy2index(x2)
        else :
            i2 = energy2index(x2) - 1
        x2 = axis.axis[i2]
        i3 = (i2+i1) / 2
        E3 = axis.axis[i3]
        if only_current is True:
            dc = signal()
            I1 = axis.scale * np.sum(dc[i1:i3], 0)
            I2 = axis.scale * np.sum(dc[i3:i2],0)
        else:
            dc = signal.data
            gi = [slice(None),] * len(dc.shape)
            gi[axis.index_in_array] = slice(i1,i3)
            I1 = axis.scale * np.sum(dc[gi], axis.index_in_array)
            gi[axis.index_in_array] = slice(i3,i2)
            I2 = axis.scale * np.sum(dc[gi],axis.index_in_array)
        try:
            r = 2*np.log(I1 / I2) / math.log(x2/x1)
            k = 1 - r
            A = k * I2 / (x2**k - E3**k)
        except:
            return False
        if only_current is True:
            self.r.value = r
            self.A.value = A
            return True
        else:
            if self.A.map is None:
                self.create_arrays(signal.axes_manager.navigation_shape)
            self.A.map['values'][:] = A
            self.A.map['is_set'][:] = True
            self.r.map['values'][:] = r
            self.r.map['is_set'][:] = True
            return True
