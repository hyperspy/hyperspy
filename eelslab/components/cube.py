#!/usr/bin/env python

# -*- coding: utf-8 -*-
# Copyright  2007 Francisco Javier de la Pena
#
# This file is part of EELSLab.
#
# EELSLab is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# EELSLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EELSLab; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA


import numpy as np
from eelslab.component import Component
from .gaussian import Gaussian

class Cube(Component):
    """
    Model for the thickness of a tilted cube
    """

    def __init__( self, sigma = 0 ):
        Component.__init__(self, ['onset', 'p1', 'p2', 'height', 'sigma'])
        self.sigma.value = sigma
        self.gaussian = Gaussian()
        self.gaussian.origin.free, self.gaussian.A.free = False, False
        self.gaussian.sigma.free = True
        self.gaussian.A.value = 1.  
        self.convolved = False
        self.convolution_axis = None
        self.name = 'Cube thickness model'
    
    def slope(self):
        return self.height.value/self.p1.value
    
        
    def function(self, x):
        if self.sigma.value:
            if self.convolution_axis is not None:
                x0,x1 = x[0], x[-1]
                x = self.convolution_axis
                xl = x.tolist()
                i0 = xl.index(x0)
                i1 = xl.index(x1)
                
        self.p1.value = float(self.p1.value)
        self.p2.value = float(self.p2.value)
        self.height.value = float(self.height.value)
        self.onset.value = float(self.onset.value)
        result = \
        np.where((x < self.onset.value), 0, 
        np.where((x < self.onset.value + self.p1.value), 
        (x-float(self.onset.value))*self.slope(), 
        np.where((x < self.onset.value + self.p2.value), 
        self.height.value, 
        np.where(x < (self.onset.value + self.p2.value + self.p1.value), 
        self.slope()*(float(self.p1.value+self.p2.value + self.onset.value) 
        -x), 0))))
        if self.sigma.value:
            self.gaussian.sigma.value = self.sigma.value
            self.gaussian.origin.value = (x[-1] + x[0]) / 2
            g = self.gaussian.function(x)
            g /= g.sum()
            result = np.convolve(g, result, 'same')
            if self.convolution_axis is not None:
                return result[i0:i1+1]
            else:
                return result
        else:
            return result
