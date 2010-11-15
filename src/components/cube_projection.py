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

import math

import numpy as np
from ..component import Component
from .gaussian import Gaussian
from .. import spectrum

r2 = math.sqrt(2)

class CubeProjection(Component):
    '''
    Model for the thickness of a tilted cube
    '''

    def __init__( self, sigma = 0 ):
        Component.__init__(self, ['edge_length', 'center', 'angle', 'mfp', 
        'sigma'])
        self.sigma.value = sigma
        self.gaussian = Gaussian()
        self.gaussian.origin.free, self.gaussian.A.free = False, False
        self.gaussian.sigma.free = False
        self.gaussian.A.value = 1.
        self.convolved = False
        self.convolution_axis = None
        self.name = 'Cube projection'
        
    def set_convolution_axis(self, axis):
        self.convolution_axis = spectrum.generate_axis(
        axis[0],axis[1]-axis[0], 3*len(axis), len(axis))
        self._i0 = len(axis)
        self._i1 = 2*len(axis)
        
    def function(self, x):
        if self.sigma.value != 0:
            if self.convolution_axis is None:
                self.set_convolution_axis(x)
            x = self.convolution_axis
        a = self.edge_length.value
        x0 = self.center.value
        cos = math.cos(self.angle.value)
        sin = math.sin(self.angle.value)
        p0 = float(x0 - a*cos/r2)
        p1 = float(x0 - a*sin/r2)
        p2 = float(x0 + a*sin/r2)
        p3 = float(x0 + a*cos/r2)
        h = float(2*a/(cos + sin))/r2/self.mfp.value
        s = float(1/(p1-p0))
        result = \
        np.where((x < p0), 0, 
        np.where((x < p1), (x-p0)*s, 
        np.where((x < p2), 1, 
        np.where(x < p3, s*(p3-x), 
        0))))*h
        if self.sigma.value:
            self.gaussian.sigma.value = self.sigma.value
            self.gaussian.origin.value = (x[-1] + x[0]) / 2.
            g = self.gaussian.function(x)
            g /= g.sum()
            result = np.convolve(g, result, 'same')
            if self.convolution_axis is not None:
                return result[self._i0:self._i1]
            else:
                return result
        else:
            return result
