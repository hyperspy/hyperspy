# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
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
from scipy.special import erf

from ..component import Component

sqrt2pi = np.sqrt(2*np.pi)

class Erf(Component):
    '''
    '''

    def __init__(self):
        Component.__init__(self, ['A','sigma','origin'])        
        self.name = 'Error Function'
                
        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None
        
        self.sigma.bmin = None
        self.sigma.bmax = None


        self.isbackground = False
        self.convolved = True
        
        # Gradients
        self.A.grad = self.grad_A
        self.sigma.grad = self.grad_sigma
        self.origin.grad = self.grad_origin

    def function(self, x):
        A = self.A.value
        sigma = self.sigma.value
        origin = self.origin.value
        return A*erf((x-origin)/math.sqrt(2)/sigma)/2

    def grad_A(self, x):
        A = self.A.value
        sigma = self.sigma.value
        origin = self.origin.value
        return erf((x-origin)/math.sqrt(2)/sigma)/2
    def grad_sigma(self,x):
        A = self.A.value
        sigma = self.sigma.value
        origin = self.origin.value
        return ((origin/(math.sqrt(2)*sigma**2)-x/(math.sqrt(2)*sigma**2))*np.exp(-(x/(math.sqrt(2)*sigma)-origin/(math.sqrt(2)*sigma))**2)*A)/math.sqrt(math.pi)
    def grad_origin(self,x):
        A = self.A.value
        sigma = self.sigma.value
        origin = self.origin.value
        return -(np.exp(-(x/(math.sqrt(2)*sigma)-origin/(math.sqrt(2)*sigma))**2)*A)/(math.sqrt(2)*math.sqrt(math.pi)*sigma)
