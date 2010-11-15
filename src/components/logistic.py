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


import numpy as np

from ..component import Component

class Logistic(Component):
    '''
    '''

    def __init__(self):
        # Define the parameters
        Component.__init__(self, ('a', 'b', 'c', 'origin'))        
        # Define the name of the component
        self.name = 'Logistic'
        self.a.grad = self.grad_a
        self.b.grad = self.grad_b
        self.c.grad = self.grad_c
        self.origin.grad = self.grad_origin

    def function(self, x):
        '''
        This functions it too complicated to explain
        '''
        a = self.a.value
        b = self.b.value
        c = self.c.value
        origin = self.origin.value
        return a/(1+b*np.exp(-c*(x-origin)))
    
    def grad_a(self, x):
        '''
        Returns d(function)/d(parameter_1)
        '''
        a = self.a.value
        b = self.b.value
        c = self.c.value
        origin = self.origin.value
        
        return 1/(1+b*np.exp(-c*(x-origin)))
    
    def grad_b(self, x):
        '''
        Returns d(function)/d(parameter_1)
        '''
        a = self.a.value
        b = self.b.value
        c = self.c.value
        origin = self.origin.value
        
        return -(a*np.exp(-c*(x-origin)))/(b*np.exp(-c*(x-origin))+1)**2
    
    def grad_c(self, x):
        '''
        Returns d(function)/d(parameter_1)
        '''
        a = self.a.value
        b = self.b.value
        c = self.c.value
        origin = self.origin.value
        
        return -(a*b*(origin-x)*np.exp(-c*(x-origin))) / \
    (b*np.exp(-c*(x-origin))+1)**2
    
    def grad_origin(self, x):
        '''
        Returns d(function)/d(parameter_1)
        '''
        a = self.a.value
        b = self.b.value
        c = self.c.value
        origin = self.origin.value
        
        return -(a*b*c*np.exp(-c*(x-origin)))/(b*np.exp(-c*(x-origin))+1)**2



