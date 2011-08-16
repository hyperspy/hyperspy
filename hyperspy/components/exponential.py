# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
#
# This file is part of Hyperspy.
#
# Hyperspy is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Hyperspy; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301
# USA

import numpy as np

from hyperspy.component import Component


class Exponential(Component):
    """
    """

    def __init__(self):
        Component.__init__(self, ['A', 'tau'])
        self.isbackground = False
        self.name = 'Exponential'
        self.A.grad = self.grad_A
        self.tau.grad = self.grad_tau

    def function( self, x ) :
        """
        """
        A = self.A.value
        tau = self.tau.value
        return A*np.exp(-x/tau)
    
    def grad_A(self,x):
        return self.function(x) / self.A.value
    
    def grad_tau(self,x):
        A = self.A.value
        tau = self.tau.value
        return x*(np.exp(-x/tau))*A/tau**2
        
    



