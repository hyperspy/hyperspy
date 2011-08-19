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

class Lorentzian(Component):
    """
    """

    def __init__(self, A=1., gamma=1.,origin = 0.):
        Component.__init__(self, ('A', 'gamma', 'origin'))
        
        self.A.value = A
        self.gamma.value = gamma
        self.origin.value = origin
        
        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None
        self.gamma.bmin = None
        self.gamma.bmax = None

        self.isbackground = False
        self.convolved = True
        
        # Gradients
        self.A.grad = self.grad_A
        self.gamma.grad = self.grad_gamma
        self.origin.grad = self.grad_origin

    def __repr__(self):
        return 'Lorentzian'

    def function( self, x ) :
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the background model, returns the background
        model for the current parameters.
        """
        return (self.A.value / np.pi) * (self.gamma.value / 
        ((x - self.origin.value)**2 + self.gamma.value**2))
    def grad_A(self, x):
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the gradient of the background model,
        returns the gradient of parameter A for the current value of the
        parameters.
        """
        return self.function(x) / self.A.value
    def grad_gamma(self,x):
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the gradient of the background model,
        returns the gradient of parameter gamma for the current value of
        the parameters.
        """
        return self.A.value / (np.pi * (self.gamma.value**2 + 
        (x - self.origin.value)**2)) - ((2 * self.A.value * self.gamma.value**2) 
        / (np.pi*(self.gamma.value**2+(x-self.origin.value)**2)**2))
    def grad_origin(self,x):
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the gradient of the background model,
        returns the gradient of parameter origin for the current value of
        the parameters.
        """
        return (2 * (x - self.origin.value) * self.A.value * self.gamma.value
        )/(np.pi * (self.gamma.value**2 + (x - self.origin.value)**2)**2)
        
        
