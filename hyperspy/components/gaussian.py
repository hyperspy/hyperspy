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

sqrt2pi = np.sqrt(2*np.pi)

class Gaussian(Component):
    """
    """

    def __init__(self, A=1., sigma=1.,origin = 0.):
        Component.__init__(self, ['A','sigma','origin'])
        self.A.value = A
        self.sigma.value = sigma
        self.origin.value = origin

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
        self.name = 'Normalized Gaussian'

    def function(self, x) :
        a0 = self.A.value
        a1 = self.origin.value
        a2 = self.sigma.value
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the background model, returns the background
        model for the current parameters.
        """
        return self.A.value * (1 / (self.sigma.value * sqrt2pi)) * np.exp(
        -(x-self.origin.value)**2 / (2 * self.sigma.value**2))
    
    def grad_A(self, x):
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the gradient of the background model,
        returns the gradient of parameter A for the current value of the
        parameters.
        """
        return self.function(x) / self.A.value
    
    def grad_sigma(self,x):
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the gradient of the background model,
        returns the gradient of parameter sigma for the current value of
        the parameters.
        """
        return ((x - self.origin.value)**2 * np.exp(-(x - self.origin.value)**2 
        /(2 * self.sigma.value**2)) * self.A.value) / (sqrt2pi * 
        self.sigma.value**4)-(np.exp(-(x - self.origin.value)**2 / (2 * 
        self.sigma.value**2)) * self.A.value) / (sqrt2pi * self.sigma.value**2)
    
    def grad_origin(self,x):
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the gradient of the background model,
        returns the gradient of parameter origin for the current value of
        the parameters.
        """
        return ((x - self.origin.value) * np.exp(-(x - self.origin.value)**2/(2 
        * self.sigma.value**2)) * self.A.value) / (sqrt2pi * 
        self.sigma.value**3)
