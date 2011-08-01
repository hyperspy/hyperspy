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

from eelslab.component import Component
from .gaussian import Gaussian

sqrt2pi = np.sqrt(2*np.pi)

class SEE(Component):
    """
    """

    def __init__(self, A=1., Phi=1.,B = 0., sigma = 0):
        Component.__init__(self, ('A', 'Phi', 'B', 'sigma'))
        self.name = 'SEE'
        self.A.value, self.Phi.value, self.B.value,self.sigma.value = \
        A, Phi, B, sigma
        
        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None

        self.convolved = True

        # Gradients
        self.A.grad = self.grad_A
        self.Phi.grad = self.grad_Phi
        self.B.grad = self.grad_B
        self.sigma.grad = self.grad_sigma
   
        # Resolution functions
        self.gaussian = Gaussian()
        self.gaussian.origin.free, self.gaussian.A.free = False, False
        self.gaussian.sigma.free = True
        self.gaussian.A.value = 1.

    def __repr__(self):
        return 'SEE'

    def function(self, x):
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the background model, returns the background
        model for the current parameters.
        """
        if self.sigma.value:
            self.gaussian.sigma.value = self.sigma.value
            self.gaussian.origin.value = (x[-1] + x[0]) / 2
            return np.convolve(self.gaussian.function(x), 
        np.where(x > self.Phi.value, self.A.value * \
        (x-self.Phi.value) / (x - self.Phi.value + self.B.value)**4, 0), 'same')
        else:
            return np.where(x > self.Phi.value, self.A.value * \
        (x-self.Phi.value) / (x - self.Phi.value + self.B.value)**4, 0)

    def grad_A(self, x):
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the gradient of the background model,
        returns the gradient of parameter A for the current value of the
        parameters.
        """
        if self.sigma.value:
            self.gaussian.sigma.value = self.sigma.value
            self.gaussian.origin.value = (x[-1] + x[0]) / 2
            return np.convolve(self.gaussian.function(x), \
        np.where(x > self.Phi.value,(x-self.Phi.value) / \
    (x - self.Phi.value + self.B.value)**4, 0), 'same')
        else:
            return np.where(x > self.Phi.value,(x-self.Phi.value) / \
    (x - self.Phi.value + self.B.value)**4, 0)
    def grad_sigma(self,x):
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the gradient of the background model,
        returns the gradient of parameter sigma for the current value of
        the parameters.
        """
        self.gaussian.sigma.value = self.sigma.value
        self.gaussian.origin.value = (x[-1] + x[0]) / 2
        return np.convolve(self.gaussian.grad_sigma(x), 
        np.where(x > self.Phi.value, self.A.value * \
        (x-self.Phi.value) / (x - self.Phi.value + self.B.value)**4, 0), 'same')
    def grad_Phi(self,x):
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the gradient of the background model,
        returns the gradient of parameter origin for the current value of
        the parameters.
        """
        if self.sigma.value:
            self.gaussian.sigma.value = self.sigma.value
            self.gaussian.origin.value = (x[-1] + x[0]) / 2
            return np.convolve(self.gaussian.function(x), 
            np.where(x > self.Phi.value, 
        (4*(x-self.Phi.value)*self.A.value)/(self.B.value + \
    x-self.Phi.value)**5-self.A.value/(self.B.value+x-self.Phi.value)**4, 0), 
    'same')
        else:
            return np.where(x > self.Phi.value,(4*(x-self.Phi.value)*self.A.value)/(self.B.value + \
    x-self.Phi.value)**5-self.A.value/(self.B.value+x-self.Phi.value)**4, 0)
    
    def grad_B(self,x):
        if self.sigma.value:
            self.gaussian.sigma.value = self.sigma.value
            self.gaussian.origin.value = (x[-1] + x[0]) / 2
            return np.convolve(self.gaussian.function(x),
            np.where(x > self.Phi.value, 
            -(4 * (x - self.Phi.value) * self.A.value) / \
            (self.B.value+x-self.Phi.value)**5, 0),'same')
        else:
            return np.where(x > self.Phi.value, 
        -(4 * (x - self.Phi.value) * self.A.value) / \
        (self.B.value+x-self.Phi.value)**5, 0)
