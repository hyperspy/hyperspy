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

sqrt2pi = math.sqrt(2*math.pi)

class Gaussian(Component):
    """Normalized gaussian function component
    
    .. math::

        f(x) = \\frac{a}{\sqrt{2\pi c^{2}}}e^{-\\frac{\left(x-b\\right)^{2}}{2c^{2}}}
        
    +------------+-----------+
    | Parameter  | Attribute |
    +------------+-----------+
    +------------+-----------+
    |     a      |     A     |
    +------------+-----------+
    |     b      |  centre   |
    +------------+-----------+
    |     c      |   sigma   |
    +------------+-----------+
                  
    
    """

    def __init__(self, A=1., sigma=1.,centre = 0.):
        Component.__init__(self, ['A','sigma','centre'])
        self.A.value = A
        self.sigma.value = sigma
        self.centre.value = centre

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
        self.centre.grad = self.grad_centre

    def function(self, x) :
        A = self.A.value
        sigma = self.sigma.value
        centre = self.centre.value
        return A * (1 / (sigma * sqrt2pi)) * np.exp(
                                            -(x - centre)**2 / (2 * sigma**2))
    
    def grad_A(self, x):
        return self.function(x) / self.A.value
    
    def grad_sigma(self,x):
        return ((x - self.centre.value)**2 * np.exp(-(x - self.centre.value)**2 
        /(2 * self.sigma.value**2)) * self.A.value) / (sqrt2pi * 
        self.sigma.value**4)-(np.exp(-(x - self.centre.value)**2 / (2 * 
        self.sigma.value**2)) * self.A.value) / (sqrt2pi * self.sigma.value**2)
    
    def grad_centre(self,x):
        return ((x - self.centre.value) * np.exp(-(x - self.centre.value)**2/(2 
        * self.sigma.value**2)) * self.A.value) / (sqrt2pi * 
        self.sigma.value**3)
        
    def estimate_parameters(self, signal, E1, E2, only_current = False):
        """Estimate the gaussian by calculating the momenta.

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
        
        Notes
        -----
        Adapted from http://www.scipy.org/Cookbook/FittingData
        
        Example
        -------
        
        import numpy as np
        from hyperspy.hspy import *
        from hyperspy.signals.spectrum import Spectrum  
        g = components.Gaussian()
        x = np.arange(-10,10, 0.01)
        data = np.zeros((32,32,2000))
        data[:] = g.function(x).reshape((1,1,2000))
        s = Spectrum({'data' : data})
        s.axes_manager.axes[-1].offset = -10
        s.axes_manager.axes[-1].scale = 0.01
        g.estimate_parameters(s, -10,10, False)
            
        """
        axis = signal.axes_manager.signal_axes[0]
        
        energy2index = axis.value2index
        i1 = energy2index(E1)
        i2 = energy2index(E2)
        X = axis.axis[i1:i2]
        if only_current is True:
            data = signal()[i1:i2]
            X_shape = (len(X),)
            i = 0
            center_shape = (1,)
        else:
            # TODO: write the rest of the code to estimate the parameters of 
            # the full dataset
            i = axis.index_in_array 
            data_gi = [slice(None),] * len(signal.data.shape)
            data_gi[axis.index_in_array] = slice(i1,i2)
            data = signal.data[data_gi]
            X_shape = [1,] * len(signal.data.shape)
            X_shape[axis.index_in_array] = data.shape[i]
            center_shape = list(data.shape)
            center_shape[i] = 1
        
        center = np.sum(X.reshape(X_shape) * data, i
            ) / np.sum(data, i)
        
        sigma = np.sqrt(np.abs(np.sum((X.reshape(X_shape) - center.reshape(
        center_shape)) ** 2 * data, i) / np.sum(data, i)))
        height = data.max(i)
        if only_current is True:
            self.centre.value = center
            self.sigma.value = sigma
            self.A.value = height * sigma * sqrt2pi
            return True
        else:
            if self.A.map is None:
                self.create_arrays(signal.axes_manager.navigation_shape)
            self.A.map['values'][:] = height * sigma * sqrt2pi
            self.A.map['is_set'][:] = True
            self.sigma.map['values'][:] = sigma
            self.sigma.map['is_set'][:] = True
            self.centre.map['values'][:] = center
            self.centre.map['is_set'][:] = True
            return True
