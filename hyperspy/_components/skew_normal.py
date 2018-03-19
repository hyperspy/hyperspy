# -*- coding: utf-8 -*-
# Copyright 2007-2018 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import math
import numpy as np
from scipy.special import erf
from hyperspy.component import Component

class SkewNormal(Component):

    """Skew normal distribution component.
    
    Asymmetric peak shape based on a normal distribution.
    For definition see https://en.wikipedia.org/wiki/Skew_normal_distribution
    See also http://azzalini.stat.unipd.it/SN/
    
    .. math::

        f(x) = 2 A \\phi(x) \\Phi(x)
        phi(x) = \\frac{1}{\\sqrt{2\\pi}}\\mathrm{exp}{\\left[-\\frac{t(x)^2}{2}\\right]} 
        Phi(x) = \\frac{1}{2}\\left[1 + \\mathrm{erf}{\\left(\\frac{shape t(x)}{\\sqrt{2}}\\right)\\right] 
        t(x) = \\frac{x-x0}{scale}
    
    Parameters
    -----------
        A : float
            Height parameter of the peak.
        x0 : float
            Location of the peak position.
        scale : float
            Width (sigma) parameter.
        shape: float 
            Skewness (asymmetry) parameter. For scale=0, the normal distribution (Gaussian) is obtained. The distribution is right skewed (longer tail to the right) if scale>0 and is left skewed if scale<0.
    """

    def __init__(self, x0=0, A=1, scale=1, shape=0):
        # Define the parameters
        Component.__init__(self, ('x0', 'A', 'scale', 'shape'))

        # Set the initial values
        self.x0.value = x0
        self.A.value = A
        self.scale.value = scale
        self.shape.value = shape

        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None
        
        # Gradients
        self.x0.grad = self.grad_x0
        self.A.grad = self.grad_A
        self.scale.grad = self.grad_scale
        self.shape.grad = self.grad_shape
        self._position = self.x0

    # Define the function as a function of the already defined parameters, x
    # being the independent variable value
    def function(self, x):
        x0 = self.x0.value
        A = self.A.value
        scale = self.scale.value
        shape = self.shape.value
        t = (x - x0) / scale
        normpdf = np.exp(- t ** 2 / 2) / math.sqrt(2 * np.pi)
        normcdf = (1 + erf(shape * t / math.sqrt(2))) / 2
        return 2 * A * normpdf * normcdf
    
    # Gradient functions
    def grad_x0(self, x):
        x0 = self.x0.value
        A = self.A.value
        scale = self.scale.value
        shape = self.shape.value
        t = (x - x0) / scale
        normpdf = np.exp(- t ** 2 / 2) / math.sqrt(2 * np.pi)
        normcdf = (1 + erf(shape * t / math.sqrt(2))) / 2
        derpdf = -math.sqrt(2)*(-2*x + 2*x0)*np.exp(-(x - x0)**2/(2*scale**2))/(4*math.sqrt(pi)*scale**2)
        dercdf = -math.sqrt(2)*shape*np.exp(-shape**2*(x - x0)**2/(2*scale**2))/(2*math.sqrt(pi)*scale)
        return 2 * A * (normpdf * dercdf + derpdf * normcdf)
    
    def grad_A(self, x):
        x0 = self.x0.value
        scale = self.scale.value
        shape = self.shape.value
        t = (x - x0) / scale
        normpdf = np.exp(- t ** 2 / 2) / math.sqrt(2 * np.pi)
        normcdf = (1 + erf(shape * t / math.sqrt(2))) / 2
        return 2 * normpdf * normcdf
    
    def grad_scale(self, x):
        x0 = self.x0.value
        A = self.A.value
        scale = self.scale.value
        shape = self.shape.value
        t = (x - x0) / scale
        normpdf = np.exp(- t ** 2 / 2) / math.sqrt(2 * np.pi)
        normcdf = (1 + erf(shape * t / math.sqrt(2))) / 2
        derpdf = math.sqrt(2)*(x - x0)**2*np.exp(-(x - x0)**2/(2*scale**2))/(2*math.sqrt(pi)*scale**3)
        dercdf = -math.sqrt(2)*shape*(x - x0)*np.exp(-shape**2*(x - x0)**2/(2*scale**2))/(2*math.sqrt(pi)*scale**2)
        return 2 * A * (normpdf * dercdf + derpdf * normcdf)
    
    def grad_shape(self, x):
        x0 = self.x0.value
        A = self.A.value
        scale = self.scale.value
        shape = self.shape.value
        t = (x - x0) / scale
        normpdf = np.exp(- t ** 2 / 2) / math.sqrt(2 * np.pi)
        dercdf = math.sqrt(2)*(x - x0)*np.exp(-shape**2*(x - x0)**2/(2*scale**2))/(2*math.sqrt(pi)*scale)
        return 2 * A * normpdf * dercdf
