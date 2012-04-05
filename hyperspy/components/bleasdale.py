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

from hyperspy.component import Component, Parameter

class Bleasdale(Component):
    """Bleasdale function component.
    
    f(x) = (a+b*x)^(-1/c)
    
    Attributes
    ----------
    a : Float
    b : Float
    c : Float
    
    """

    def __init__(self):
        # Define the parameters
        Component.__init__(self, ('a', 'b', 'c'))        
        # Define the name of the component

    def function(self, x):
        """
        """
        a = self.a.value
        b = self.b.value
        c = self.c.value
        return (a+b*x)**(-1./c)
    
    def grad_a(self, x):
        """
        Returns d(function)/d(parameter_1)
        """
        a = self.a.value
        b = self.b.value
        c = self.c.value
        
        return -(b*x+a)**(-1./c-1.)/c
    
    def grad_b(self, x):
        """
        Returns d(function)/d(parameter_1)
        """
        a = self.a.value
        b = self.b.value
        c = self.c.value
        
        return -(x*(b*x+a)**(-1/c-1))/c
    
    def grad_c(self, x):
        """
        Returns d(function)/d(parameter_1)
        """
        a = self.a.value
        b = self.b.value
        c = self.c.value
        return np.log(b*x+a)/(c**2.*(b*x+a)**(1./c))
