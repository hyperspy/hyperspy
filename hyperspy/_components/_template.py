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

from hyperspy.component import Component

class My_Component(Component):
    """
    """

    def __init__(self, parameter_1 = 1, parameter_2 = 2):
        # Define the parameters
        Component.__init__(self, ('parameter_1', 'parameter_2'))        
        # Define the identification name of the component
                
        # Optionally we can set the initial values
#        self.parameter_1.value = parameter_1
#        self.parameter_1.value = parameter_1
        
        # The units
#        self.parameter_1.units = 'Tesla'
#        self.parameter_2.units = 'Kociak'
        
        # Once defined we can give default values to the attribute is we want
        # For example we fix the attribure_1
#        self.parameter_1.attribute_1.free = False
        # And we set the boundaries
#        self.parameter_1.bmin = 0.
#        self.parameter_1.bmax = None


        
        # Optionally, to boost the optimization speed we can define also define
        # the gradients of the function we the syntax: 
        # self.parameter.grad = function
        
#        self.parameter_1.grad = self.grad_parameter_1
#        self.parameter_2.grad = self.grad_parameter_2
        
    
    # Define the function as a function of the already defined parameters, x 
    # being the independent variable value
    def function(self, x):
        """
        This functions it too complicated to explain
        """
        p1 = self.parameter_1.value
        p2 = self.parameter_2.value
        return p1 + x*p2
    
    # Optionally define the gradients of each parameter
#    def grad_parameter_1(self, x):
#        """
#        Returns d(function)/d(parameter_1)
#        """
#        return 0
#    def grad_parameter_2(self, x):
#        """
#        Returns d(function)/d(parameter_2)
#        """
#        return x


