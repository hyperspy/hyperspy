# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of Hyperspy.
#
# Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Hyperspy. If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from hyperspy.component import Component

class Arctan(Component):
    """Arctan function components
    
    f(x) = A*arctan{k*(x-x0)}
    
    +------------+-----------+
    | Parameter  | Attribute |
    +------------+-----------+
    +------------+-----------+
    |     A      |     A     |
    +------------+-----------+
    |     k      |     k     |
    +------------+-----------+
    |     x      |     x     |
    +------------+-----------+
    |     x0     |     x0    |
    +------------+-----------+

    """

    def __init__(self, A=1. , k=1. , centre=1., y0=1.):
        Component.__init__(self, ['A', 'k', 'centre', 'y0'])
        self.A.value = A
        self.A.grad = self.grad_A

        self.k.value = k
        self.k.grad = self.grad_k

        self.centre.value = centre
        self.centre.grad = self.grad_centre

        self.y0.value = y0
        self.centre.grad = self.grad_y0

        self.isbackground = False
        self.isconvolved = False

    def function(self,x):
        A = self.A.value
        k = self.k.value
        centre = self.centre.value
        y0 = self.y0.value
        return A*np.arctan(k*(x-centre)) + y0
    
    def grad_A(self,x):
        A = self.A.value
        k = self.k.value
        centre = self.centre.value
        return np.arctan(k*(x-centre))

    def grad_k(self,x):
        A = self.A.value
        k = self.k.value
        centre = self.centre.value
        return A*(x-centre)/(1+(k*(x-centre))**2)

    def grad_centre(self, x):
        A = self.A.value
        k = self.k.value
        centre = self.centre.value
        return -A*k/(1+(k*(x-centre))**2)

    def grad_y0(self, x):
        return 1
