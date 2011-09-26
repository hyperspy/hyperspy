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

class Line(Component):
    """
    Line component defined by the a and b parameters: f(x) = a + b*x
    """

    def __init__( self, a = 0, b = 1 ):
        Component.__init__(self, ['a','b'])
        self.name = 'Line'
        self.a.free, self.b.free = True, True
        self.a.value, self.b.value = a, b

        self.isbackground = True
        self.convolved = False
        self.a.grad = self.grad_a
        self.b.grad = self.grad_b
        self.start_from = None
        
    def function(self, x):
        if self.start_from is None:
            return self.a.value + self.b.value*x
        else:
            return np.where(x > self.start_from, self.a.value + self.b.value*x, 0)
    def grad_a(self, x):
        return np.ones((len(x)))
    def grad_b(self, x):
        if self.start_from is None:
            return np.where(x > self.start_from, x, 0)
    
