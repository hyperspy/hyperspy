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

class PowerLaw(Component):
    """
    """

    def __init__(self, A=10e5, r=3.,origin = 0.):
        Component.__init__(self, ('A', 'r', 'origin'))
        self.name = 'Power Law'
        self.A.value = A
        self.r.value = r
        self.origin.value = origin
        self.origin.free = False
        self.left_cutoff = 20.

        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None
        self.r.bmin = 1.
        self.r.bmax = 5.

        self.isbackground = True
        self.convolved = False

    def function(self, x):
        return np.where(x > self.left_cutoff, self.A.value * 
        (x - self.origin.value)**(-self.r.value), 0)
    def grad_A(self, x):
        return self.function(x) / self.A.value
    def grad_r(self,x):
        return np.where(x > self.left_cutoff, -self.A.value * 
        np.log(x - self.origin.value) * 
        (x - self.origin.value)**(-self.r.value),0 )
    def grad_origin(self,x):
        return np.where( x > self.left_cutoff , self.r.value * 
        (x - self.origin.value)**(-self.r.value - 1) * self.A.value, 0)

