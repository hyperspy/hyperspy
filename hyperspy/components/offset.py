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

class Offset(Component):
    """
    Given an array of the same shape as Spectrum energy_axis, returns it as
    a component that can be added to a model.
    """

    def __init__( self, offset = 0. ):
        Component.__init__(self, ('offset',))
        self.name = 'offset'
        self.offset.free = True
        self.offset.value = offset

        self.isbackground = True
        self.convolved = False

        # Gradients
        self.offset.grad = self.grad_offset
        
    def function(self, x):
        return np.ones((len(x))) * self.offset.value
    def grad_offset(self, x):
        return np.ones((len(x)))
