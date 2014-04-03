# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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


import numpy as np

from hyperspy.component import Component


class DoubleOffset(Component):

    """
    Given an array of the same shape as Spectrum energy_axis, returns it as
    a component that can be added to a model.
    """

    def __init__(self):
        Component.__init__(self, ('offset', 'step'))
        self.isbackground = True
        self.convolved = False

        # Options
        self.interfase = 0
        # Gradients
        self.offset.grad = self.grad_offset
        self.step.grad = self.grad_step

    def function(self, x):

        return np.where(x < self.interfase, self.offset.value + x * 0,
                        self.offset.value + self.step.value + x * 0)

    def grad_offset(self, x):
        return np.ones((len(x)))

    def grad_step(self, x):
        np.where(x < self.interfase, x * 0, 1 + x * 0)
