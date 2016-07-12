# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.

import math

import numpy as np

from hyperspy.component import Component


class Arctan(Component):

    """Arctan function component

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

    def __init__(self, A=1., k=1., x0=1., minimum_at_zero=False):
        Component.__init__(self, ['A', 'k', 'x0'])
        self.A.value = A
        self.A.grad = self.grad_A

        self.k.value = k
        self.k.grad = self.grad_k

        self.x0.value = x0
        self.x0.grad = self.grad_x0

        self.minimum_at_zero = minimum_at_zero
        self._whitelist['minimum_at_zero'] = ('init', minimum_at_zero)

        self.isbackground = False
        self.isconvolved = False
        self._position = self.x0

    def function(self, x):
        A = self.A.value
        k = self.k.value
        x0 = self.x0.value
        if self.minimum_at_zero:
            return A * (math.pi / 2 + np.arctan(k * (x - x0)))
        else:
            return A * np.arctan(k * (x - x0))

    def grad_A(self, x):
        k = self.k.value
        x0 = self.x0.value
        if self.minimum_at_zero:
            return offset + np.arctan(k * (x - x0))
        else:
            return np.arctan(k * (x - x0))

    def grad_k(self, x):
        A = self.A.value
        k = self.k.value
        x0 = self.x0.value
        return A * (x - x0) / (1 + (k * (x - x0)) ** 2)

    def grad_x0(self, x):
        A = self.A.value
        k = self.k.value
        x0 = self.x0.value
        return -A * k / (1 + (k * (x - x0)) ** 2)
