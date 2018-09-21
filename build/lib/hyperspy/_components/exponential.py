# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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


class Exponential(Component):

    """Exponentian function components

    f(x) = A*e^{-x/k}

    +------------+-----------+
    | Parameter  | Attribute |
    +------------+-----------+
    +------------+-----------+
    |     A      |     A     |
    +------------+-----------+
    |     k      |    tau    |
    +------------+-----------+

    """

    def __init__(self):
        Component.__init__(self, ['A', 'tau'])
        self.isbackground = False
        self.A.grad = self.grad_A
        self.tau.grad = self.grad_tau

    def function(self, x):
        """
        """
        A = self.A.value
        tau = self.tau.value
        return A * np.exp(-x / tau)

    def grad_A(self, x):
        return self.function(x) / self.A.value

    def grad_tau(self, x):
        A = self.A.value
        tau = self.tau.value
        return x * (np.exp(-x / tau)) * A / tau ** 2
