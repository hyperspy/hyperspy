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


class HeavisideStep(Component):

    """The Heaviside step function

    .. math::

        f(x) =
        \\begin{cases}
          0     $ \\quad \\text{if } x < n \\\\
          A/2     $ \\quad \\text{if } x = n \\\\
          A     $ \\quad \\text{if } x > n \\\\
        \\end{cases}

    """

    def __init__(self, A=1, n=0):
        Component.__init__(self, ('n', 'A'))
        self.A.value = A
        self.n.value = n
        self.isbackground = True
        self.convolved = False

        # Gradients
        self.A.grad = self.grad_A
        self.n.grad = self.grad_n

    def function(self, x):
        x = np.asanyarray(x)
        return np.where(x < self.n.value,
                        0,
                        np.where(x == self.n.value,
                                 self.A.value * 0.5,
                                 self.A.value)
                        )

    def grad_A(self, x):
        x = np.asanyarray(x)
        return np.ones(x.shape)

    def grad_n(self, x):
        x = np.asanyarray(x)
        return np.where(x < self.n.value,
                        0,
                        np.where(x == self.n.value,
                                 0.5,
                                 1)
                        )
