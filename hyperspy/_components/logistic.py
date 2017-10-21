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


class Logistic(Component):

    """Logistic function component

    f(x) = a/(1+b*exp(-c*(x-origin)))

    Attributes
    ----------
    a : Float
    b : Float
    c : Float
    origin : Float

    """

    def __init__(self):
        # Define the parameters
        Component.__init__(self, ('a', 'b', 'c', 'origin'))
        # Define the name of the component
        self.a.grad = self.grad_a
        self.b.grad = self.grad_b
        self.c.grad = self.grad_c
        self.origin.grad = self.grad_origin
        self._position = self.origin

    def function(self, x):
        """
        """
        a = self.a.value
        b = self.b.value
        c = self.c.value
        origin = self.origin.value
        return a / (1 + b * np.exp(-c * (x - origin)))

    def grad_a(self, x):
        """
        Returns d(function)/d(parameter_1)
        """
        b = self.b.value
        c = self.c.value
        origin = self.origin.value

        return 1 / (1 + b * np.exp(-c * (x - origin)))

    def grad_b(self, x):
        """
        Returns d(function)/d(parameter_1)
        """
        a = self.a.value
        b = self.b.value
        c = self.c.value
        origin = self.origin.value

        return -(a * np.exp(-c * (x - origin))) / \
            (b * np.exp(-c * (x - origin)) + 1) ** 2

    def grad_c(self, x):
        """
        Returns d(function)/d(parameter_1)
        """
        a = self.a.value
        b = self.b.value
        c = self.c.value
        origin = self.origin.value

        return -(a * b * (origin - x) * np.exp(-c * (x - origin))) / \
            (b * np.exp(-c * (x - origin)) + 1) ** 2

    def grad_origin(self, x):
        """
        Returns d(function)/d(parameter_1)
        """
        a = self.a.value
        b = self.b.value
        c = self.c.value
        origin = self.origin.value

        return -(a * b * c * np.exp(-c * (x - origin))) / \
            (b * np.exp(-c * (x - origin)) + 1) ** 2
