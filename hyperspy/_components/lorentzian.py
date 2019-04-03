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


class Lorentzian(Component):

    r"""Cauchy-Lorentz distribution (a.k.a. Lorentzian function) component

    .. math::

        f(x)=\frac{a}{\pi}\left[\frac{\gamma}{\left(x-x_{0}\right)^{2}+\gamma^{2}}\right]

    +---------------------+-----------+
    |     Parameter       | Attribute |
    +---------------------+-----------+
    +---------------------+-----------+
    |      :math:`a`      |     A     |
    +---------------------+-----------+
    |    :math:`\gamma`   |   gamma   |
    +---------------------+-----------+
    |      :math:`x_0`    |  centre   |
    +---------------------+-----------+

    """

    def __init__(self, A=1., gamma=1., centre=0.):
        Component.__init__(self, ('A', 'gamma', 'centre'))
        self.A.value = A
        self.gamma.value = gamma
        self.centre.value = centre

        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None
        self.gamma.bmin = None
        self.gamma.bmax = None
        self._position = self.centre

        self.isbackground = False
        self.convolved = True

        # Gradients
        self.A.grad = self.grad_A
        self.gamma.grad = self.grad_gamma
        self.centre.grad = self.grad_centre

    def function(self, x):
        """
        """
        A = self.A.value
        gamma = self.gamma.value
        centre = self.centre.value

        return A / np.pi * (gamma / ((x - centre) ** 2 + gamma ** 2))

    def grad_A(self, x):
        """
        """
        return self.function(x) / self.A.value

    def grad_gamma(self, x):
        """
        """
        return self.A.value / (np.pi * (self.gamma.value ** 2 + (x - self.centre.value) ** 2)) - (
            (2 * self.A.value * self.gamma.value ** 2) / (np.pi * (self.gamma.value ** 2 + (x - self.centre.value) ** 2) ** 2))

    def grad_centre(self, x):
        """
        """
        return (2 * (x - self.centre.value) * self.A.value * self.gamma.value) / \
            (np.pi *
             (self.gamma.value ** 2 + (x - self.centre.value) ** 2) ** 2)
