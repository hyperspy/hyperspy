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

import math
import numpy as np
from hyperspy.component import Component

pi2 = 2 * math.pi
sigma2fwhm = 2 * math.sqrt(2 * math.log(2))


class SymmetricGaussian2D(Component):
    """Normalized symmetric 2D elliptical gaussian function component

    .. math::
        f(x,y) = \\frac{A}{2\pi s^{2}}}}e^{-\\frac{\left(x-x0\\right)
        ^{2}}{2s^{2}}\\frac{\left(y-y0\\right)^{2}}{2s^{2}}}
    +------------+-----------+
    | Parameter  | Attribute |
    +------------+-----------+
    +------------+-----------+
    |      a     | amplitude |
    +------------+-----------+
    |    x0,y0   |  centre   |
    +------------+-----------+
    |      s     |   sigma   |
    +------------+-----------+

    See also
    --------
    Gaussian2D : elliptical 2D Gaussian
    """

    def __init__(self,
                 A=1.,
                 sigma=1.,
                 centre_x=0.,
                 centre_y=0.,
                 ):
        Component.__init__(self, ['A',
                                  'sigma',
                                  'centre_x',
                                  'centre_y',
                                  ])
        self.A.value = A
        self.sigma.value = sigma
        self.centre_x.value = centre_x
        self.centre_y.value = centre_y

    def function(self, x, y):
        A = self.A.value
        s = self.sigma.value
        x0 = self.centre_x.value
        y0 = self.centre_y.value

        s2 = s**2

        return A * (1 / (pi2 * s2)) * np.exp(
                -((x - x0) ** 2 / (2 * s2) + (y - y0) ** 2 / (2 * s2)))

    @property
    def fwhm(self):
        return self.sigma.value * sigma2fwhm

    @fwhm.setter
    def fwhm(self, value):
        self.sigma.value = value / sigma2fwhm
