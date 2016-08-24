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


class Gaussian2D(Component):
    """Normalized 2D elliptical gaussian function component

    .. math::
        f(x,y) = \\frac{A}{2\pi s_x s_y}}}e^{-\\frac{\left(x-x0\\right)
        ^{2}}{2s_{x}^{2}}\\frac{\left(y-y0\\right)^{2}}{2s_{y}^{2}}}
    +------------+-----------+
    | Parameter  | Attribute |
    +------------+-----------+
    +------------+-----------+
    |      a     | amplitude |
    +------------+-----------+
    |    x0,y0   |  centre   |
    +------------+-----------+
    |   s_x,s_y  |   sigma   |
    +------------+-----------+
    |    theta   | rotation  |
    +------------+-----------+

    The rotation value is in radians from the x-axis.
    The rotation in degrees can be accessed using the property
    rotation_degrees which will give the angle between the x-axis and the
    major axis, ergo between the x-axis and the highest sigma.

    For convenience the `fwhm_x` and `fwhm_y` attributes can be used to
    get and set the full-with-half-maximums.

    The `ellipticity` attribute returns `sigma_x`/`sigma_y`.

    The `rotation_degrees` attribute returns the angle between
    the major axis (the largest sigma), and the positive
    horizontal axis.
    """

    def __init__(self,
                 A=1.,
                 sigma_x=1.,
                 sigma_y=1.,
                 centre_x=0.,
                 centre_y=0.,
                 rotation=0.01,
                 ):
        """Note: the rotation value is given in
        radians."""
        Component.__init__(self, ['A',
                                  'sigma_x',
                                  'sigma_y',
                                  'centre_x',
                                  'centre_y',
                                  'rotation',
                                  ])
        self.A.value = A
        self.sigma_x.value = sigma_x
        self.sigma_y.value = sigma_y
        self.centre_x.value = centre_x
        self.centre_y.value = centre_y
        self.rotation.value = rotation

    def function(self, x, y):
        A = self.A.value
        sx = self.sigma_x.value
        sy = self.sigma_y.value
        x0 = self.centre_x.value
        y0 = self.centre_y.value
        theta = self.rotation.value

        sx2 = sx**2
        sy2 = sy**2
        cos_theta2 = math.cos(theta)**2
        sin_theta2 = math.sin(theta)**2
        sin_2theta = math.sin(2*theta)

        a = cos_theta2/(2*sx2) + sin_theta2/(2*sy2)
        b = -sin_2theta/(4*sx2) + sin_2theta/(4*sy2)
        c = sin_theta2/(2*sx2) + cos_theta2/(2*sy2)

        return A * (1 / (sx * sy * pi2)) * np.exp(-(a*(x - x0) ** 2 +
                                                  2*b*(x - x0) * (y - y0) +
                                                  c*(y - y0) ** 2))

    @property
    def rotation(self):
        return self.rotation.value

    @rotation.setter
    def rotation(self, value):
        self.rotation.value = value % math.pi

    @property
    def ellipticity(self):
        return self.sigma_x.value/self.sigma_y.value

    @property
    def fwhm_x(self):
        return self.sigma_x.value * sigma2fwhm

    @fwhm_x.setter
    def fwhm_x(self, value):
        self.sigma_x.value = value / sigma2fwhm

    @property
    def fwhm_y(self):
        return self.sigma_y.value * sigma2fwhm

    @fwhm_y.setter
    def fwhm_y(self, value):
        self.sigma_y.value = value / sigma2fwhm

    @property
    def rotation_degrees(self):
        """Angle between major axis and x-axis."""
        rotation = math.fmod(self.rotation.value, pi2)
        if self.sigma_x.value > self.sigma_y.value:
            return math.degrees(rotation)
        else:
            return math.degrees(rotation-pi2/4)
