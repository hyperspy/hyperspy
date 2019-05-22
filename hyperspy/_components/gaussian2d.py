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
from hyperspy._components.expression import Expression

sigma2fwhm = 2 * math.sqrt(2 * math.log(2))


class Gaussian2D(Expression):
    r"""Normalized 2D elliptical Gaussian function component.

    .. math::

        f(x,y) = \frac{A}{2\pi s_x s_y}\exp\left[-\frac{\left(x-x_0\right)
        ^{2}}{2s_{x}^{2}}\frac{\left(y-y_0\right)^{2}}{2s_{y}^{2}}\right]


    =============== =============
    Variable         Parameter 
    =============== =============
    :math:`A`        A 
    :math:`s_x,s_y`  sigma_x/y 
    :math:`x_0,y_0`  centre_x/y 
    =============== =============


    Parameters
    ----------
    A : float
        Amplitude (height of the peak scaled by :math:`2 \pi s_x s_y`).
    sigma_x : float
        Width (scale parameter) of the Gaussian distribution in `x` direction.
    sigma_y : float
        Width (scale parameter) of the Gaussian distribution in `y` direction.
    centre_x : float
        Location of the Gaussian maximum (peak position) in `x` direction.
    centre_x : float
        Location of the Gaussian maximum (peak position) in `y` direction.


    For convenience the `fwhm_x` and `fwhm_y` attributes can be used to get 
    and set the full-with-half-maxima along the two axes.
    """

    def __init__(self, A=1., sigma_x=1., sigma_y=1., centre_x=0.,
                 centre_y=0, module="numexpr", **kwargs):
        super(Gaussian2D, self).__init__(
            expression="A * (1 / (sigma_x * sigma_y * 2 * pi)) * \
                       exp(-((x - centre_x) ** 2 / (2 * sigma_x ** 2) \
                       + (y - centre_y) ** 2 / (2 * sigma_y ** 2)))",
            name="Gaussian2D",
            A=A,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            centre_x=centre_x,
            centre_y=centre_y,
            position=("centre_x", "centre_y"),
            module=module,
            autodoc=False,
            **kwargs)

        # Boundaries
        self.A.bmin = 0.

        self.sigma_x.bmin = 0.
        self.sigma_y.bmin = 0.

        self.isbackground = False
        self.convolved = True

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
