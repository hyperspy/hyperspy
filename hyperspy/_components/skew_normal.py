# -*- coding: utf-8 -*-
# Copyright 2007-2018 The HyperSpy developers
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
from hyperspy._components.expression import Expression
from distutils.version import LooseVersion
import sympy


class SkewNormal(Expression):

    r"""Skew normal distribution component.

    |  Asymmetric peak shape based on a normal distribution.
    |  For definition see
       https://en.wikipedia.org/wiki/Skew_normal_distribution
    |  See also http://azzalini.stat.unipd.it/SN/
    |

    .. math::

        f(x) &= 2 A \phi(x) \Phi(x) \\
        \phi(x) &= \frac{1}{\sqrt{2\pi}}\mathrm{exp}{\left[
                   -\frac{t(x)^2}{2}\right]} \\
        \Phi(x) &= \frac{1}{2}\left[1 + \mathrm{erf}\left(\frac{
                   \alpha~t(x)}{\sqrt{2}}\right)\right] \\
        t(x) &= \frac{x-x_0}{\omega}


    ============== =============
    Variable        Parameter 
    ============== =============
    :math:`x_0`     x0 
    :math:`A`       A 
    :math:`\omega`  scale 
    :math:`\alpha`  shape 
    ============== =============


    Parameters
    -----------
    x0 : float
        Location of the peak position (not maximum, which is given by
        the `mode` property).
    A : float
        Height parameter of the peak.
    scale : float
        Width (sigma) parameter.
    shape: float
        Skewness (asymmetry) parameter. For shape=0, the normal
        distribution (Gaussian) is obtained. The distribution is
        right skewed (longer tail to the right) if shape>0 and is
        left skewed if shape<0.


    The properties `mean` (position), `variance`, `skewness` and `mode`
    (=position of maximum) are defined for convenience.
    """

    def __init__(self, x0=0., A=1., scale=1., shape=0., module="scipy",
                 **kwargs):
        if LooseVersion(sympy.__version__) < LooseVersion("1.3"):
            raise ImportError("The `SkewNormal` component requires "
                              "SymPy >= 1.3")
        super(SkewNormal, self).__init__(
            expression="2 * A * normpdf * normcdf; normpdf = exp(- t ** 2 / 2) \
                / sqrt(2 * pi); normcdf = (1 + erf(shape * t / sqrt(2))) / 2; \
                t = (x - x0) / scale",
            name="SkewNormal",
            x0=x0,
            A=A,
            scale=scale,
            shape=shape,
            module=module,
            autodoc=False,
            **kwargs,
        )

        # Boundaries
        self.A.bmin = 0.

        self.scale.bmin = 0

        self.isbackground = False
        self.convolved = True

    @property
    def mean(self):
        delta = self.shape.value / np.sqrt(1 + self.shape.value**2)
        return self.x0.value + self.scale.value * delta * np.sqrt(2 / np.pi)

    @property
    def variance(self):
        delta = self.shape.value / np.sqrt(1 + self.shape.value**2)
        return self.scale.value**2 * (1 - 2 * delta**2 / np.pi)

    @property
    def skewness(self):
        delta = self.shape.value / np.sqrt(1 + self.shape.value**2)
        return (4 - np.pi)/2 * (delta * np.sqrt(2/np.pi))**3 / (1 - \
            2 * delta**2 / np.pi)**(3/2)

    @property
    def mode(self):
        delta = self.shape.value / np.sqrt(1 + self.shape.value**2)
        muz = np.sqrt(2 / np.pi) * delta
        sigmaz = np.sqrt(1 - muz**2)
        if self.shape.value == 0:
            return self.x0.value
        else:
            m0 = muz - self.skewness * sigmaz / 2 - np.sign(self.shape.value) \
                / 2 * np.exp(- 2 * np.pi / np.abs(self.shape.value))
            return self.x0.value + self.scale.value * m0
