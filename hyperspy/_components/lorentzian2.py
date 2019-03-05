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

from hyperspy._components.expression import Expression

class Lorentzian2(Expression):

    """Cauchy-Lorentz distribution (a.k.a. Lorentzian function) component implemented as expression

    .. math::

        f(x)=\frac{a}{\pi}\left[\frac{\Gamma}{\left(x-x_{0}\right)^{2}+\Gamma^{2}}\right]

    +---------------------+-----------+
    |     Parameter       | Attribute |
    +---------------------+-----------+
    +---------------------+-----------+
    |      :math:`a`      |     A     |
    +---------------------+-----------+
    |    :math:`\Gamma`   |   Gamma   |
    +---------------------+-----------+
    |      :math:`x_0`    |  centre   |
    +---------------------+-----------+
    
    For convenience the `fwhm` attribute can be used to get and set
    the full-with-half-maximum.

    """

    def __init__(self, A=1., Gamma=1., centre=0., module="numexpr", **kwargs):
        super(Lorentzian2, self).__init__(
            expression="A / pi * (Gamma / ((x - centre)**2 + Gamma**2))",
            name="Lorentzian2",
            A=A,
            Gamma=Gamma,
            centre=centre,
            position="centre",
            module=module,
            autodoc=False,
            **kwargs)

        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None

        self.Gamma.bmin = 0.
        self.Gamma.bmax = None

        self.isbackground = False
        self.convolved = True


    @property
    def fwhm(self):
        return self.Gamma.value * 2

    @fwhm.setter
    def fwhm(self, value):
        self.Gamma.value = value / 2
