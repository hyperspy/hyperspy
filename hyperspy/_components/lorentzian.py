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

class Lorentzian(Expression):

    r"""Cauchy-Lorentz distribution (a.k.a. Lorentzian function) component.

    .. math::

        f(x)=\frac{A}{\pi}\left[\frac{\gamma}{\left(x-x_{0}\right)^{2}
            +\gamma^{2}}\right]

    ================= ===========
     Variable          Parameter
    ================= ===========
      :math:`A`         A     
      :math:`\gamma`   gamma   
      :math:`x_0`      centre   
    ================= ===========

    
    Parameters
    -----------
    A : float
        Height parameter, where :math:`A/(\gamma\pi)` is the maximum of the 
        peak.
    gamma : float
        Scale parameter corresponding to the half-width-at-half-maximum of the 
        peak, which corresponds to the interquartile spread.
    centre : float
        Location of the peak maximum.
    **kwargs
        Extra keyword arguments are passed to the ``Expression`` component.
    
    For convenience the `fwhm` attribute can be used to get and set
    the full-with-half-maximum.

    """

    def __init__(self, A=1., gamma=1., centre=0., module="numexpr", **kwargs):
        super(Lorentzian, self).__init__(
            expression="A / pi * (_gamma / ((x - centre)**2 + _gamma**2))",
            name="Lorentzian",
            A=A,
            _gamma=gamma,
            centre=centre,
            position="centre",
            module=module,
            autodoc=False,
            **kwargs)

        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None

        self._gamma.bmin = 0.
        self._gamma.bmax = None

        self.isbackground = False
        self.convolved = True


    @property
    def fwhm(self):
        return self._gamma.value * 2

    @fwhm.setter
    def fwhm(self, value):
        self._gamma.value = value / 2
        
    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
