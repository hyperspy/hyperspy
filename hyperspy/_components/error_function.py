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

#import math

import numpy as np
#from scipy.special import erf

from hyperspy._components.expression import Expression

sqrt2pi = np.sqrt(2 * np.pi)


class Erf(Expression):

r"""Error function component.

    .. math::
    
        \frac{A}{2} \mathrm{erf}\left[\frac{(x - x_0)}{\sqrt{2} \sigma}\right]

    ============== =============
    Variable        Parameter 
    ============== =============
    :math:`A`       A
    :math:`\simga`  sigma
    :math:`x_0`     origin
    ============== =============

    Parameters
    -----------
        A : float
            Height parameter.
        sigma : float
        
        origin : float
        
    """
    
    def __init__(self, A=1., sigma=1., origin=0., module="scipy",
                 **kwargs):
        super(Erf, self).__init__(
            expression="A * erf((x - origin) / sqrt(2) / sigma) / 2",
            name="Erf",
            A=A,
            sigma=sigma,
            origin=origin,
            module=module,
            autodoc=False,
            **kwargs,
        )

        # Boundaries
        self.A.bmin = 0.

        self.isbackground = False
        self.convolved = True
