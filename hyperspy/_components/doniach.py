# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of  GNU General Public License as published by
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

tiny = np.finfo(np.float64).eps

class Doniach(Expression):

    """ 
    Doniach Sunjic lineshape
    An asymmetric lineshape, originially design for xps but generally useful
    for fitting peaks with low side tails e.g. compton peaks

    .. math::

        f(x)==\frac{cos[ \frac{{\pi\alpha}}{2}+
                   (1-\alpha)tan^{-1}(\frac{x-centre}{\sigma})]}
                   {(\sigma^2 + (x-centre)^2)^{\frac{(1-\alpha)}{2}}}
                   
    +---------------------+-----------+
    |     Parameter       | Attribute |
    +---------------------+-----------+
    |      :math:`A`      |     A     |
    +---------------------+-----------+
    |    :math:`\sigma`   |   sigma   |
    +---------------------+-----------+
    |    :math:`\alpha`   |  alpha    |
    +---------------------+-----------+
    |   :math:`centre`    |  centre   |
    +---------------------+-----------+

    References
    ----------
    [1] Doniach S. and Sunjic M., J. Phys. 4C31, 285 (1970)
    [2] http://www.casaxps.com/help_manual/line_shapes.htm
        

    """

    def __init__(self, centre=0., A=1., sigma=1., alpha=0.,
                 module=["numpy","scipy"], **kwargs):
        super(Doniach, self).__init__(
            expression="A/(sigma**(1.0-alpha))*cos(0.5*pi*alpha+\
                        ((1.0 - alpha) * arctan( (x-centre)/sigma) ) )\
                /(1.0 + ((x-centre)/sigma)**2)**(0.5 * (1.0 - alpha))",
            name="Doniach",
            centre=centre,
            A=A,
            sigma=sigma,
            alpha=alpha,
            module=module,
            autodoc=False,
            **kwargs,
        )
        #
        self.sigma.bmin = 1.0e-8
        self.alpha.bmin = 1.0e-8
        self.isbackground = False
        self.convolved = True

