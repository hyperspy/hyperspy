# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

    r""" Doniach Sunjic lineshape

    .. math::
        :nowrap:
            
        \[
        f(x) = \frac{A \cos[ \frac{{\pi\alpha}}{2}+
        (1-\alpha)\tan^{-1}(\frac{x-centre+dx}{\sigma})]}
        {(\sigma^2 + (x-centre+dx)^2)^{\frac{(1-\alpha)}{2}}}
        \]


        \[
        dx = \frac{2.354820\sigma}{2 tan[\frac{\pi}{2-\alpha}]}
        \]


    =============== ===========
    Variable         Parameter
    =============== ===========
    :math:`A`        A
    :math:`\sigma`   sigma
    :math:`\alpha`   alpha
    :math:`centre`   centre
    =============== ===========

    Parameters
    -----------
    A : float
        Height
    sigma : float
        Variance parameter of the distribution
    alpha : float
        Tail or asymmetry parameter
    centre : float
        Location of the maximum (peak position).
    **kwargs
        Extra keyword arguments are passed to the ``Expression`` component.

    Note
    -----
    This is an asymmetric lineshape, originially design for xps but generally 
    useful for fitting peaks with low side tails
    See Doniach S. and Sunjic M., J. Phys. 4C31, 285 (1970) 
    or http://www.casaxps.com/help_manual/line_shapes.htm for a more detailed
    description
        
    """

    def __init__(self, centre=0., A=1., sigma=1., alpha=0.,
                 module=["numpy","scipy"], **kwargs):
        super(Doniach, self).__init__(
            expression="A*cos(0.5*pi*alpha+\
            ((1.0 - alpha) * arctan( (x-centre+offset)/sigma) ) )\
            /(sigma**2 + (x-centre+offset)**2)**(0.5 * (1.0 - alpha));\
            offset = 2.354820*sigma / (2 * tan(pi / (2 - alpha)))",
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

