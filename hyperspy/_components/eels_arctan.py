# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.

import math

import numpy as np

from hyperspy._components.expression import Expression

class Arctan(Expression):
    # Legacy class to be removed in v2.0
    """This is the legacy Arctan component dedicated to EELS measurements
    that will renamed to `EELSArctan` in v2.0. To use the new Arctan component 
    set `minimum_at_zero=False`. See the documentation of 
    :meth:`hyperspy._components.acrtan.Arctan` for details on 
    the usage of the new Arctan component.
    
    .. math::
    
        f(x) = A \cdot \arctan\left[ k \left( x-x_0 \right)\right]
    
    EELSArctan (`minimum_at_zero=True`) shifts the function by pi/2 in the y 
    direction
    
    """

    def __init__(self, minimum_at_zero=True, **kwargs):
        if minimum_at_zero:
            from hyperspy.misc.utils import deprecation_warning
            msg = (
                "The API of the `Arctan` component will change in v2.0. "
                "This component will become `EELSArctan`."
                "To use the new API set `minimum_at_zero=False`.")
            deprecation_warning(msg)

            self.__class__ = EELSArctan
            self.__init__(**kwargs)
        else:
            from hyperspy._components.arctan import Arctan
            self.__class__ = Arctan
            self.__init__(**kwargs)


class EELSArctan(Expression):

    r"""Arctan function component for EELS (with minimum at zero).

    .. math::
    
        f(x) = A \cdot \left{\frac{\pi}{2} + 
               \arctan\left[ k \left( x-x_0 \right)\right]\right}

    ============ =============
    Variable      Parameter 
    ============ =============
    :math:`A`     A 
    :math:`k`     k 
    :math:`x_0`   x0 
    ============ =============

    Parameters
    -----------
    A : float
    k : float
    x0 : float

    """

    def __init__(self, A=1., k=1., x0=1., module=["numpy", "scipy"], **kwargs):
        # Not to break scripts once we remove the legacy Arctan
        if "minimum_at_zero" in kwargs:
            del kwargs["minimum_at_zero"]
        super(EELSArctan, self).__init__(
            expression="A * (pi /2 + arctan(k * (x - x0)))",
            name="Arctan",
            A=A,
            k=k,
            x0=x0,
            position="x0",
            module=module,
            autodoc=False,
            **kwargs,
        )

        self.isbackground = False
        self.convolved = True
