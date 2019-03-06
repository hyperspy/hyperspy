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

import numpy as np

from hyperspy.docstrings.parameters import FUNCTION_ND_DOCSTRING
from hyperspy._components.expression import Expression

class Bleasdale(Expression):

    """Bleasdale function component.

    .. math::
    
        f(x) = (a+b*x)^(-1/c)

    Attributes
    ----------
    a : Float
    b : Float
    c : Float
    
    For (a+b*x)<=0, the component will return 0.

    """
    
    def __init__(self, a=1., b=1., c=1., module="numexpr", **kwargs):
        super(Bleasdale, self).__init__(
            expression="(a + b * x) ** (-1 / c)",
            name="Bleasdale",
            a=a,
            b=b,
            c=c,
            module=module,
            autodoc=False,
            **kwargs)
    
    def function(self, x):
        return np.where((a + b * x) > 0., super().function(x), 0.)

    def function_nd(self, axis):
        """%s

        """
        return np.where((a + b * axis) > 0, super().function_nd(axis), 0)

    function_nd.__doc__ %= FUNCTION_ND_DOCSTRING

