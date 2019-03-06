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

class RC(Expression):

    """
    RC function component (based on the time-domain capacitor voltage response of an RC-circuit)
    
    .. math::

        f(x) = V0 + V * (1 - exp(-x / \tau))
        
    Parameters
    -----------
        V0: float
            vertical offset
        V: float
            maximum voltage, asymptote of the function for lim(x->infty)
        tau: float
            tau=RC is the RC circuit time constant (voltage rise time)
    
    """

    def __init__(self, V=1., V0=0., tau=1., module="numexpr", **kwargs):
        super(RC, self).__init__(
            expression="V0 + V * (1 - exp(-x / tau))",
            name="RC",
            V=V,
            V0=V0,
            tau=tau,
            module=module,
            autodoc=False,
            **kwargs,
        )
        
        self.isbackground = True
