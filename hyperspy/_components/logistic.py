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


class Logistic(Expression):

    """Logistic function (sigmoid/s-shaped curve) component

    .. math::
    
        f(x) = a / (1 + b * exp(-c * (x - origin)))

    Parameters
    -----------
    a : Float
        The curve's maximum y-value, lim(x->infty)=a
    b : Float
        Additional parameter: 
        b>1 shifts origin to larger values
        0<b<1 shifts origin to smaller values
        b<0 introduces an asymptote
    c : Float
        logistic growth rate or steepness of the curve
    origin : Float
        position of the sigmoid's midpoint

    """

    def __init__(self, a=1., b=1., c=1., origin=0., module="numexpr", **kwargs):
        super(Logistic, self).__init__(
            expression="a / (1 + b * exp(-c * (x - origin)))",
            name="Logistic",
            a=a,
            b=b,
            c=c,
            origin=origin,
            position="origin",
            module=module,
            autodoc=False,
            **kwargs)

        # Boundaries
        self.isbackground = False
