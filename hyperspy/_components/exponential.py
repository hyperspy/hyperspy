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

from hyperspy._components.expression import Expression

class Exponential(Expression):

    """Exponentian function components

    f(x) = A*e^{-x/k}

    +------------+-----------+
    | Parameter  | Attribute |
    +------------+-----------+
    +------------+-----------+
    |     A      |     A     |
    +------------+-----------+
    |     k      |    tau    |
    +------------+-----------+

    """

    def __init__(self, A=1., tau=1., module="numexpr", **kwargs):
        super(Exponential, self).__init__(
            expression="A * exp(-x / tau)",
            name="Exponential",
            A=A,
            tau=tau,
            module=module,
            autodoc=False,
            **kwargs,
        )

        self.isbackground = False
