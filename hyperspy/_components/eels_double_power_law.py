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


class DoublePowerLaw(Expression):

    """
    """

    def __init__(self, A=1e-5, r=3., origin=0., shift=20., ratio=1.,
                 **kwargs):
        super(DoublePowerLaw, self).__init__(
            expression="A * (ratio * (x - origin - shift) ** -r + (x - origin) ** -r)",
            name="DoublePowerLaw",
            A=A,
            r=r,
            origin=origin,
            shift=shift,
            ratio=ratio,
            position="origin",
            autodoc=True,
            **kwargs,
        )

        self.origin.free = False
        self.shift.value = 20.
        self.shift.free = False
        self.left_cutoff = 0.  # in x-units

        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None
        self.r.bmin = 1.
        self.r.bmax = 5.

        self.isbackground = True
        self.convolved = False

    def function(self, x):
        return np.where(x > self.left_cutoff, super().function(x), 0)

    def function_nd(self, axis):
        """%s

        """
        return np.where(axis > self.left_cutoff, super().function_nd(axis), 0)

    function_nd.__doc__ %= FUNCTION_ND_DOCSTRING
