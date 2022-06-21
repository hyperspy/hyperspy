# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import numpy as np
import math

from hyperspy.component import Component
from hyperspy.docstrings.parameters import FUNCTION_ND_DOCSTRING

sqrt2pi = np.sqrt(2 * np.pi)


def _calculate_shirley_background(values):
    cf = np.cumsum(values, axis=-1)
    # necessary to work with `function_nd`
    return cf[..., -1][np.newaxis].T - cf


class PESCoreLineShape(Component):

    """
    """

    def __init__(self, A=1., FWHM=1., origin=0., ab=0.0, shirley=0.0):
        Component.__init__(self, ['A', 'FWHM', 'origin', 'ab', 'shirley'])
        self.ab.value = 0
        self.ab.free = False
        self.A.value = A
        self.FWHM.value = FWHM
        self.origin.value = origin
        self._position = self.origin

        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None
        self.FWHM.bmin = None
        self.FWHM.bmax = None

        self.isbackground = False
        self.convolved = True

        # Gradients
        self.A.grad = self.grad_A
        self.FWHM.grad = self.grad_FWHM
        self.origin.grad = self.grad_origin
        self.ab.grad = self.grad_ab

        # Options
        self.Shirley = False
        self._whitelist['Shirley'] = None

    @property
    def Shirley(self):
        return self._Shirley

    @Shirley.setter
    def Shirley(self, value):
        self._Shirley = value
        self.shirley.free = value

    def _function(self, x, A, origin, FWHM, ab, shirley):
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the background model, returns the background
        model for the current parameters.
        """
        f = A * np.exp(-1 * math.log(2) * ((x - (origin - ab)) / FWHM) ** 2)
        if self.Shirley:
            return _calculate_shirley_background(f) * shirley + f
        else:
            return f

    def function(self, x):
        return self._function(x, self.A.value,
                              self.origin.value,
                              self.FWHM.value,
                              self.ab.value,
                              self.shirley.value)

    def function_nd(self, axis):
        """%s

        """
        if self._is_navigation_multidimensional:
            x = axis[np.newaxis, :]
            A = self.A.map['values'][..., np.newaxis]
            origin = self.origin.map['values'][..., np.newaxis]
            FWHM = self.FWHM.map['values'][..., np.newaxis]
            ab = self.ab.map['values'][..., np.newaxis]
            shirley = self.shirley.map['values'][..., np.newaxis]
            return self._function(x, A, origin, FWHM, ab, shirley)
        else:
            return self.function(axis)

    function_nd.__doc__ %= FUNCTION_ND_DOCSTRING

    def grad_A(self, x):
        return self.function(x) / self.A.value

    def grad_FWHM(self, x):
        a0 = self.A.value
        a1 = self.origin.value
        a2 = self.FWHM.value
        a3 = self.ab.value
        return (2 * math.log(2) * a0 * (x + a3 - a1) ** 2 *
                np.exp(-(math.log(2) * (x + a3 - a1) ** 2) / a2 ** 2)) / a2 ** 3

    def grad_origin(self, x):
        a0 = self.A.value
        a1 = self.origin.value
        a2 = self.FWHM.value
        a3 = self.ab.value
        return (2 * math.log(2) * a0 * (x + a3 - a1) *
                np.exp(-(math.log(2) * (x + a3 - a1) ** 2) / a2 ** 2)) / a2 ** 2

    def grad_ab(self, x):
        return -self.grad_origin(x)

    def grad_shirley(self, x):
        a0 = self.A.value
        a1 = self.origin.value
        a2 = self.FWHM.value
        a3 = self.ab.value
        f = a0 * np.exp(-1 * math.log(2) * ((x - (a1 - a3)) / a2) ** 2)
        return _calculate_shirley_background(f)
