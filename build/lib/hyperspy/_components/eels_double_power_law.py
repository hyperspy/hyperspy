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

from hyperspy.component import Component


class DoublePowerLaw(Component):

    """
    """

    def __init__(self, A=1e-5, r=3., origin=0.,):
        Component.__init__(self, ('A', 'r', 'origin', 'shift', 'ratio'))
        self.A.value = A
        self.r.value = r
        self.origin.value = origin
        self.origin.free = False
        self.shift.value = 20.
        self.shift.free = False
        self.ratio.value = 1.E-2
        self.left_cutoff = 20.  # in x-units

        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None
        self.r.bmin = 1.
        self.r.bmax = 5.

        self.isbackground = True
        self.convolved = False

    def function(self, x):
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the background model, returns the background
        model for the current parameters.
        """
        a = self.A.value
        b = self.ratio.value
        s = self.shift.value
        r = self.r.value
        x0 = self.origin.value
        return np.where(x > self.left_cutoff,
                        a * (b / (-x0 + x - s) ** r + 1 / (x - x0) ** r),
                        0)

    def grad_A(self, x):
        return self.function(x) / self.A.value

    def grad_ratio(self, x):
        a = self.A.value
        s = self.shift.value
        r = self.r.value
        x0 = self.origin.value
        return np.where(x > self.left_cutoff, a / (-x0 + x - s) ** r, 0)

    def grad_origin(self, x):
        a = self.A.value
        b = self.ratio.value
        s = self.shift.value
        r = self.r.value
        x0 = self.origin.value
        return np.where(
            x > self.left_cutoff,
            a * (
                b * r * (-x0 + x - s) ** (-r - 1) +
                r * (x - x0) ** (-r - 1)),
            0)

    def grad_shift(self, x):
        a = self.A.value
        b = self.ratio.value
        s = self.shift.value
        r = self.r.value
        x0 = self.origin.value
        return np.where(
            x > self.left_cutoff, a * b * r * (-x0 + x - s) ** (-r - 1), 0)

    def grad_r(self, x):
        a = self.A.value
        b = self.ratio.value
        s = self.shift.value
        r = self.r.value
        x0 = self.origin.value
        return np.where(
            x > self.left_cutoff,
            a * (
                -(b * np.log(-x0 + x - s)) /
                (-x0 + x - s) ** r - np.log(x - x0) /
                (x - x0) ** r),
            0)
