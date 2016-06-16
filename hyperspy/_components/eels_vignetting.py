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
from .gaussian import Gaussian


class Vignetting(Component):

    """
    Model the vignetting of the lens with a cos^4 law multiplied by lines on
    the edges
    """

    def __init__(self):
        Component.__init__(self,
                           ['optical_center',
                            'height',
                            'period',
                            'left_slope',
                            'right_slope',
                            'left',
                            'right',
                            'sigma'])
        self.left.value = np.nan
        self.right.value = np.nan
        self.side_vignetting = False
        self.fix_side_vignetting()
        self.gaussian = Gaussian()
        self.gaussian.origin.free, self.gaussian.A.free = False, False
        self.sigma.value = 1.
        self.gaussian.A.value = 1.
        self.extension_nch = 100
        self._position = self.optical_center

    def function(self, x):
        sigma = self.sigma.value
        x0 = self.optical_center.value
        A = self.height.value
        period = self.period.value
        la = self.left_slope.value
        ra = self.right_slope.value
        l = self.left.value
        r = self.right.value
        ex = self.extension_nch
        if self.side_vignetting is True:

            x = x.tolist()
            x = list(range(-ex, 0)) + x + \
                list(range(int(x[-1]) + 1, int(x[-1]) + ex + 1))
            x = np.array(x)
            v1 = A * np.cos((x - x0) / (2 * np.pi * period)) ** 4
            v2 = np.where(x < l,
                          1. - (l - x) * la,
                          np.where(x < r,
                                   1.,
                                   1. - (x - r) * ra))
            self.gaussian.sigma.value = sigma
            self.gaussian.origin.value = (x[-1] + x[0]) / 2
            result = np.convolve(self.gaussian.function(x), v1 * v2, 'same')
            return result[ex:-ex]
        else:
            return A * np.cos((x - x0) / (2 * np.pi * period)) ** 4

    def free_side_vignetting(self):
        self.left.free = True
        self.right.free = True
        self.left_slope.free = True
        self.right_slope.free = True
        self.sigma.free = True

    def fix_side_vignetting(self):
        self.left.free = False
        self.right.free = False
        self.left_slope.free = False
        self.right_slope.free = False
        self.sigma.free = False

    def free_cos_vignetting(self):
        self.optical_center.free = True
        self.period.free = True
        self.height.free = True

    def fix_cos_vignetting(self):
        self.optical_center.free = False
        self.period.free = False
        self.height.free = False
