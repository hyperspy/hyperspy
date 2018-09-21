# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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


class ReducedChiSquaredWeight(object):

    def __init__(self):
        self.expected = 1.0
        self.model = None

    def function(self, ind):
        return np.abs(self.model.red_chisq.data[ind] - self.expected)

    def map(self, mask, slices=slice(None, None)):
        thing = self.model.red_chisq.data[slices].copy()
        thing = thing.astype('float64')
        thing[np.logical_not(mask)] = np.nan
        return np.abs(thing - self.expected)
