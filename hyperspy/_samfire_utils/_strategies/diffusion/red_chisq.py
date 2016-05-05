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

from hyperspy._samfire_utils.strategy import DiffusionStrategy
from hyperspy._samfire_utils._weights.red_chisq import ReducedChiSquaredWeight


def exp_decay(distances):
    """Exponential decay function."""
    return np.exp(-distances)


class ReducedChiSquaredStrategy(DiffusionStrategy):
    """Reduced chi-squared Diffusion strategy of the SAMFire. Uses reduced
    chi-squared as the weight, and exponential decay as the decau function.
    """

    def __init__(self):
        DiffusionStrategy.__init__(self, 'Reduced chi squared strategy')
        self.weight = ReducedChiSquaredWeight()
        self.radii = 3.
        self.decay_function = exp_decay
