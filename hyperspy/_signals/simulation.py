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

from hyperspy.decorators import auto_replot
from hyperspy.signal import BaseSignal


class Simulation(BaseSignal):
    _signal_origin = "simulation"

    def __init__(self, *args, **kwargs):
        super(Simulation, self).__init__(*args, **kwargs)

    @auto_replot
    def add_poissonian_noise(self, **kwargs):
        """Add Poissonian noise to the data"""
        original_type = self.data.dtype
        self.data = np.random.poisson(self.data, **kwargs).astype(
            original_type)

    @auto_replot
    def add_gaussian_noise(self, std):
        """Add Gaussian noise to the data
        Parameters
        ----------
        std : float

        """
        noise = np.random.normal(0,
                                 std,
                                 self.data.shape)
        original_dtype = self.data.dtype
        self.data = (
            self.data.astype(
                noise.dtype) +
            noise).astype(original_dtype)
