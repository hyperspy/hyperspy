# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

from hyperspy.decorators import auto_replot
from hyperspy.signals.spectrum import Spectrum

class SpectrumSimulation(Spectrum):
    def __init__(self, dic = None, shape = None, dtype = 'float64'):
        if dic is None and shape is not None:
            data = np.zeros((shape), dtype = dtype)
            dic = {'data' : data}
        Spectrum.__init__(self, dic)

    @auto_replot
    def add_poissonian_noise(self):
        """Add Poissonian noise to the data"""
        original_type = self.data.dtype
        self.data = np.random.poisson(self.data).astype(original_type)

    @auto_replot
    def add_gaussian_noise(self, std):
        """Add Gaussian noise to the data
        Parameters
        ----------
        std : float

        """
        noise = np.random.normal(0, std, self.data.shape)
        self.data += noise

    @auto_replot    
    def apply_spectral_gaussian_filter(self, FWHM):
        """Applies a Gaussian filter in the spectral dimension.
        
        Parameters
        ----------
        FWHM : float
        
        """
        if FWHM > 0:
            self.data = gaussian_filter1d(self.data, axis = -1, 
            sigma = FWHM/2.35482)

