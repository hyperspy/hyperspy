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

from nose.tools import assert_true, assert_equal, assert_not_equal
from hyperspy.signals.eels import EELSSpectrum
from hyperspy.components import Lorentzian, Bleasdale

class Test_Estimate_Elastic_Scattering_Threshold:
    def setUp(self):
        # Create an empty spectrum
        s = EELSSpectrum(np.zeros((32,32,1024)))
        ejeE = s.axes_manager.signal_axes[0]
        ejeE.scale = 0.02
        ejeE.offset = -5

        LOR = Lorentzian()
        GAP = Bleasdale()

        rnd=np.random.random
        ij=s.axes_manager
        LOR.gamma.value = 0.2

        GAP.b.value = 10000
        GAP.c.value = -2 #sqrt

        for i in enumerate(s):
            LOR.centre.value = 0
            LOR.A.value = 5000 + (rnd() - 0.5) * 5000
            # The ZLP
            s.data[ij.coordinates] = LOR.function(ejeE.axis)
            # NOTICE THE GAP IS SET AT 3+-.5 eV
            GAP.a.value = -3 + (rnd() - 0.5) 
            data = GAP.function(ejeE.axis)
            whereAreNaNs = np.isnan(data)
            data[whereAreNaNs] = 0
            # The Gap
            s.data[ij.coordinates] += data
            
        s.data = np.random.poisson(s.data)
        self.signal = s
        
    def test_min_in_window(self):
        s = self.signal
        thr = s.estimate_elastic_scattering_threshold(
                        window = 7,
                        npoints=20,
                        tol=0.05)
        np.allclose(thr.data, 3, rtol = 0.5)
            
    def test_min_not_in_window(self):
        # If I use a much lower window, this is the value that has to be
        # returned as threshold.
        window =1.5 
        s = self.signal
        thr = s.estimate_elastic_scattering_threshold(window)
        np.allclose(thr.data,1.5, rtol = 0.01)
