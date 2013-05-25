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

from hyperspy.signals import EELSSpectrumSimulation
from hyperspy.components import Gaussian

class Test_Estimate_Elastic_Scattering_Threshold:
    def setUp(self):
        # Create an empty spectrum
        s = EELSSpectrumSimulation(np.zeros((3,2,1024)))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 0.02
        energy_axis.offset = -5

        gauss = Gaussian()
        gauss.centre.value = 0
        gauss.A.value = 5000
        gauss.sigma.value = 0.5
        gauss2 = Gaussian() 
        gauss2.sigma.value = 0.5
        # Inflexion point 1.5
        gauss2.A.value = 5000
        gauss2.centre.value = 5
        s.data[:] = (gauss.function(energy_axis.axis) + 
                     gauss2.function(energy_axis.axis))
#        s.add_poissonian_noise()
        self.signal = s
        
    def test_min_in_window_with_smoothing(self):
        s = self.signal
        thr = s.estimate_elastic_scattering_threshold(
                        window = 5,
                        number_of_points=5,
                        tol=0.00001)
        assert_true(np.allclose(thr.data, 2.5))
        
    def test_min_in_window_without_smoothing(self):
        s = self.signal
        thr = s.estimate_elastic_scattering_threshold(
                        window = 5,
                        number_of_points=0,
                        tol=0.001)
        assert_true(np.allclose(thr.data, 2.49))
            
    def test_min_not_in_window(self):
        # If I use a much lower window, this is the value that has to be
        # returned as threshold.
        s = self.signal
        data = s.estimate_elastic_scattering_threshold(window=1.5,
                                                       tol=0.001).data
        assert_true(np.all(np.isnan(data)))
        
class TestEstimateZLPCentre():
    def setUp(self):
        s = EELSSpectrumSimulation(np.diag(np.arange(1.5,3.5,0.2)))
        s.axes_manager[-1].scale = 0.1
        s.axes_manager[-1].offset = 100
        self.spectrum = s
    def test_calibrate_false(self):
        s = self.spectrum
        assert_equal(s.estimate_zero_loss_peak_centre(calibrate=False), 100.45)
        
    def test_calibrate_true(self):
        s = self.spectrum
        s.estimate_zero_loss_peak_centre()
        assert_true(np.allclose(s.estimate_zero_loss_peak_centre(), 0))
        
    def test_also_align(self):
        s = self.spectrum
        sc = s.deepcopy()
        s.estimate_zero_loss_peak_centre(calibrate=True, also_apply_to=[sc,])
        assert_true(np.allclose(sc.estimate_zero_loss_peak_centre(), 0))
                
    
        
