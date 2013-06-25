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

from hyperspy.signals import EDSTEMSpectrum
from hyperspy.defaults_parser import preferences
from hyperspy.io import load

class Test_mapped_parameters:
    def setUp(self):
        # Create an empty spectrum
        s = EDSTEMSpectrum(np.ones((4,2,1024)))
        s.mapped_parameters.TEM.EDS.live_time = 3.1              
        self.signal = s
        
    def test_sum_live_time(self):
        s = self.signal
        sSum = s.sum(0)
        assert_equal(sSum.mapped_parameters.TEM.EDS.live_time, 3.1*2)
    
    def test_rebin_live_time(self):
        s = self.signal
        dim = s.axes_manager.shape
        s.rebin([dim[0]/2,dim[1]/2,dim[2]])
        assert_equal(s.mapped_parameters.TEM.EDS.live_time, 3.1*2*2)
 
    def test_set_X_line(self):
        s = self.signal
        results = []
        mp = s.mapped_parameters
        s.set_elements(['Al','Ni'],['Ka','La'])
        results.append(mp.Sample.Xray_lines[0])
        results.append(mp.Sample.elements[1])
        mp.TEM.beam_energy = 15.0
        s.set_elements(['Al','Ni'])
        results.append(mp.Sample.Xray_lines[1])
        mp.TEM.beam_energy = 10.0
        s.set_elements(['Al','Ni'])
        results.append(mp.Sample.Xray_lines[1])
        s.add_elements(['Fe'])
        results.append(mp.Sample.Xray_lines[1])    
        assert_equal(results, ['Al_Ka','Ni','Ni_Ka','Ni_La','Fe_La'])
        
    def test_default_param(self):
        s = self.signal
        mp = s.mapped_parameters
        assert_equal(mp.TEM.EDS.energy_resolution_MnKa,
            preferences.EDS.eds_mn_ka)
            
    def test_SEM_to_TEM(self):
        s = self.signal[0,0]
        signal_type = 'EDS_SEM'
        mp = s.mapped_parameters
        mp.TEM.EDS.energy_resolution_MnKa = 125.3
        sSEM = s.deepcopy()
        sSEM.set_signal_type(signal_type)        
        mpSEM = sSEM.mapped_parameters            
        results = [mp.TEM.EDS.energy_resolution_MnKa]
        results.append(signal_type)        
        resultsSEM = [mpSEM.SEM.EDS.energy_resolution_MnKa]
        resultsSEM.append(mpSEM.signal_type)        
        assert_equal(results,resultsSEM )
        
    def test_get_calibration_from(self):
        s = self.signal
        scalib = EDSTEMSpectrum(np.ones((1024)))
        energy_axis = scalib.axes_manager.signal_axes[0]
        energy_axis.scale = 0.01
        energy_axis.offset = -0.10
        s.get_calibration_from(scalib)
        assert_equal(s.axes_manager.signal_axes[0].scale,
            energy_axis.scale)
        
        
class Test_get_intentisity_map:
    def setUp(self):
        # Create an empty spectrum
        s = EDSTEMSpectrum(np.ones((4,2,1024)))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 0.01
        energy_axis.offset = -0.10
        energy_axis.units = 'keV'                
        self.signal = s
    
    def test(self):        
        s = self.signal
        s.set_elements(['Al','Ni'],['Ka','La'])
        sAl = s.get_intensity_map(plot_result=True)[0]
        assert_true(np.allclose(s[...,0].data*15.0, sAl.data))

