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
from __future__ import division

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import traits.api as t

from hyperspy.signals.spectrum import Spectrum
from hyperspy.signals.eds import EDSSpectrum
from hyperspy.signals.image import Image
from hyperspy.misc.eds.elements import elements as elements_db
import hyperspy.axes
from hyperspy.gui.egerton_quantification import SpikesRemoval
from hyperspy.decorators import only_interactive
from hyperspy.gui.eds import SEMParametersUI
from hyperspy.defaults_parser import preferences
import hyperspy.gui.messages as messagesui
from hyperspy.misc.progressbar import progressbar
from hyperspy.components.power_law import PowerLaw
from hyperspy.io import load



class EDSSEMSpectrum(EDSSpectrum):
    
    def __init__(self, *args, **kwards):
        EDSSpectrum.__init__(self, *args, **kwards)
        # Attributes defaults        
        if hasattr(self.mapped_parameters, 'SEM.EDS') == False:            
            self._load_microscope_param()
        

    def get_calibration_from(self, ref, nb_pix=1):
        """Copy the calibration and all metadata of a reference.

        Primary use: To add a calibration to ripple file from INCA 
        software
                
        Parameters
        ----------
        ref : signal
            The reference contains the calibration in its 
            mapped_parameters 
        nb_pix : int
            The live time (real time corrected from the "dead time")
            is divided by the number of pixel (spectrums), giving an 
            average live time.          
        """
        
        
        self.original_parameters = ref.original_parameters
        # Setup the axes_manager
        ax_m = self.axes_manager.signal_axes[0]
        if hasattr(self.original_parameters, 'CHOFFSET'):
            ax_m.scale = ref.original_parameters.CHOFFSET
        if hasattr(self.original_parameters, 'OFFSET'):
            ax_m.offset = ref.original_parameters.OFFSET
        if hasattr(self.original_parameters, 'XUNITS'):            
            ax_m.units = ref.original_parameters.XUNITS
            if hasattr(self.original_parameters, 'CHOFFSET'):      
                if self.original_parameters.XUNITS == 'keV':
                    ax_m.scale = ref.original_parameters.CHOFFSET / 1000
         
        
        # Setup mapped_parameters
        if hasattr(ref.mapped_parameters, 'SEM'):
            mp_ref = ref.mapped_parameters.SEM 
        elif hasattr(ref.mapped_parameters, 'TEM'):
            mp_ref = ref.mapped_parameters.TEM
        else:
            raise ValueError("The reference has no mapped_parameters.TEM"
            "\n nor mapped_parameters.SEM ")
            
        mp = self.mapped_parameters
                 
        if hasattr(mp_ref, 'tilt_stage'):
            mp.SEM.tilt_stage = mp_ref.tilt_stage
        if hasattr(mp_ref, 'beam_energy'):
            mp.SEM.beam_energy = mp_ref.beam_energy  
        if hasattr(mp_ref.EDS, 'live_time'):
            mp.SEM.EDS.live_time = mp_ref.EDS.live_time / nb_pix
        if hasattr(mp_ref.EDS, 'azimuth_angle'):
            mp.SEM.EDS.azimuth_angle = mp_ref.EDS.azimuth_angle
        if hasattr(mp_ref.EDS, 'elevation_angle'):
            mp.SEM.EDS.elevation_angle = mp_ref.EDS.elevation_angle
            
    def _load_microscope_param(self): 
        """Transfer mapped_parameters.TEM to mapped_parameters.SEM,
            defined default value"""      
         
        mp = self.mapped_parameters                     
        if mp.has_item('SEM') is False:
            mp.add_node('SEM')
        if mp.has_item('SEM.EDS') is False:
            mp.SEM.add_node('EDS')        
               
        #Set to value to default 
        mp.SEM.EDS.tilt_stage = 0
        mp.SEM.EDS.elevation_angle = 35
        mp.SEM.EDS.energy_resolution_MnKa = 130
        mp.SEM.EDS.azimuth_angle = 90
        
        #Transfer    
        if hasattr(mp,'TEM'):
            if hasattr(mp.TEM, 'tilt_stage'):
                mp.SEM.tilt_stage = mp.TEM.tilt_stage
            if hasattr(mp.TEM, 'beam_energy'):
                mp.SEM.beam_energy = mp.TEM.beam_energy  
            if hasattr(mp.TEM.EDS, 'live_time'):
                mp.SEM.EDS.live_time = mp.TEM.EDS.live_time
            if hasattr(mp.TEM.EDS, 'azimuth_angle'):
                mp.SEM.EDS.azimuth_angle = mp.TEM.EDS.azimuth_angle 
            if hasattr(mp.TEM.EDS, 'elevation_angle'):
                mp.SEM.EDS.elevation_angle = mp.TEM.EDS.elevation_angle
               
        
                
               
    def set_microscope_parameters(self, beam_energy=None, live_time=None,
     tilt_stage=None, azimuth_angle=None, elevation_angle=None,
     energy_resolution_MnKa=None):
        """Set the microscope parameters that are necessary to quantify
        the spectrum.
        
        If not all of them are defined, raises in interactive mode 
        raises an UI item to fill the values
        
        Parameters
        ----------------
        beam_energy: float
            The energy of the electron beam in keV
            
        live_time : float
            In second
            
        tilt_stage : float
            In degree
            
        azimuth_angle : float
            In degree
            
        elevation_angle : float
            In degree  
            
        energy_resolution_MnKa : float
            In eV
                      
        """       
        mp_mic = self.mapped_parameters.SEM   
        
        if beam_energy is not None:
            mp_mic.beam_energy = beam_energy
        if live_time is not None:
            mp_mic.EDS.live_time = live_time
        if tilt_stage is not None:
            mp_mic.tilt_stage = tilt_stage
        if azimuth_angle is not None:
            mp_mic.EDS.azimuth_angle = azimuth_angle
        if tilt_stage is not None:
            mp_mic.EDS.elevation_angle = elevation_angle
        if energy_resolution_MnKa is not None:
            mp_mic.EDS.energy_resolution_MnKa  = energy_resolution_MnKa
        
        self._are_microscope_parameters_missing()
                
            
    @only_interactive            
    def _set_microscope_parameters(self):       
        
        
        tem_par = SEMParametersUI()            
        mapping = {
        'SEM.beam_energy' : 'tem_par.beam_energy',        
        'SEM.tilt_stage' : 'tem_par.tilt_stage',
        'SEM.EDS.live_time' : 'tem_par.live_time',
        'SEM.EDS.azimuth_angle' : 'tem_par.azimuth_angle',
        'SEM.EDS.elevation_angle' : 'tem_par.elevation_angle',
        'SEM.EDS.energy_resolution_MnKa' : 'tem_par.energy_resolution_MnKa',}
       
        for key, value in mapping.iteritems():
            if self.mapped_parameters.has_item(key):
                exec('%s = self.mapped_parameters.%s' % (value, key))
        tem_par.edit_traits()
                  
        mapping = {
        'SEM.beam_energy' : tem_par.beam_energy,        
        'SEM.tilt_stage' : tem_par.tilt_stage,
        'SEM.EDS.live_time' : tem_par.live_time,
        'SEM.EDS.azimuth_angle' : tem_par.azimuth_angle,
        'SEM.EDS.elevation_angle' : tem_par.elevation_angle,
        'SEM.EDS.energy_resolution_MnKa' : tem_par.energy_resolution_MnKa,}
        
        
        for key, value in mapping.iteritems():
            if value != t.Undefined:
                exec('self.mapped_parameters.%s = %s' % (key, value))
        self._are_microscope_parameters_missing()
     
    def _are_microscope_parameters_missing(self):
        """Check if the EDS parameters necessary for quantification
        are defined in mapped_parameters. Raise in interactive mode 
         an UI item to fill or change the values"""
        
        must_exist = (
            'SEM.beam_energy',            
            'SEM.tilt_stage',
            'SEM.EDS.live_time', 
            'SEM.EDS.azimuth_angle',
            'SEM.EDS.elevation_angle',)
        
        missing_parameters = []
        for item in must_exist:
            exists = self.mapped_parameters.has_item(item)
            if exists is False:
                missing_parameters.append(item)
        
        if preferences.General.interactive is True:
            par_str = "The following parameters are missing:\n"
            for par in missing_parameters:
                par_str += '%s\n' % par
            par_str += 'Please set them in the following wizard'
            is_ok = messagesui.information(par_str)
            if is_ok:
                self._set_microscope_parameters()                
            else:
                return True
        else:
            return True
                
           
                        
                      
#    def build_SI_from_substracted_zl(self,ch, taper_nch = 20):
#        """Modify the SI to have fit with a smoothly decaying ZL
#        
#        Parameters
#        ----------
#        ch : int
#            channel index to start the ZL decay to 0
#        taper_nch : int
#            number of channels in which the ZL will decay to 0 from `ch`
#        """
#        sp = copy.deepcopy(self)
#        dc = self.zl_substracted.data_cube.copy()
#        dc[0:ch,:,:] *= 0
#        for i in xrange(dc.shape[1]):
#            for j in xrange(dc.shape[2]):
#                dc[ch:ch+taper_nch,i,j] *= np.hanning(2 * taper_nch)[:taper_nch]
#        sp.zl_substracted.data_cube = dc.copy()
#        dc += self.zero_loss.data_cube
#        sp.data_cube = dc.copy()
#        return sp
#        

#        
#    def correct_dual_camera_step(self, show_lev = False, mean_interval = 3, 
#                                 pca_interval = 20, pcs = 2, 
#                                 normalize_poissonian_noise = False):
#        """Correct the gain difference in a dual camera using PCA.
#        
#        Parameters
#        ----------
#        show_lev : boolen
#            Plot PCA lev
#        mean_interval : int
#        pca_interval : int
#        pcs : int
#            number of principal components
#        normalize_poissonian_noise : bool
#        """ 
#        # The step is between pixels 1023 and 1024
#        pw = pca_interval
#        mw = mean_interval
#        s = copy.deepcopy(self)
#        s.energy_crop(1023-pw, 1023 + pw)
#        s.decomposition(normalize_poissonian_noise)
#        if show_lev:
#            s.plot_lev()
#            pcs = int(raw_input('Number of principal components? '))
#        sc = s.get_decomposition_model(pcs)
#        step = sc.data_cube[(pw-mw):(pw+1),:,:].mean(0) - \
#        sc.data_cube[(pw+1):(pw+1+mw),:,:].mean(0)
#        self.data_cube[1024:,:,:] += step.reshape((1, step.shape[0], 
#        step.shape[1]))
#        self._replot()
#        return step
