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

import traits.api as t

from hyperspy.signals.eds import EDSSpectrum
from hyperspy.decorators import only_interactive
from hyperspy.gui.eds import TEMParametersUI
from hyperspy.defaults_parser import preferences
import hyperspy.gui.messages as messagesui

#TEM spectrum is just a copy of the basic function of SEM spectrum.
class EDSTEMSpectrum(EDSSpectrum):
    _signal_type = "EDS_TEM"    
    
    def __init__(self, *args, **kwards):
        EDSSpectrum.__init__(self, *args, **kwards)
        # Attributes defaults
        if hasattr(self.mapped_parameters, 'TEM.EDS') == False: 
            self._load_from_SEM_param()
        self._set_default_param()
        
    def _load_from_SEM_param(self): 
        """Transfer mapped_parameters.SEM to mapped_parameters.TEM"""      
         
        mp = self.mapped_parameters                     
        if mp.has_item('TEM') is False:
            mp.add_node('TEM')
        if mp.has_item('TEM.EDS') is False:
            mp.TEM.add_node('EDS') 
        mp.signal_type = 'EDS_TEM'
        
        #Transfer    
        if hasattr(mp,'SEM'):
            mp.TEM = mp.SEM
            del mp.__dict__['SEM']
        
    def _set_default_param(self): 
        """Set to value to default (defined in preferences)
        """    
        
        mp = self.mapped_parameters                     
        if mp.has_item('TEM') is False:
            mp.add_node('TEM')
        if mp.has_item('TEM.EDS') is False:
            mp.TEM.add_node('EDS')
            
        mp.signal_type = 'EDS_TEM'
           
        mp = self.mapped_parameters
        if hasattr(mp.TEM, 'tilt_stage') is False:
            mp.TEM.tilt_stage = preferences.EDS.eds_tilt_stage
        if hasattr(mp.TEM.EDS, 'elevation_angle') is False:
            mp.TEM.EDS.elevation_angle = preferences.EDS.eds_detector_elevation
        if hasattr(mp.TEM.EDS, 'energy_resolution_MnKa') is False:
            mp.TEM.EDS.energy_resolution_MnKa = preferences.EDS.eds_mn_ka
        if hasattr(mp.TEM.EDS, 'azimuth_angle') is False:
            mp.TEM.EDS.azimuth_angle = preferences.EDS.eds_detector_azimuth  
        
               
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
        mp_mic = self.mapped_parameters.TEM   
        
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
        
        self._set_microscope_parameters()
                
            
    @only_interactive            
    def _set_microscope_parameters(self):
        #mp = self.mapped_parameters                     
        #if mp.has_item('TEM') is False:
            #mp.add_node('TEM')
        #if mp.has_item('TEM.EDS') is False:
            #mp.TEM.add_node('EDS')         
        tem_par = TEMParametersUI() 
        mapping = {
        'TEM.beam_energy' : 'tem_par.beam_energy',        
        'TEM.tilt_stage' : 'tem_par.tilt_stage',
        'TEM.EDS.live_time' : 'tem_par.live_time',
        'TEM.EDS.azimuth_angle' : 'tem_par.azimuth_angle',
        'TEM.EDS.elevation_angle' : 'tem_par.elevation_angle',
        'TEM.EDS.energy_resolution_MnKa' : 'tem_par.energy_resolution_MnKa',}
        for key, value in mapping.iteritems():
            if self.mapped_parameters.has_item(key):
                exec('%s = self.mapped_parameters.%s' % (value, key))
        tem_par.edit_traits()
        
        mapping = {
        'TEM.beam_energy' : tem_par.beam_energy,        
        'TEM.tilt_stage' : tem_par.tilt_stage,
        'TEM.EDS.live_time' : tem_par.live_time,
        'TEM.EDS.azimuth_angle' : tem_par.azimuth_angle,
        'TEM.EDS.elevation_angle' : tem_par.elevation_angle,
        'TEM.EDS.energy_resolution_MnKa' : tem_par.elevation_angle,}
        
        for key, value in mapping.iteritems():
            if value != t.Undefined:
                exec('self.mapped_parameters.%s = %s' % (key, value))
        self._are_microscope_parameters_missing()
     
    def _are_microscope_parameters_missing(self):
        """Check if the EDS parameters necessary for quantification
        are defined in mapped_parameters. Raise in interactive mode 
         an UI item to fill or cahnge the values"""        
        must_exist = (
            'TEM.beam_energy',             
            'TEM.EDS.live_time',) 

        missing_parameters = []
        for item in must_exist:
            exists = self.mapped_parameters.has_item(item)
            if exists is False:
                missing_parameters.append(item)
        if missing_parameters: 
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
        else:
            return False          

                
                
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
        
        
        self.original_parameters = ref.original_parameters.deepcopy()
        # Setup the axes_manager
        ax_m = self.axes_manager.signal_axes[0]
        ax_ref = ref.axes_manager.signal_axes[0]
        ax_m.scale = ax_ref.scale
        ax_m.units = ax_ref.units 
        ax_m.offset = ax_ref.offset
        
        #if hasattr(self.original_parameters, 'CHOFFSET'):
            #ax_m.scale = ref.original_parameters.CHOFFSET
        #if hasattr(self.original_parameters, 'OFFSET'):
            #ax_m.offset = ref.original_parameters.OFFSET
        #if hasattr(self.original_parameters, 'XUNITS'):            
            #ax_m.units = ref.original_parameters.XUNITS
            #if hasattr(self.original_parameters, 'CHOFFSET'):      
                #if self.original_parameters.XUNITS == 'keV':
                    #ax_m.scale = ref.original_parameters.CHOFFSET / 1000
         
        
        # Setup mapped_parameters
        if hasattr(ref.mapped_parameters, 'TEM'):
            mp_ref = ref.mapped_parameters.TEM 
        elif hasattr(ref.mapped_parameters, 'SEM'):
            mp_ref = ref.mapped_parameters.SEM
        else:
            raise ValueError("The reference has no mapped_parameters.TEM"
            "\n nor mapped_parameters.SEM ")
            
        mp = self.mapped_parameters
        
        mp.TEM = mp_ref.deepcopy()
        
        #if hasattr(mp_ref, 'tilt_stage'):
            #mp.SEM.tilt_stage = mp_ref.tilt_stage
        #if hasattr(mp_ref, 'beam_energy'):
            #mp.SEM.beam_energy = mp_ref.beam_energy
        #if hasattr(mp_ref.EDS, 'energy_resolution_MnKa'):
            #mp.SEM.EDS.energy_resolution_MnKa = mp_ref.EDS.energy_resolution_MnKa
        #if hasattr(mp_ref.EDS, 'azimuth_angle'):
            #mp.SEM.EDS.azimuth_angle = mp_ref.EDS.azimuth_angle
        #if hasattr(mp_ref.EDS, 'elevation_angle'):
            #mp.SEM.EDS.elevation_angle = mp_ref.EDS.elevation_angle
        
        if hasattr(mp_ref.EDS, 'live_time'):
            mp.TEM.EDS.live_time = mp_ref.EDS.live_time / nb_pix
