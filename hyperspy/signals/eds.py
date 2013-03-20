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
from hyperspy.signals.image import Image
from hyperspy.misc.eds.elements import elements as elements_db
import hyperspy.axes
from hyperspy.gui.egerton_quantification import SpikesRemoval
from hyperspy.decorators import only_interactive
from hyperspy.gui.eds import TEMParametersUI
from hyperspy.gui.eds import SEMParametersUI
from hyperspy.defaults_parser import preferences
import hyperspy.gui.messages as messagesui
from hyperspy.misc.progressbar import progressbar
from hyperspy.components.power_law import PowerLaw
from hyperspy.io import load


class EDSSpectrum(Spectrum):
    
    def __init__(self, *args, **kwards):
        Spectrum.__init__(self, *args, **kwards)
        # Attributes defaults
        #self.subshells = set()
        self.elements = set()
        #self.edges = list()
        if hasattr(self.mapped_parameters, 'SEM.EDS') == False and \
        hasattr(self.mapped_parameters, 'TEM.EDS') == False:            
            self._load_microscope_param()
        if hasattr(self.mapped_parameters, 'Sample') and \
        hasattr(self.mapped_parameters.Sample, 'elements'):
            print('Elemental composition read from file')
            self.add_elements(self.mapped_parameters.Sample.elements)

    def calibrate_on(self, ref, nb_pix=1):
        """Copy the calibration and all metadata of a reference.

        Primary use: To add a calibration to ripple file from INCA software
                
        Parameters
        ----------
        ref : signal
            The ref contains in its original_parameters the calibration
        nb_pix : int
            The live time (real time corrected from the "dead time")
            is divided by the number of pixel (spectrums), giving an average live time.          
        """
        
        #ref = load(filename_ref)
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
        mp = self.mapped_parameters
        microscope = 'TEM'  
        if  mp.signal_type == 'EDS_SEM':
            microscope = 'SEM'             
        if mp.has_item(microscope) is False:
            mp.add_node(microscope)            
          
        if  mp.signal_type == 'EDS_SEM':            
            mp_mic = self.mapped_parameters.SEM      
        else:
            mp_mic = self.mapped_parameters.TEM   
        if mp.has_item(microscope+'.EDS') is False:
            mp_mic.add_node('EDS') 
               
        if hasattr(self.original_parameters, 'LIVETIME'):
            mp_mic.live_time = ref.original_parameters.LIVETIME / nb_pix
        if hasattr(self.original_parameters, 'XTILTSTGE'):
            mp_mic.tilt_stage = ref.original_parameters.XTILTSTGE
        if hasattr(self.original_parameters, 'BEAMKV'):
            mp_mic.beam_energy = ref.original_parameters.BEAMKV        
        if hasattr(self.original_parameters, 'AZIMANGLE'):
            mp_mic.EDS.azimuth_angle = ref.original_parameters.AZIMANGLE
        if hasattr(self.original_parameters, 'ELEVANGLE'):
            mp_mic.EDS.elevation_angle = ref.original_parameters.ELEVANGLE
            
    def _load_microscope_param(self): 
        """load the important parameter of from original_parameters"""      
         
        mp = self.mapped_parameters
        if hasattr(mp, 'signal_type') == False: 
            mp.signal_type = 'EDS'
            
        microscope = 'TEM'  
        if  mp.signal_type == 'EDS_SEM':
            microscope = 'SEM'             
        if mp.has_item(microscope) is False:
            mp.add_node(microscope)
                        
         
        if  mp.signal_type == 'EDS_SEM':            
            mp_mic = self.mapped_parameters.SEM  
        else:
            mp_mic = self.mapped_parameters.TEM  
        if mp.has_item(microscope+'.EDS') is False:
            mp_mic.add_node('EDS') 
               
        if hasattr(self.original_parameters, 'LIVETIME'):
            mp_mic.live_time = self.original_parameters.LIVETIME
        if hasattr(self.original_parameters, 'XTILTSTGE'):
            mp_mic.tilt_stage = self.original_parameters.XTILTSTGE
        if hasattr(self.original_parameters, 'BEAMKV'):
            mp_mic.beam_energy = self.original_parameters.BEAMKV        
        if hasattr(self.original_parameters, 'AZIMANGLE'):
            mp_mic.EDS.azimuth_angle = self.original_parameters.AZIMANGLE
        if hasattr(self.original_parameters, 'ELEVANGLE'):
            mp_mic.EDS.elevation_angle = self.original_parameters.ELEVANGLE     

                    

    def add_elements(self, elements):
        """Declare the elements present in the sample.
        
        
        Parameters
        ----------
        elements : tuple of strings
            The symbol of the elements.        
            
        Examples
        --------
        
        >>> s = signals.EDSSpectrum({'data' : np.arange(1024)})
        >>> s.add_elements(('C', 'O'))        
        
        """
        
        for element in elements:
            
            if element in elements_db:               
                self.elements.add(element)
            else:
                print(
                    "%s is not a valid symbol of an element" % element)
        if not hasattr(self.mapped_parameters, 'Sample'):
            self.mapped_parameters.add_node('Sample')
        self.mapped_parameters.Sample.elements = list(self.elements)
        
               
    def set_microscope_parameters(self, beam_energy=None, live_time=None,
     tilt_stage=None, azimuth_angle=None, elevation_angle=None):
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
                      
        """      
           
        if  self.mapped_parameters.signal_type == 'EDS_SEM':            
            mp_mic = self.mapped_parameters.SEM     
        else: 
            mp_mic = self.mapped_parameters.TEM  
        
        if beam_energy is not None:
            mp_mic.beam_energy = beam_energy
        if live_time is not None:
            mp_mic.convergence_angle = live_time
        if tilt_stage is not None:
            mp_mic.tilt_stage = tilt_stage
        if azimuth_angle is not None:
            mp_mic.EDS.azimuth_angle = azimuth_angle
        if tilt_stage is not None:
            mp_mic.EDS.elevation_angle = elevation_angle       
        
        self._are_microscope_parameters_missing()
                
            
    @only_interactive            
    def _set_microscope_parameters(self):
        #if self.mapped_parameters.has_item('TEM') is False:
         #   self.mapped_parameters.add_node('TEM')
        #if self.mapped_parameters.has_item('TEM.EELS') is False:
        #    self.mapped_parameters.TEM.add_node('EELS')
        
        if self.mapped_parameters.signal_type == 'EDS_SEM':
            tem_par = SEMParametersUI()            
            mapping = {
            'SEM.beam_energy' : 'tem_par.beam_energy',
            'SEM.live_time' : 'tem_par.live_time',
            'SEM.tilt_stage' : 'tem_par.tilt_stage',
            'SEM.EDS.azimuth_angle' : 'tem_par.azimuth_angle',
            'SEM.EDS.elevation_angle' : 'tem_par.elevation_angle',}
        else:
            tem_par = TEMParametersUI() 
            mapping = {
            'TEM.beam_energy' : 'tem_par.beam_energy',
            'TEM.live_time' : 'tem_par.live_time',
            'TEM.tilt_stage' : 'tem_par.tilt_stage',
            'TEM.EDS.azimuth_angle' : 'tem_par.azimuth_angle',
            'TEM.EDS.elevation_angle' : 'tem_par.elevation_angle',}
        for key, value in mapping.iteritems():
            if self.mapped_parameters.has_item(key):
                exec('%s = self.mapped_parameters.%s' % (value, key))
        tem_par.edit_traits()
        if self.mapped_parameters.signal_type == 'EDS_SEM':            
            mapping = {
            'SEM.beam_energy' : tem_par.beam_energy,
            'SEM.live_time' : tem_par.live_time,
            'SEM.tilt_stage' : tem_par.tilt_stage,
            'SEM.EDS.azimuth_angle' : tem_par.azimuth_angle,
            'SEM.EDS.elevation_angle' : tem_par.elevation_angle,}
        else:
            mapping = {
            'TEM.beam_energy' : tem_par.beam_energy,
            'TEM.live_time' : tem_par.live_time,
            'TEM.tilt_stage' : tem_par.tilt_stage,
            'TEM.EDS.azimuth_angle' : tem_par.azimuth_angle,
            'TEM.EDS.elevation_angle' : tem_par.elevation_angle,}
        
        for key, value in mapping.iteritems():
            if value != t.Undefined:
                exec('self.mapped_parameters.%s = %s' % (key, value))
        self._are_microscope_parameters_missing()
     
    def _are_microscope_parameters_missing(self):
        """Check if the EDS parameters necessary for quantification
        are defined in mapped_parameters. If not, in interactive mode 
        raises an UI item to fill the values"""
        if self.mapped_parameters.signal_type == 'EDS_SEM':
            must_exist = (
                'SEM.beam_energy', 
                'SEM.live_time', 
                'SEM.tilt_stage',
                'SEM.EDS.azimuth_angle',
                'SEM.EDS.elevation_angle',)
        else:
            must_exist = (
                'TEM.beam_energy', 
                'TEM.live_time', 
                'TEM.tilt_stage', 
                'TEM.EDS.azimuth_angle',
                'TEM.EDS.elevation_angle',)
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
