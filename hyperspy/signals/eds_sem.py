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
import scipy.optimize
import matplotlib.pyplot as plt
import traits.api as t
import math


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
from hyperspy.misc.eds.FWHM import FWHM_eds
import hyperspy.components as components



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
            
        self._set_microscope_parameters()
            
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
        are defined in mapped_parameters. If not, in interactive mode 
        raises an UI item to fill the values"""       
        
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
            
    def link_standard(self, std_folder, std_file_extension='msa'):
        """
        Seek for standard spectra (spectrum recorded on known composition
        sample) in the std_file folder and link them to the analyzed 
        elements of 'mapped_parameters.Sample.elements'. A standard spectrum 
        is linked if its file name contains the elements name. 
        
        Store the standard spectra in 'mapped_parameters.Sample.standard_spec'

        
        Parameters
        ----------------
        std_folder: path name
            The path of the folder containing the standard file.
            
        std_file_extension: extension name
            The name of the standard file extension.
            
        See also
        --------
        set_elements, add_elements
        
        """
        
        if not hasattr(self.mapped_parameters, 'Sample') :
            raise ValueError("Add elements first, see 'set_elements'")
        if not hasattr(self.mapped_parameters.Sample, 'elements'):
            raise ValueError("Add elements first, see 'set_elements'")
        

        std_tot = load(std_folder+"//*."+std_file_extension,signal_type = 'EDS_SEM')
        mp = self.mapped_parameters        
        mp.Sample.standard_spec = []
        for element in mp.Sample.elements:            
            for std in std_tot:    
                mp_std = std.mapped_parameters
                if element in mp_std.original_filename:
                    print("Standard file for %s : %s" % (element, mp_std.original_filename))
                    mp_std.title = element+"_std"
                    mp.Sample.standard_spec.append(std)
        
    def top_hat(self, line_energy, width_windows = 1):
        """
        Substact the background with a top hat filter. The width of the
        lobs are defined with the width of the peak at the line_energy.
        
        Parameters
        ----------------
        line_energy: float
            The energy in keV used to set the lob width calculate with
            FHWM_eds.
            
        width_windows: float
            The width of the windows on which is applied the top_hat. 
            By default set to 1, which is equivalent to the size of the 
            filtering object. 
                       
        Notes
        -----
        See the textbook of Goldstein et al., Plenum publisher, 
        third edition p 399
        
        """
        offset = self.axes_manager.signal_axes[0].offset
        FWHM_MnKa = self.mapped_parameters.SEM.EDS.energy_resolution_MnKa
        line_FWHM = FWHM_eds(FWHM_MnKa, line_energy)              
        det = width_windows*line_FWHM
        scale_s = self.axes_manager.signal_axes[0].scale
        olob = int(round(line_FWHM/scale_s/2)*2)
        g = []
        for lob in range(-olob,olob):
            if abs(lob) > olob/2:
                g.append(-1./olob)
            else:
                g.append(1./(olob+1))    
        g = np.array(g)
        
        bornA = [int(round((line_energy-det-offset)/scale_s)),\
        int(round((line_energy+det-offset)/scale_s))]
        
        a = []
        for i in range(bornA[0],bornA[1]):
            a.append(self.data[...,i-olob:i+olob])
        a = np.array(a)
        
        dim = len(self.data.shape)
        spec_th = EDSSEMSpectrum({'data' : np.rollaxis(a.dot(g),0,dim)})
        
        #spec_th.get_calibration_from(self)
        return spec_th
        
    def k_ratio(self):
        """
        Calculate the k-ratios by least-square fitting of the standard 
        sepectrum after background substraction with a top hat filtering
        
        Return a display of the resutls and store them in 
        'mapped_parameters.Sample.k_ratios'
        
        See also
        --------
        set_elements, link_standard, top_hat 
        
        """
        
        if not hasattr(self.mapped_parameters, 'Sample') :
            raise ValueError("Add elements first, see 'set_elements'")
        if not hasattr(self.mapped_parameters.Sample, 'elements'):
            raise ValueError("Add elements first, see 'set_elements'")
        if not hasattr(self.mapped_parameters.Sample, 'standard_spec') :
            raise ValueError("Add Standard, see 'link_standard'")
        
        
        from hyperspy.hspy import create_model        
        width_windows=0.75 
        mp = self.mapped_parameters         
        mp.Sample.kratios = []   
        i=0    
        for Xray_line in mp.Sample.Xray_lines:
            element = Xray_line[:-3]
            line = Xray_line[-2:] 
            std = mp.Sample.standard_spec[i]
            mp_std = std.mapped_parameters
            line_energy = elements_db[element]['Xray_energy'][line]
            diff_ltime = mp.SEM.EDS.live_time/mp_std.SEM.EDS.live_time
            #Fit with least square
            m = create_model(self.top_hat(line_energy,width_windows))
            fp = components.ScalableFixedPattern(std.top_hat(line_energy,width_windows))
            fp.set_parameters_not_free(['offset','xscale','shift'])
            m.append(fp)
            m.multifit(fitter='leastsq')
            #Store k-ratio
            if (self.axes_manager.navigation_dimension == 0):
                kratio = fp.yscale.value/diff_ltime
                print("k-ratio of %s : %s" % (Xray_line, kratio))
            else:
                if (self.axes_manager.navigation_dimension == 3):
                    kratio = self.to_image(1)[1].deepcopy()
                else:
                    kratio = self.to_image()[1].deepcopy()
                kratio.data = fp.yscale.as_signal().data/diff_ltime
                kratio.mapped_parameters.title = 'k-ratio ' + Xray_line
                kratio.plot(None,False)
                      
            mp.Sample.kratios.append(kratio)
            i += 1
       
        
    #def k_ratio(self):
        #mp = self.mapped_parameters  
        #i=0
        #width_windows=0.5
        #kratios = []        
        #for Xray_line in mp.Sample.Xray_lines:
            #element = Xray_line[:-3]
            #line = Xray_line[-2:] 
            #std = mp.Sample.standard_spec[i]
            #mp_std = std.mapped_parameters
            #def residuals(a):
                #return -self.top_hat(line_energy,width_windows).data+a*std.top_hat(line_energy,width_windows).data
            #line_energy = elements_db[element]['Xray_energy'][line]
            #diff_ltime = mp.SEM.EDS.live_time/mp_std.SEM.EDS.live_time            
            #kratio = scipy.optimize.leastsq(residuals,[0.5])[0][0]/diff_ltime
            #print("k-ratio of %s : %s" % (Xray_line, kratio))            
            #kratios.append(kratio)
            #i += 1
        #mp.Sample.kratios = kratios
        
        
        
                
