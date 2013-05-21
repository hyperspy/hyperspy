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
#import scipy.interpolate
#import scipy.optimize
import matplotlib.pyplot as plt
import traits.api as t
import math
import codecs
import subprocess
import os

from hyperspy.signals.spectrum import Spectrum
from hyperspy.signals.eds import EDSSpectrum
from hyperspy.signals.image import Image
from hyperspy.misc.eds.elements import elements as elements_db
import hyperspy.axes
from hyperspy.gui.eds import SEMParametersUI
from hyperspy.defaults_parser import preferences
import hyperspy.gui.messages as messagesui
from hyperspy.decorators import only_interactive
from hyperspy.io import load
from hyperspy.misc.eds.FWHM import FWHM_eds
from hyperspy.misc.eds.TOA import TOA
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
        is linked if its file name contains the elements name. "C.msa",
        "Co.msa" but not "Co4.msa".
        
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
        #for element in mp.Sample.elements:
        for Xray_line in mp.Sample.Xray_lines:
            element = Xray_line[:-3]
            test_file_exist=False           
            for std in std_tot:    
                mp_std = std.mapped_parameters
                if element + "." in mp_std.original_filename:
                    test_file_exist=True
                    print("Standard file for %s : %s" % (element, mp_std.original_filename))
                    mp_std.title = element+"_std"
                    mp.Sample.standard_spec.append(std)
            if test_file_exist == False:
                print("\nStandard file for %s not found\n" % element)
        
    def top_hat(self, line_energy, width_windows = 1.):
        """
        Substact the background with a top hat filter. The width of the
        lobs are defined with the width of the peak at the line_energy.
        
        Parameters
        ----------------
        line_energy: float
            The energy in keV used to set the lob width calculate with
            FHWM_eds.
            
        width_windows: float or list(min,max)
            The width of the windows on which is applied the top_hat. 
            By default set to 1, which is equivalent to the size of the 
            filtering object. 
                       
        Notes
        -----
        See the textbook of Goldstein et al., Plenum publisher, 
        third edition p 399
        
        """
        offset = np.copy(self.axes_manager.signal_axes[0].offset)
        scale_s = np.copy(self.axes_manager.signal_axes[0].scale)
        FWHM_MnKa = self.mapped_parameters.SEM.EDS.energy_resolution_MnKa
        line_FWHM = FWHM_eds(FWHM_MnKa, line_energy) 
        if np.ndim(width_windows) == 0:            
            det = [width_windows*line_FWHM,width_windows*line_FWHM]
        else :
            det = width_windows
        
        olob = int(round(line_FWHM/scale_s/2)*2)
        g = []
        for lob in range(-olob,olob):
            if abs(lob) > olob/2:
                g.append(-1./olob)
            else:
                g.append(1./(olob+1))    
        g = np.array(g)
   
        bornA = [int(round((line_energy-det[0]-offset)/scale_s)),\
        int(round((line_energy+det[1]-offset)/scale_s))]
  
        data_s = []
        for i in range(bornA[0],bornA[1]):
            data_s.append(self.data[...,i-olob:i+olob].dot(g))
            #data_s.append(self.data[...,i-olob:i+olob])
        data_s = np.array(data_s)
 
        dim = len(self.data.shape)
        #spec_th = EDSSEMSpectrum(np.rollaxis(data_s.dot(g),0,dim))
        spec_th = EDSSEMSpectrum(np.rollaxis(data_s,0,dim))

        return spec_th
        
    def _get_kratio(self,Xray_lines,plot_result):
        """
        Calculate the k-ratio without deconvolution
        """
        from hyperspy.hspy import create_model        
        width_windows=0.75 
        mp = self.mapped_parameters  
        
        for Xray_line in Xray_lines :
            element = Xray_line[:-3]
            line = Xray_line[-2:]
            std = self.get_result(element,'standard_spec') 
            mp_std = std.mapped_parameters
            line_energy = elements_db[element]['Xray_energy'][line]
            diff_ltime = mp.SEM.EDS.live_time/mp_std.SEM.EDS.live_time
            #Fit with least square
            m = create_model(self.top_hat(line_energy,width_windows))
            fp = components.ScalableFixedPattern(std.top_hat(line_energy,width_windows))
            fp.set_parameters_not_free(['offset','xscale','shift'])
            m.append(fp)          
            m.multifit(fitter='leastsq') 
            #store k-ratio
            if (self.axes_manager.navigation_dimension == 0):
                self._set_result( Xray_line, 'kratios',\
                    fp.yscale.value/diff_ltime, plot_result)
            else:
                self._set_result( Xray_line, 'kratios',\
                    fp.yscale.as_signal().data/diff_ltime, plot_result)
            
    
        
    def get_kratio(self,deconvolution=None,plot_result=True):
        
        """
        Calculate the k-ratios by least-square fitting of the standard 
        sepectrum after background substraction with a top hat filtering
        
        Return a display of the resutls and store them in 
        'mapped_parameters.Sample.k_ratios'
        
        Parameters
        ----------
        plot_result : bool
            If True (default option), plot the k-ratio map.
        
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

        mp = self.mapped_parameters         
        mp.Sample.kratios = list(np.zeros(len(mp.Sample.Xray_lines)))
        Xray_lines = list(mp.Sample.Xray_lines)
        
        if deconvolution is not None: 
            for deconvo in deconvolution:
                print("1")
                self._deconvolve_kratio(deconvo[0],deconvo[1],deconvo[2],plot_result)
                print("2")
                for Xray_line in deconvo[0]:
                    Xray_lines.remove(Xray_line)
        if len(Xray_lines) > 0:     
            self._get_kratio(Xray_lines,plot_result)
    
    def _deconvolve_kratio(self,Xray_lines,elements,width_energy,\
        plot_result=True):
        """
        Calculate the k-ratio with deconvolution
        """
        
        from hyperspy.hspy import create_model 
        line_energy = np.mean(width_energy)
        width_windows=[line_energy-width_energy[0],width_energy[1]-line_energy]
        
        m = create_model(self.top_hat(line_energy, width_windows))
        mp = self.mapped_parameters 

        if (self.axes_manager.navigation_dimension == 3):
            axes_kratio = self.to_image(1)[1].axes_manager 
        else:              
            axes_kratio = self.to_image()[1].axes_manager        
        diff_ltime =[]
        fps = []
        for element in elements:
            std = self.get_result(element,'standard_spec')
            fp = components.ScalableFixedPattern(std.top_hat(line_energy,width_windows))
            fp.set_parameters_not_free(['offset','xscale','shift'])
            fps.append(fp)    
            m.append(fps[-1])
            diff_ltime.append(mp.SEM.EDS.live_time/std.mapped_parameters.SEM.EDS.live_time)
        m.multifit(fitter='leastsq')
        i=0
        for Xray_line in Xray_lines:
            if (self.axes_manager.navigation_dimension == 0):
                self._set_result( Xray_line, 'kratios',\
                    fps[i].yscale.value/diff_ltime[i], plot_result)
            else:
                self._set_result( Xray_line, 'kratios',\
                    fps[i].yscale.as_signal().data/diff_ltime[i], plot_result)
            i += 1

    def deconvolve_intensity(self,width_windows='all',plot_result=True):
        """
        Calculate the intensity by fitting standard spectra to the spectrum.
        
        Deconvoluted intensity is thus obtained compared to 
        get_intensity_map
        
        Needs standard to be set
        
        Parameters
        ----------
        
        width_windows: 'all' | [min energy, max energy]
            Set the energy windows in which the fit is done. If 'all'
            (default option), the whole energy range is used.
            
        plot_result : bool
            If True (default option), plot the intensity maps.
            
        See also
        --------
        
        set_elements, link_standard, get_intensity_map
        
        
        """
        from hyperspy.hspy import create_model 
        m = create_model(self)
        mp = self.mapped_parameters 
                
        elements = mp.Sample.elements
       
        fps = []
        for element in elements:
            std = self.get_result(element,'standard_spec')
            fp = components.ScalableFixedPattern(std)
            fp.set_parameters_not_free(['offset','xscale','shift'])
            fps.append(fp)    
            m.append(fps[-1])
        if width_windows != 'all':
            m.set_signal_range(width_windows[0],width_windows[1])
        m.multifit(fitter='leastsq')
        mp.Sample.intensities = list(np.zeros(len(elements)))
        i=0
        for element in elements:
            if (self.axes_manager.navigation_dimension == 0):
                self._set_result( element, 'intensities',\
                    fps[i].yscale.value, plot_result)
                if plot_result and i == 0:
                    m.plot()
                    plt.title('Fitted standard') 
            else:
                self._set_result( element, 'intensities',\
                    fps[i].yscale.as_signal().data, plot_result)
            i += 1
            
    def check_kratio(self,Xray_lines,width_energy='auto'):
        """
        Plot the spectrum, the standard spectra and the sum of the sectra.
       
        Works only for spectrum.
        
        Parameters
        ----------        
        
        Xray_lines: list of string
            the X-ray lines to display.
        
        width_windows: 'auto' | [min energy, max energy]
            Set the width of the display windows If 'auto'
            (default option), the display is adjusted to the higest/lowest
            energy line.
        
        
        """
        if width_energy=='auto':
            line_energy =[]
            for Xray_line in Xray_lines:
                element = Xray_line[:-3]
                line = Xray_line[-2:] 
                line_energy.append(elements_db[element]['Xray_energy'][line])
            width_energy = [0,0]
            width_energy[0] = np.min(line_energy)-FWHM_eds(130,np.min(line_energy))*2
            width_energy[1] = np.max(line_energy)+FWHM_eds(130,np.max(line_energy))*2
                
        line_energy = np.mean(width_energy)
        width_windows=[line_energy-width_energy[0],width_energy[1]-line_energy]
            
        mp = self.mapped_parameters
        fig = plt.figure()
        self_data = self.top_hat(line_energy, width_windows).data
        plt.plot(self_data)
        leg_plot = ["Spec"]
        line_energies =[]
        intensities = []
        spec_sum = np.zeros(len(self.top_hat(line_energy, width_windows).data))
        for Xray_line in Xray_lines:
            element = Xray_line[:-3]
            line = Xray_line[-2:] 
            line_energy = elements_db[element]['Xray_energy'][line]
            width_windows=[line_energy-width_energy[0],width_energy[1]-line_energy]
            print("%s" % width_windows)
            leg_plot.append(Xray_line)
            std_spec = self.get_result(element,'standard_spec')
            kratio = self.get_result(Xray_line,'kratios').data[0]
            diff_ltime = mp.SEM.EDS.live_time/std_spec.mapped_parameters.SEM.EDS.live_time
            std_data = std_spec.top_hat(line_energy,width_windows).data*kratio*diff_ltime
            plt.plot(std_data)
            line_energies.append((line_energy-width_energy[0])/self.axes_manager[0].scale-self.axes_manager[0].offset)
            intensities.append(std_data[int(line_energies[-1])])
            spec_sum = spec_sum + std_data
        plt.plot(spec_sum)
        plt.plot(self_data-spec_sum)
        leg_plot.append("Sum")
        leg_plot.append("Residual")
        plt.legend(leg_plot)
        print("Tot residual: %s" % np.abs(self_data-spec_sum).sum())
        for i in range(len(line_energies)):
                plt.annotate(Xray_lines[i],xy = (line_energies[i],intensities[i]))
        fig.show()
        
    def save_result(self, result, filename, Xray_lines = 'all',extension = 'hdf5'):
        """
        Save the result in a file (results stored in 
        'mapped_parameters.Sample')
        
        Parameters
        ----------
        result : string {'kratios'|'quant'|'intensities'}
            The result to save
            
        filename:
            the file path + the file name. The result and the Xray-lines
            is added at the end.
        
        Xray_lines: list of string
            the X-ray lines to save. If 'all' (default), save all X-ray lines
        
        Extension: 
            the extension in which the result is saved.
            
        See also
        -------        
        get_kratio, deconvolove_intensity, quant
        
        
        """
        
        mp = self.mapped_parameters 
        if Xray_lines is 'all':
            if result == 'intensities':
                 Xray_lines = mp.Sample.elements
            else:
                Xray_lines = mp.Sample.Xray_lines
        for Xray_line in Xray_lines:
            if result == 'intensitiesS':
                res = self.intensity_map([Xray_line],plot_result=False)[0]
            else:
                res = self.get_result(Xray_line, result)
            if res.data.dtype == 'float64':
                a = 1
                res.change_dtype('float32')
                #res.change_dtype('uint32')
            res.save(filename=filename+"_"+result+"_"+Xray_line,extension = extension, overwrite = True)
        
    def get_result(self, Xray_line, result):
        """
        get the result of one X-ray line (result stored in 
        'mapped_parameters.Sample'):
        
         Parameters
        ----------        
        result : string {'kratios'|'quant'|'intensities'}
            The result to get
            
        Xray_lines: string
            the X-ray line to get.
        
        """
        mp = self.mapped_parameters 
        for res in mp.Sample[result]:
            if Xray_line in res.mapped_parameters.title:
                return res
        print("Didn't find it")
        
    def _set_result(self, Xray_line, result, data_res, plot_result):
        """
        Transform data_res (a result) into an image or a spectrum and
        stored it in 'mapped_parameters.Sample'
        """
        
        mp = self.mapped_parameters
        if len(Xray_line) < 3 :
            Xray_lines = mp.Sample.elements
        else:
            Xray_lines = mp.Sample.Xray_lines
                
        for j in range(len(Xray_lines)):
            if Xray_line == Xray_lines[j]:
                break
                
        #axes_res= self[...,0].axes_manager 
        #Should work but doesn't for 2D
         
        if (self.axes_manager.navigation_dimension == 3):
            axes_res = self.to_image(1)[1].axes_manager 
        elif (self.axes_manager.navigation_dimension == 2):
            axes_res = self.to_image()[1].axes_manager 
        else:              
            axes_res = self[...,0].axes_manager
        
                
        if self.axes_manager.navigation_dimension == 0:
            res_img = Spectrum(np.array([data_res]))
        else:
            res_img = Image(data_res)
        res_img.axes_manager = axes_res
        res_img.mapped_parameters.title = result + ' ' + Xray_line
        if plot_result:                
            if self.axes_manager.navigation_dimension == 0:
                #to be changed with new version
                print("%s of %s : %s" % (result, Xray_line, data_res))
            else:
                res_img.plot(False)
        else:
            print("%s of %s calculated" % (result, Xray_line))
        mp.Sample[result][j] = res_img   
    
    
    def quant(self,plot_result=True):        
        """
        Quantify using stratagem, a commercial software. A licence is needed.
        
        k-ratios needs to be calculated before. Return a display of the 
        results and store them in 'mapped_parameters.Sample.quants'
        
        Parameters
        ----------   
        plot_result: bool
            If true (default option), plot the result.
        
        See also
        --------
        set_elements, link_standard, top_hat, get_kratio
        
        """
        
        foldername = os.path.realpath("")+"//algo//v1_6Quant//"
        self._write_nbData_tsv(foldername + 'essai')
        self._write_donnee_tsv(foldername + 'essai')
        p = subprocess.Popen(foldername + 'Debug//essai.exe')
        p.wait()
        self._read_result_tsv(foldername + 'essai',plot_result)
        
    def _read_result_tsv(self,foldername,plot_result):
        encoding = 'latin-1'
        mp=self.mapped_parameters
        
        f = codecs.open(foldername+'//result.tsv', encoding = encoding,errors = 'replace') 
        dim = list(self.data.shape)
        a = []
        for Xray_line in mp.Sample.Xray_lines:
            a.append([])        
        for line in f.readlines():
            for i in range(len(mp.Sample.Xray_lines)):
                a[i].append(float(line.split()[3+i]))            
        f.close()
        i=0
        mp.Sample.quant = list(np.zeros(len(mp.Sample.Xray_lines)))
        for Xray_line in mp.Sample.Xray_lines:  
            if (self.axes_manager.navigation_dimension == 0):
                    data_quant=a[i][0]
            else:
                if (self.axes_manager.navigation_dimension == 3):                    
                    data_quant=np.array(a[i]).reshape((dim[2],dim[1],dim[0])).T
                else:
                    data_quant=np.array(a[i]).reshape((dim[1],dim[0])).T
            self._set_result( Xray_line, 'quant',data_quant, plot_result)        
            i += 1
        
    def _write_donnee_tsv(self, foldername):
        encoding = 'latin-1'
        mp=self.mapped_parameters
        Xray_lines = mp.Sample.Xray_lines
        f = codecs.open(foldername+'//donnee.tsv', 'w', encoding = encoding,errors = 'ignore') 
        dim = np.copy(self.axes_manager.navigation_shape).tolist()
        dim.reverse()
        if self.axes_manager.navigation_dimension == 0:
            f.write("1_1\r\n")
            for i in range(len(mp.Sample.Xray_lines)):
                f.write("%s\t" % mp.Sample.kratios[i])
        elif self.axes_manager.navigation_dimension == 2:
            for x in range(dim[1]):
                for y in range(dim[0]):
                    f.write("%s_%s\r\n" % (x+1,y+1))
                    for Xray_line in Xray_lines:
                        f.write("%s\t" % self.get_result(Xray_line,'kratios').data[y,x])
                    f.write('\r\n')
        elif self.axes_manager.navigation_dimension == 3:
            for x in range(dim[2]):
                for y in range(dim[1]):
                    f.write("%s_%s\r\n" % (x+1,y+1))
                    for z in range(dim[0]):
                        for Xray_line in Xray_lines:
                            f.write("%s\t" % self.get_result(Xray_line,'kratios').data[z,y,x])
                        f.write('\r\n')
        f.close()       
        
    
    def _write_nbData_tsv(self, foldername):
        encoding = 'latin-1'
        mp=self.mapped_parameters
        f = codecs.open(foldername+'//nbData.tsv', 'w', encoding = encoding,errors = 'ignore') 
        dim = np.copy(self.axes_manager.navigation_shape).tolist()
        dim.reverse()
        dim.append(1)
        dim.append(1)
        if dim[0] == 0:
            dim[0] =1
        f.write("nbpixel_x\t%s\r\n" % dim[0])
        f.write('nbpixel_y\t%s\r\n' % dim[1])
        f.write('nbpixel_z\t%s\r\n' % dim[2])
        #f.write('pixelsize_z\t%s' % self.axes_manager[0].scale*1000)
        f.write('pixelsize_z\t100\r\n')
        f.write('nblayermax\t5\r\n')
        f.write('Limitkratio0\t0.001\r\n')
        f.write('Limitcompsame\t0.01\r\n')
        f.write('Itermax\t49\r\n')
        f.write('\r\n')
        f.write('HV\t%s\r\n'% mp.SEM.beam_energy)
        f.write('TOA\t%s\r\n'% TOA(self))
        f.write('azimuth\t%s\r\n'% mp.SEM.EDS.azimuth_angle)
        f.write('tilt\t%s\r\n'% mp.SEM.tilt_stage)
        f.write('\r\n')
        f.write('nbelement\t%s\r\n'% mp.Sample.Xray_lines.shape[0])
        elements = 'Element'
        z_el = 'Z'
        line_el = 'line'
        for Xray_line in mp.Sample.Xray_lines:
            elements = elements + '\t' + Xray_line[:-3]
            z_el = z_el + '\t' + str(elements_db[Xray_line[:-3]]['Z'])
            if Xray_line[-2:] == 'Ka':
                line_el = line_el + '\t0'
            if Xray_line[-2:]== 'La':
                line_el = line_el + '\t1'
            if Xray_line[-2:] == 'Ma':
                line_el = line_el + '\t2'    
        f.write('%s\r\n'% elements)
        f.write('%s\r\n'% z_el)
        f.write('%s\r\n'% line_el)
        f.close()
