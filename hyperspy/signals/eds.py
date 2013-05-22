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
import matplotlib.pyplot as plt



from hyperspy.signals.spectrum import Spectrum
from hyperspy.signals.image import Image
from hyperspy.misc.eds.elements import elements as elements_db
from hyperspy.misc.eds.FWHM import FWHM_eds


class EDSSpectrum(Spectrum):
    
    def __init__(self, *args, **kwards):
        Spectrum.__init__(self, *args, **kwards)
        # Attributes defaults  
        #self.elements = set()
        #self.Xray_lines = set()
        #if hasattr(self.mapped_parameters, 'Sample') and \
        #hasattr(self.mapped_parameters.Sample, 'elements'):
            #print('Elemental composition read from file')
            #self.add_elements(self.mapped_parameters.Sample.elements)
            
    def set_elements(self, elements, lines=None):
        """Set elements present in the sample and defined the corresponding
        X-ray lines.
        
        The X-ray lines can be choosed manually or automatically.
        
        
        Parameters
        ----------
        elements : list of strings
            The symbol of the elements.  
        
        lines : list of strings
            One X-ray line for each element ('K', 'L' or 'M'). If none 
            the set of highest ionized lines with sufficient intensity 
            is selected. The beam energy is needed.
            
        See also
        --------
        add_elements, 
            
        Examples
        --------
        
        >>> s = signals.EDSSEMSpectrum(np.arange(1024))
        >>> s.set_elements(['Ni', 'O'],['K','K'])   
        Adding Ni_Ka Line
        Adding O_Ka Line
        
        >>> s.mapped_paramters.SEM.beam_energy = 10
        >>> s.set_elements(['Ni', 'O'])
        Adding Ni_La Line
        Adding O_Ka Line
        
        """          
        #Erase previous elements and X-ray lines
        self.elements = set()
        self.Xray_lines = set()
        if hasattr(self.mapped_parameters, 'Sample'):
            self.mapped_parameters.Sample.elements = []
            self.mapped_parameters.Sample.Xray_lines = []
                
        self.add_elements(elements, lines)
           
        
        
    def add_elements(self, elements, lines=None):
        """Add elements present in the sample and defined the corresponding
        X-ray lines.
        
        The X-ray lines can be choosed manually or automatically.
        
        
        Parameters
        ----------
        elements : list of strings
            The symbol of the elements.  
        
        lines : list of strings
            One X-ray line for each element ('K', 'L' or 'M'). If none 
            the set of highest ionized lines with sufficient intensity 
            is selected. The beam energy is needed.
            
        See also
        --------
        set_elements, 
            
        Examples
        --------
        
        >>> s = signals.EDSSEMSpectrum(np.arange(1024))
        >>> s.add_elements(['Ni', 'O'],['K','K'])   
        Adding Ni_Ka Line
        Adding O_Ka Line
        
        >>> s.mapped_paramters.SEM.beam_energy = 10
        >>> s.add_elements(['Ni', 'O'])
        Adding Ni_La Line
        Adding O_Ka Line
        
        """
        
        for element in elements:            
            if element in elements_db:               
                self.elements.add(element)
            else:
                print(
                    "%s is not a valid symbol of an element" % element)
                    
        if not hasattr(self.mapped_parameters, 'Sample'):
            self.mapped_parameters.add_node('Sample')
        self.mapped_parameters.Sample.elements = np.sort(list(self.elements))
        
               
        #Set X-ray lines
        if lines is None:
            self.add_lines_auto()
        else:
           self.add_lines(elements,lines)                
        self.mapped_parameters.Sample.Xray_lines = np.sort(list(self.Xray_lines))
        
    
        
    def add_lines(self,elements,lines):
        
        end_energy = self.axes_manager.signal_axes[0].axis[-1]            
            
        i = 0
        for line in lines:
            element = elements[i]
            line = line + 'a'
            if element in elements_db: 
                if line in ('Ka','La','Ma'):
                    if line in elements_db[element]['Xray_energy']:
                        print("Adding %s_%s line" % (element,line))
                        self.Xray_lines.add(element+'_'+line)                                                   
                        if elements_db[element]['Xray_energy'][line] >\
                        end_energy:
                          print("Warning: %s %s is higher than signal range." 
                          % (element,line))  
                    else:
                        print("%s is not a valid line of %s." % (line,element))
                else:
                    print(
                    "%s is not a valid symbol of a line." % line)
            else:
                print(
                    "%s is not a valid symbol of an element." % element)
            i += 1
        
            
    def add_lines_auto(self):
        """Choose the highest set of X-ray lines for the elements 
        present in self.elements  
        
        Possible line are in the current energy range and below the beam 
        energy. The highest line above an overvoltage of 2 
        (< beam energy / 2) is prefered.
            
        """
        if not hasattr(self.mapped_parameters.SEM,'beam_energy'):
            raise ValueError("Beam energy is needed in "
            "mapped_parameters.SEM.beam_energy")
        
        end_energy = self.axes_manager.signal_axes[0].axis[-1]
        beam_energy = self.mapped_parameters.SEM.beam_energy
        if beam_energy < end_energy:
           end_energy = beam_energy
           
        true_line = []
        
        for element in self.elements: 
            
            #Possible line           
            for line in ('Ka','La','Ma'):
                if line in elements_db[element]['Xray_energy']:
                    if elements_db[element]['Xray_energy'][line] < \
                    end_energy:
                        true_line.append(line)
            
            #Choose the better line
            i = 0
            select_this = -1            
            for line in true_line:
                if elements_db[element]['Xray_energy'][line] < \
                beam_energy/2:
                    select_this = i
                    break
                i += 1           
                     
            if true_line == []:
                print("No possible line for %s" % element)
            else:       
                self.Xray_lines.add(element+'_'+true_line[select_this])
                print("Adding %s_%s line" % (element,true_line[select_this]))
                
         
    
    def get_intensity_map(self, Xray_lines = 'auto',plot_result=True,
        width_energy_reso=1):
        """Return the intensity map of selected Xray lines.
        
        The intensity is the sum over several energy channels. The width
        of the sum is determined using the energy resolution of the detector
        defined with the FWHM of Mn Ka in
        self.mapped_parameters.SEM.EDS.energy_resolution_MnKa
        (by default 130eV). 
        
        
        Parameters
        ----------
        
        Xray_lines: list of string | 'auto'
            If 'auto' (default option), the lines defined with set_elements are used, which 
            are in 'mapped.parameters.Sample.X_ray_lines'. 
        
        width_energy_reso: Float
            factor to change the width used for the sum. 1 is equivalent
            of a width of 2 X FWHM 
            
        Examples
        --------
        
        >>> specImg.plot_intensity_map(["C_Ka", "Ta_Ma"])
        
        See also
        --------
        
        deconvolve_intensity
        
        """
        
                
        if Xray_lines == 'auto':
            if hasattr(self.mapped_parameters, 'Sample') and \
            hasattr(self.mapped_parameters.Sample, 'Xray_lines'):
                Xray_lines = self.mapped_parameters.Sample.Xray_lines
            else:
                raise ValueError("Not X-ray line, set them with add_elements")            
        
        if self.mapped_parameters.signal_type == 'EDS_SEM':
            FWHM_MnKa = self.mapped_parameters.SEM.EDS.energy_resolution_MnKa
        else:
            FWHM_MnKa = self.mapped_parameters.TEM.EDS.energy_resolution_MnKa
                        
        intensities = []
        if self.axes_manager.navigation_dimension > 1:
            signal_to_index = self.axes_manager.navigation_dimension - 2                  
            for Xray_line in Xray_lines:
                element = Xray_line[:-3]
                line = Xray_line[-2:]            
                line_energy = elements_db[element]['Xray_energy'][line]
                line_FWHM = FWHM_eds(FWHM_MnKa,line_energy)
                img = self.to_image(signal_to_index)
                img.mapped_parameters.title = 'Intensity of ' + Xray_line +\
                ' at ' + str(line_energy) + ' keV'
                det = width_energy_reso*line_FWHM
                if plot_result:
                    img[line_energy-det:line_energy+det].sum(0).plot(False)
                intensities.append(img[...,line_energy-det:line_energy+det].sum(0))
        else:
            for Xray_line in Xray_lines:
                element = Xray_line[:-3]
                line = Xray_line[-2:]            
                line_energy = elements_db[element]['Xray_energy'][line]
                line_FWHM = FWHM_eds(FWHM_MnKa,line_energy)
                det = width_energy_reso*line_FWHM
                if plot_result:
                    print("%s at %s keV : Intensity = %s" 
                    % (Xray_line, line_energy,\
                     self[line_energy-det:line_energy+det].sum(0).data) )
                intensities.append(self[line_energy-det:line_energy+det].sum(0).data)
        return intensities
                 
    def running_sum(self) :
        """
        Apply a running sum on the data.
        
        """
        dim = self.data.shape
        data_s = np.zeros_like(self.data)        
        data_s = np.insert(data_s, 0, 0,axis=-3)
        data_s = np.insert(data_s, 0, 0,axis=-2)
        end_mirrors = [[0,0],[-1,0],[0,-1],[-1,-1]]
        
        for end_mirror in end_mirrors:  
            tmp_s=np.insert(self.data, end_mirror[0], self.data[...,end_mirror[0],:,:],axis=-3)
            data_s += np.insert(tmp_s, end_mirror[1], tmp_s[...,end_mirror[1],:],axis=-2)
        data_s = data_s[...,1::,:,:][...,1::,:]
        
        if hasattr(self.mapped_parameters, 'SEM'):            
            mp = self.mapped_parameters.SEM
        else:
            mp = self.mapped_parameters.TEM
        if hasattr(mp, 'EDS') and hasattr(mp.EDS, 'live_time'):
            mp.EDS.live_time = mp.EDS.live_time * len(end_mirrors)
        self.data = data_s
        
    def plot_Xray_line(self):
        """
        Annotate a spec.plot() with the name of the selected X-ray 
        lines
        
        See also
        --------
        
        set_elements
        
        """
        if self.axes_manager.navigation_dimension > 0:
            raise ValueError("Works only for single spectrum")
        
        
        mp = self.mapped_parameters
        line_energy =[]
        intensity = []
        Xray_lines = mp.Sample.Xray_lines
        for Xray_line in Xray_lines:
            element = Xray_line[:-3]
            line = Xray_line[-2:] 
            line_energy.append(elements_db[element]['Xray_energy'][line])
            intensity.append(self[line_energy[-1]].data[0])
        
        self.plot() 
        for i in range(len(line_energy)):
            plt.annotate(Xray_lines[i],xy = (line_energy[i],intensity[i]))
    
            
    
       
       

    
