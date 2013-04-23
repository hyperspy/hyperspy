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



from hyperspy.signals.spectrum import Spectrum
from hyperspy.signals.image import Image
from hyperspy.misc.eds.elements import elements as elements_db
from hyperspy.misc.eds.FWHM import FWHM_eds


class EDSSpectrum(Spectrum):
    
    def __init__(self, *args, **kwards):
        Spectrum.__init__(self, *args, **kwards)
        # Attributes defaults  
        self.elements = set()
        self.Xray_lines = set()
        if hasattr(self.mapped_parameters, 'Sample') and \
        hasattr(self.mapped_parameters.Sample, 'elements'):
            print('Elemental composition read from file')
            self.add_elements(self.mapped_parameters.Sample.elements)
            
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
        
        >>> s = signals.EDSSEMSpectrum({'data' : np.arange(1024)})
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
        
        >>> s = signals.EDSSEMSpectrum({'data' : np.arange(1024)})
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
                
    
    def plot_intensity_map(self, Xray_lines = None, width_energy_reso=1):
        """Plot the intensity map of selected Xray lines.
        
        The intensity is the sum over several energy channels. The width
        of the sum is determined using the energy resolution of the detector
        defined with the FWHM of Mn Ka in
        self.mapped_parameters.SEM.EDS.energy_resolution_MnKa
        (by default 130eV). 
        
        
        Parameters
        ----------
        
        Xray_lines: list of string
            If None, the lines defined with set_elements are used, which 
            are in 'mapped.parameters.Sample.X_ray_lines'. 
        
        width_energy_reso: Float
            factor to change the width used for the sum. 1 is equivalent
            of a width of 2 X FWHM 
            
        Examples
        --------
        
        >>> specImg.plot_intensity_map(["C_Ka", "Ta_Ma"])
        
        """
        
        if not hasattr(self.mapped_parameters, 'Sample') and \
        hasattr(self.mapped_parameters.Sample, 'Xray_lines'):
            raise ValueError("Not X-ray line, set them with add_elements")        
        if Xray_lines == None:
            Xray_lines = self.mapped_parameters.Sample.Xray_lines
        
        if self.mapped_parameters.signal_type == 'EDS_SEM':
            FWHM_MnKa = self.mapped_parameters.SEM.EDS.energy_resolution_MnKa
        else:
            FWHM_MnKa = self.mapped_parameters.TEM.EDS.energy_resolution_MnKa
            
        
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
                img[line_energy-det:line_energy+det].sum(signal_to_index).plot(None,False)
        else:
            for Xray_line in Xray_lines:
                element = Xray_line[:-3]
                line = Xray_line[-2:]            
                line_energy = elements_db[element]['Xray_energy'][line]
                line_FWHM = FWHM_eds(FWHM_MnKa,line_energy)
                det = width_energy_reso*line_FWHM
                print("%s at %s keV : Intensity = %s" 
                % (Xray_line, line_energy,\
                 self[line_energy-det:line_energy+det].sum(0).data) )
                 
    def running_sum(self) :
        dim = self.data.shape
        if len(dim)==3:
            s = np.zeros(np.array(dim)+[1,1,0])
        elif len(dim)==4:
            s = np.zeros(np.array(dim)+[0,1,1,0])
        else:
            raise ValueError("Data dimension not supported")
        dat = self.deepcopy().data
        def add_beg(nb,si) :
            ap = range(si)
            ap.reverse()
            return np.append(ap,range(nb))
        def add_end(nb,si) :
            ap = range(-1,-si-1,-1)
            return np.append(range(nb),ap)  
        no = 1
        if len(dim)==3:
            s = s + dat[add_beg(dim[0],no)][...,add_end(dim[1],no),...]
            s = s + dat[add_end(dim[0],no)][...,add_end(dim[1],no),...]
            s = s + dat[add_beg(dim[0],no)][...,add_end(dim[1],no),...]
            s = s + dat[add_end(dim[0],no)][...,add_end(dim[1],no),...]
            s = s[1::][...,1::,...]
        elif len(dim)==4:
            s = s + dat[...,add_beg(dim[1],no),...,...][...,...,add_end(dim[2],no),...]
            s = s + dat[...,add_end(dim[1],no),...,...][...,...,add_end(dim[2],no),...]
            s = s + dat[...,add_beg(dim[1],no),...,...][...,...,add_end(dim[2],no),...]
            s = s + dat[...,add_end(dim[1],no),...,...][...,...,add_end(dim[2],no),...]
            s = s[...,1::,...,...][...,...,1::,...]
        
        if hasattr(self.mapped_parameters, 'SEM'):            
            mp = self.mapped_parameters.SEM
        else:
            mp = self.mapped_parameters.TEM
        if hasattr(mp, 'EDS') and hasattr(mp.EDS, 'live_time'):
            mp.EDS.live_time = mp.EDS.live_time * 4
            
        return self.get_deepcopy_with_new_data(s.astype(int))
            
    
       
       

    
