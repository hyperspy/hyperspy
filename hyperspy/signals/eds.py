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
from hyperspy.misc.eds.elements import elements as elements_db
from hyperspy.misc.eds.FWHM import FWHM_eds
from hyperspy.misc.eds import utils as utils_eds

class EDSSpectrum(Spectrum):
    _signal_type = "EDS"
    
    def __init__(self, *args, **kwards):
        Spectrum.__init__(self, *args, **kwards)
        if self.mapped_parameters.signal_type == 'EDS':
            print('The microscope type is not set. Use '
            'set_signal_type(\'EDS_TEM\') or set_signal_type(\'EDS_SEM\')')
        # Attributes defaults
        if hasattr(self,'elements')==False:
            self.elements = set()
        if hasattr(self,'Xray_lines')==False:
            self.Xray_lines = set()
            
    def sum(self,axis):
        """Sum the data over the given axis.

        Parameters
        ----------
        axis : {int, string}
           The axis can be specified using the index of the axis in 
           `axes_manager` or the axis name.

        Returns
        -------
        s : Signal

        See also
        --------
        sum_in_mask, mean

        Usage
        -----
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.sum(-1).data.shape
        (64,64)
        # If we just want to plot the result of the operation
        s.sum(-1, True).plot()
        
        """
        #modify time spend per spectrum
        if hasattr(self.mapped_parameters, 'SEM'):
            mp = self.mapped_parameters.SEM
        else:
            mp = self.mapped_parameters.TEM
        if hasattr(mp, 'EDS') and hasattr(mp.EDS, 'live_time'):
            mp.EDS.live_time = mp.EDS.live_time * self.axes_manager.shape[axis]
        return super(EDSSpectrum, self).sum( axis)
        
    def rebin(self, new_shape):
        """Rebins the data to the new shape

        Parameters
        ----------
        new_shape: tuple of ints
            The new shape must be a divisor of the original shape
            
        """
        new_shape_in_array = []
        for axis in self.axes_manager._axes:
            new_shape_in_array.append(
                new_shape[axis.index_in_axes_manager])
        factors = (np.array(self.data.shape) / 
                           np.array(new_shape_in_array))
        #modify time per spectrum
        if hasattr(self.mapped_parameters, 'SEM'):
            mp = self.mapped_parameters.SEM
        else:
            mp = self.mapped_parameters.TEM
        if hasattr(mp, 'EDS') and hasattr(mp.EDS, 'live_time'):
            for factor in factors:
                mp.EDS.live_time = mp.EDS.live_time * factor
        Spectrum.rebin(self, new_shape)
    
    def set_elements(self, elements, lines=None):
        """Erase all elements and set them with the corresponding
        X-ray lines.
        
        The X-ray lines can be choosed manually or automatically.
        
        
        Parameters
        ----------
        elements : list of strings
            The symbol of the elements.  
        
        lines : list of strings
            One X-ray line for each element ('Ka', 'La', 'Ma',...). If None 
            the set of highest ionized lines with sufficient intensity 
            is selected. The beam energy is needed.
            
        See also
        --------
        add_elements, 
            
        Examples
        --------
        
        >>> s = signals.EDSSEMSpectrum(np.arange(1024))
        >>> s.set_elements(['Ni', 'O'],['Ka','Ka'])   
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
        """Add elements and the corresponding X-ray lines.
        
        The X-ray lines can be choosed manually or automatically.        
        
        Parameters
        ----------
        elements : list of strings
            The symbol of the elements.  
        
        lines : list of strings
            One X-ray line for each element ('Ka', 'La', 'Ma',...). If none 
            the set of highest ionized lines with sufficient intensity 
            is selected. The beam energy is needed. All available lines
            are return for a wrong lines.
            
        See also
        --------
        set_elements, 
        
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
            self._add_lines_auto(elements)
        else:
           self._add_lines(elements,lines)                
        self.mapped_parameters.Sample.Xray_lines = np.sort(list(self.Xray_lines))
        
    
        
    def _add_lines(self,elements,lines):
        
        end_energy = self.axes_manager.signal_axes[0].axis[-1]            
            
        i = 0
        for line in lines:
            element = elements[i]
            if element in elements_db: 
                if line in elements_db[element]['Xray_energy']:
                    print("Adding %s_%s line" % (element,line))
                    self.Xray_lines.add(element+'_'+line)                                                   
                    if elements_db[element]['Xray_energy'][line] >\
                    end_energy:
                      print("Warning: %s %s is higher than signal range." 
                      % (element,line))  
                else:
                    print("%s is not a valid line of %s." % (line,element))
                    print("Valid lines for %s are (importance):" % element)
                    for li in elements_db[element]['Xray_energy']:
                        print("%s (%s)" % (li,
                         elements_db['lines']['ratio_line'][li]))
            else:
                print(
                    "%s is not a valid symbol of an element." % element)
            i += 1
        
            
    def _add_lines_auto(self,elements):
        """Choose the highest set of X-ray lines for the elements 
        present in self.elements  
        
        Possible line are in the current energy range and below the beam 
        energy. The highest line above an overvoltage of 2 
        (< beam energy / 2) is prefered.
            
        """
        if hasattr(self.mapped_parameters, 'SEM') and \
            hasattr(self.mapped_parameters.SEM,'beam_energy') :
            beam_energy = self.mapped_parameters.SEM.beam_energy
        elif hasattr(self.mapped_parameters, 'TEM') and \
            hasattr(self.mapped_parameters.TEM,'beam_energy') :
            beam_energy = self.mapped_parameters.TEM.beam_energy
        else:
            raise ValueError("Beam energy is needed in "
            "mapped_parameters.TEM.beam_energy  or "
            "mapped_parameters.SEM.beam_energy")
        
        end_energy = self.axes_manager.signal_axes[0].axis[-1]
        if beam_energy < end_energy:
           end_energy = beam_energy
           
        true_line = []
        
        for element in elements: 
                        
            #Possible line (existing and excited by electron)         
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
        #test 1D Spectrum (0D problem)
        if self.axes_manager.navigation_dimension > 1:
            #signal_to_index = self.axes_manager.navigation_dimension - 2                  
            for Xray_line in Xray_lines:
                element, line = utils_eds._get_element_and_line(Xray_line)           
                line_energy = elements_db[element]['Xray_energy'][line]
                line_FWHM = FWHM_eds(FWHM_MnKa,line_energy)
                det = width_energy_reso*line_FWHM
                img = self[...,line_energy-det:line_energy+det].sum(-1)\
                        .as_image([0,1])
                img.mapped_parameters.title = 'Intensity of ' + Xray_line +\
                ' at ' + str(line_energy) + ' keV'                
                if plot_result:
                    img.plot(None)                    
                intensities.append(img)
        else:
            for Xray_line in Xray_lines:
                element, line = utils_eds._get_element_and_line(Xray_line)           
                line_energy = elements_db[element]['Xray_energy'][line]
                line_FWHM = FWHM_eds(FWHM_MnKa,line_energy)
                det = width_energy_reso*line_FWHM
                if plot_result:
                    print("%s at %s keV : Intensity = %s" 
                    % (Xray_line, line_energy,\
                     self[line_energy-det:line_energy+det].sum(0).data) )
                intensities.append(self[line_energy-det:line_energy+det].sum(0).data)
        return intensities
