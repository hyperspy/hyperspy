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

import warnings

from hyperspy.exceptions import DataDimensionError
from hyperspy.signal import Signal
            
class Spectrum(Signal):
    """
    """
    _record_by = 'spectrum'
    def __init__(self, *args, **kwargs):
        Signal.__init__(self, *args, **kwargs)
        self.axes_manager.set_signal_dimension(1)

    def to_EELS(self):
        warnings.warn(
            'This method is deprecated and and will be removed '
            'in the next version. '
            'Please use `set_signal_type("EELS")` instead',
              DeprecationWarning)
        s = self.deepcopy()
        s.set_signal_type("EELS")
        return s

    
    def to_EDS(self, microscope=None):
        warnings.warn(
            'This method is deprecated and and will be removed '
            'in the next version. '
            'Please use `set_signal_type("EDS_TEM")` or '
            '`set_signal_type("EDS_SEM")` instead',
              DeprecationWarning)
        if microscope == None:             
            if self.mapped_parameters.signal_type == 'EDS_SEM':
                microscope = 'SEM'
            elif self.mapped_parameters.signal_type == 'EDS_TEM':
                microscope = 'TEM'
            else:
                microscope = 'TEM'
        s = self.deepcopy()
        s.set_signal_type("EDS_"+microscope)
        return s
        
        
        #"""Return a EDSSpectrum from a Spectrum
        
        #The microscope, which defines the quantification methods, needs 
        #to be set.
        
        #Parameters
        #----------------
        #microscope : {None | 'TEM' | 'SEM'}
            #If None the microscope defined in signal_type is used 
            #(EDS_TEM or EDS_SEM). If 'TEM' or 'SEM', the signal_type is 
            #overwritten.
            
        #"""
        #from hyperspy.signals.eds_tem import EDSTEMSpectrum
        #from hyperspy.signals.eds_sem import EDSSEMSpectrum
                
        #if microscope == None:             
            #if self.mapped_parameters.signal_type == 'EDS_SEM':
                #microscope = 'SEM'
            #elif self.mapped_parameters.signal_type == 'EDS_TEM':
                #microscope = 'TEM'
            #else:
                #raise ValueError("Set a microscope. Valid microscopes " 
                #"are: 'SEM' or 'TEM'")
            
        #dic = self._get_signal_dict()
        #if microscope == 'SEM':
            #dic['mapped_parameters']['signal_type'] = 'EDS_SEM'
            #eds = EDSSEMSpectrum(**dic)
        #elif microscope == 'TEM':
            #dic['mapped_parameters']['signal_type'] = 'EDS_TEM'
            #eds = EDSTEMSpectrum(**dic)
        #else:
            #raise ValueError("Unkown microscope. Valid microscopes " 
                #"are: 'SEM' or 'TEM'")
        
        #if hasattr(self, 'learning_results'):
            #eds.learning_results = copy.deepcopy(self.learning_results)
        #eds.tmp_parameters = self.tmp_parameters.deepcopy()
        #return eds
    
    def to_image(self):
        """Returns the spectrum as an image.
        
        See Also:
        ---------
        as_image : a method for the same purpose with more options.  
        signals.Spectrum.to_image : performs the inverse operation on images.
        
        Raises
        ------
        DataDimensionError: when data.ndim < 2
    
        """
        if self.data.ndim < 2:
            raise DataDimensionError(
                "A Signal dimension must be >= 2 to be converted to an Image")
        im = self.rollaxis(-1j, 0j)
        im.mapped_parameters.record_by = "image"
        im._assign_subclass()
        return im
