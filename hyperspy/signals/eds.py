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


import numpy as np


from hyperspy.signals.spectrum import Spectrum
from hyperspy.signals.image import Image
#from hyperspy.signals.eds_sem import EDSSEMSpectrum
#from hyperspy.signals.eds_tem import EDSTEMSpectrum


class EDSSpectrum(Spectrum):
    
    def __init__(self, *args, **kwards):
        Spectrum.__init__(self, *args, **kwards)
        # Attributes defaults  
        self.elements = set()
        if hasattr(self.mapped_parameters, 'Sample') and \
        hasattr(self.mapped_parameters.Sample, 'elements'):
            print('Elemental composition read from file')
            self.add_elements(self.mapped_parameters.Sample.elements)
        
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
        
                
    def test_hello(self):
        print("Hello")
       
       

    
