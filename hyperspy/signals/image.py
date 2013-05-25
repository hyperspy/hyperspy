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

import copy

import numpy as np

from hyperspy.signal import Signal
from hyperspy.misc import utils_varia





class Image(Signal):
    """
    """    
    def __init__(self, *args, **kw):
        super(Image,self).__init__(*args, **kw)
        self.axes_manager.set_signal_dimension(2)
                
    def to_spectrum(self, signal_axis=0):
        """Image to spectrum

        Parameters
        ----------
        signal_axis : integer
            Selected the signal axis.        
            
        Examples
        --------        
        >>> img = signals.Image(np.ones((3,4,5,6)))
        >>> img
        <Image, title: , dimensions: (3L, 4L, 5L, 6L)>

        >>> img.to_spectrum()
        <Spectrum, title: , dimensions: (4L, 5L, 6L, 3L)>

        >>> img.to_spectrum(1)
        <Spectrum, title: , dimensions: (3L, 5L, 6L, 4L)>
        
        """
        from hyperspy.signals.spectrum import Spectrum
        dic = self._get_signal_dict()
        dim = len(self.data.shape)
        dic['mapped_parameters']['record_by'] = 'spectrum'        
        dic['data'] = np.rollaxis(dic['data'], signal_axis, dim)
        dic['axes'] = utils_varia.rollelem(dic['axes'], signal_axis, dim)
        for axis in dic['axes']:
            del axis['index_in_array']
        sp = Spectrum(**dic)
        if hasattr(self, 'learning_results'):
            if signal_axis != 0 and self.learning_results.loadings is not None:
                print("The learning results won't be transfered correctly")
            else :
                sp.learning_results = copy.deepcopy(self.learning_results)
                sp.learning_results._transpose_results()
                sp.learning_results.original_shape = self.data.shape
                
        sp.tmp_parameters = self.tmp_parameters.deepcopy()
        return sp
        
