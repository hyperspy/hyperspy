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
from hyperspy import messages




class Image(Signal):
    """
    """    
    def __init__(self, *args, **kw):
        super(Image,self).__init__(*args, **kw)
        self.axes_manager.set_view('image')
                
    def to_spectrum(self):
        from hyperspy.signals.spectrum import Spectrum
        dic = self._get_signal_dict()
        dim = len(self.data.shape)
        dic['mapped_parameters']['record_by'] = 'spectrum'
        dic['data'] = np.rollaxis(dic['data'], 0, dim)
        dic['axes'] = utils_varia.rollelem(dic['axes'],0, dim)
        i = 0
        for axis in dic['axes']:
            axis['index_in_array'] = i
            i += 1
        sp = Spectrum(dic)
        sp.axes_manager._set_axes_index_in_array_from_position()
        if hasattr(self, 'learning_results'):
            sp.learning_results = copy.deepcopy(self.learning_results)
            sp.learning_results._transpose_results()
            sp.learning_results.original_shape = self.data.shape
        return sp
        
