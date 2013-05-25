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
            
class Spectrum(Signal):
    """
    """
    _default_record_by = 'spectrum'
    
    def __init__(self, *args, **kwargs):
        Signal.__init__(self, *args, **kwargs)
        self.axes_manager.set_signal_dimension(1)

    def to_image(self, signal_to_index=0):
        """Spectrum to image

        Parameters
        ----------
        signal_to_index : integer
            Position to move the signal axis.        
            
        Examples
        --------        
        >>> s = signals.Spectrum(np.ones((3,4,5,6)))
        >>> s
        <Spectrum, title: , dimensions: (3L, 4L, 5L, 6L)>

        >>> s.to_image()
        <Image, title: , dimensions: (6L, 3L, 4L, 5L)>

        >>> s.to_image(1)
        <Image, title: , dimensions: (3L, 6L, 4L, 5L)>
        
        """
        
        from hyperspy.signals.image import Image
        dic = self._get_signal_dict()
        dic['mapped_parameters']['record_by'] = 'image'
        dic['data'] = np.rollaxis(dic['data'], -1, signal_to_index)
        dic['axes'] = utils_varia.rollelem(dic['axes'],
                                           -1,
                                           signal_to_index)
        for axis in dic['axes']:
            del axis['index_in_array']
        im = Image(**dic)
        
        if hasattr(self, 'learning_results'):
            if (signal_to_index != 0 and 
                self.learning_results.loadings is not None):
                print("The learning results won't be transfered correctly")
            else:
                im.learning_results = copy.deepcopy(
                    self.learning_results)
                im.learning_results._transpose_results()
                im.learning_results.original_shape = self.data.shape

        im.tmp_parameters = self.tmp_parameters.deepcopy()
        return im

    def to_EELS(self):
        from hyperspy.signals.eels import EELSSpectrum
        dic = self._get_signal_dict()
        dic['mapped_parameters']['signal_type'] = 'EELS'
        eels = EELSSpectrum(**dic)
        if hasattr(self, 'learning_results'):
            eels.learning_results = copy.deepcopy(self.learning_results)
        eels.tmp_parameters = self.tmp_parameters.deepcopy()
        return eels
        
    def to_simulation(self):
        from hyperspy.signals.spectrum_simulation import (
                SpectrumSimulation)
        dic = self._get_signal_dict()
        if self.mapped_parameters.has_item("signal_type"):
            dic['mapped_parameters']['signal_type'] = (
                self.mapped_parameters.signal_type + '_simulation')
        simu = SpectrumSimulation(**dic)
        if hasattr(self, 'learning_results'):
            simu.learning_results = copy.deepcopy(self.learning_results)
        simu.tmp_parameters = self.tmp_parameters.deepcopy()
        return simu
