# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
#
# This file is part of EELSLab.
#
# EELSLab is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# EELSLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EELSLab; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA

from ..component import Component
from scipy.interpolate import interp1d
from .. import messages

class FixedPattern(Component):
    """
    Given an array of the same shape as Spectrum energy_axis, returns it as
    a component that can be added to a model.
    """

    def __init__( self, array=None ):
        Component.__init__(self, ['intensity', 'origin'])
        self.name = 'Fixed pattern'
        self.array = array
        self.intensity.free = True
        self.intensity.value = 1.
        self.origin.value = 0
        self.origin.free = False
        self.isbackground = True
        self.convolved = False
        self.intensity.grad = self.grad_intensity
        self.interpolate = False
        self._interpolation_ready = False
    
    def prepare_interpolator(self, x, kind = 3, fill_value = 0, **kwards):
        self.interp = interp1d(x, self.array, kind = kind, 
        fill_value = fill_value, bounds_error=False, **kwards)
        self._interpolation_ready = True
    def function(self, x):
        if self.interpolate is False:
            return self.array * self.intensity.value
        elif self._interpolation_ready is True:
            return self.interp(x - self.origin.value) * self.intensity.value
        else:
            messages.warning(
            'To use interpolation you must call prepare_interpolator first')
            
    def grad_intensity(self, x):
        return self.array
    
