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


from eelslab.component import Component
from scipy.interpolate import interp1d

class ZLPFingerprinting(Component):
    """

    """

    def __init__(self, zl):
        Component.__init__(self, ['intensity', 'xscale', 'origin', 'offset'])
        self.name = 'ZL_fingerprinting'
        self.zl = zl
        self.intensity.free = True
        self.intensity.value = 1.
        self.xscale.value = 1.
        self.offset.value = 0.
        self.origin.value = 0.
        # Options
        self.isbackground = True
        self.convolved = False
#        self.intensity.grad = self.grad_intensity
        self.f = interp1d(zl.energy_axis, zl.data_cube.squeeze(), 
        bounds_error = False, fill_value = 0.)
        
    def function(self, x):
        return self.offset.value + self.intensity.value * self.f(x*self.xscale.value - 
    self.origin.value)
    
