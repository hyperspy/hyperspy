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

import numpy as np
import math

from eelslab.component import Component

def voigt(x, FWHM=1, gamma=1, center=0, scale=1): 
    """ 
    Voigt peak. 
 
    The voigt peak is the convolution of a Lorentz peak with a Gaussian peak. 
 
    The formula used to calculate this is:: 
 
        z(x) = (x + 1j gamma) / (sqrt(2) sigma) 
        w(z) = exp(-z**2) erfc(-1j z) / (sqrt(2 pi) sigma) 
 
        V(x) = scale Re(w(z(x-center))) 
 
   :Parameters: 
     gamma : real 
      The half-width half-maximum of the Lorentzian 
     FWHM : real 
      The FWHM of the Gaussian 
     center : real 
      Location of the center of the peak 
     scale : real 
      Value at the highest point of the peak 
 
    Ref: W.I.F. David, J. Appl. Cryst. (1986). 19, 63-64 

    Note: adjusted to use stddev and HWHM rather than FWHM parameters 
    """ 
    # wofz function = w(z) = Fad[d][e][y]eva function = exp(-z**2)erfc(-iz) 
    from scipy.special import wofz
    sigma = FWHM / 2.3548200450309493
    z = (np.asarray(x)-center+1j*gamma)/(sigma*math.sqrt(2)) 
    V = wofz(z)/(math.sqrt(2*np.pi)*sigma) 
    return scale*V.real

class Voigt(Component):
    """
    """

    def __init__(self):
        Component.__init__(self, ('area', 'origin', 'FWHM', 'gamma', 'resolution', 
        'shirley_background', 'non_isochromaticity', 'transmission_function'))
        self.name = 'Voigt'
        self.FWHM.value = 1
        self.gamma.value = 0
        self.area.value = 1
        self.resolution.value = 0
        self.resolution.free = False
        self.shirley_background.free = False
        self.non_isochromaticity.value = 0
        self.non_isochromaticity.free = False
        self.transmission_function.value = 1
        self.transmission_function.free = False
        # Options
        self.shirley_background.active = False
        self.spin_orbit_splitting = False
        self.spin_orbit_branching_ratio = 0.5
        self.spin_orbit_splitting_energy = 0.61
        
        self.isbackground = False
        self.convolved = True

    def function(self, x):
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the background model, returns the background
        model for the current parameters.
        """
        area = self.area.value * self.transmission_function.value
        origin = self.origin.value
        ab = self.non_isochromaticity.value
        if self.resolution.value == 0:
            FWHM = self.FWHM.value
        else:
            FWHM = math.sqrt(self.FWHM.value**2 + self.resolution.value**2)
        gamma = self.gamma.value
        k = self.shirley_background.value
        f = voigt(x, 
        FWHM = FWHM, gamma = gamma, center = origin - ab, scale = area)
        if self.spin_orbit_splitting is True:
            ratio = self.spin_orbit_branching_ratio
            shift = self.spin_orbit_splitting_energy
            f2 = voigt(x, FWHM = FWHM, gamma = gamma, 
            center = origin - ab - shift, scale = area*ratio)
            f += f2
        if self.shirley_background.active:
            cf = np.cumsum(f)
            cf = cf[-1] - cf
            self.cf = cf
            return cf*k + f
        else:
            return f

