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

import math

import numpy as np

from hyperspy.component import Component


class VolumePlasmonDrude(Component):
    """Drude volume plasmon energy loss function component
    
    .. math::

       Energy loss function defined as:
       
       f(E) = \\frac{E(\Delta E_p)E_p^2}{(E^2-E_p^2)^2+(E\Delta E_p)^2}       
        
    +------------+-----------------+
    | Parameter  |    Attribute    |
    +------------+-----------------+
    |    E_p     |  plasmon_energy |
    +------------+-----------------+
    | delta_E_p  |plasmon_linewidth|
    +------------+-----------------+
    | intensity  |   intensity     |
    +------------+-----------------+
    
    Notes
    -----
    Refer to Egerton, R. F., Electron Energy-Loss Spectroscopy in the 
    Electron Microscope, 2nd edition, Plenum Press 1996, pp. 154-158
    for details, including original equations.
    
    
    """

    def __init__(self):
        Component.__init__(self, ['intensity', 'plasmon_energy',
        'plasmon_linewidth'])
        self.intensity.value = 1
        self.plasmon_energy.value = 7.1
        self.plasmon_linewidth.value = 2.3
        self.plasmon_energy.grad = self.grad_plasmon_energy
        self.plasmon_linewidth.grad = self.grad_plasmon_linewidth
        self.intensity.grad = self.grad_intensity
    
    def function(self, x):
        plasmon_energy = self.plasmon_energy.value
        plasmon_linewidth = self.plasmon_linewidth.value
        intensity = self.intensity.value
        return np.where(x > 0, 
            intensity*(plasmon_energy**2 * x * plasmon_linewidth) / (
            (x**2 - plasmon_energy**2)**2 + (x * plasmon_linewidth)**2),
            0)
            
    # Partial derivative with respect to the plasmon energy E_p
    def grad_plasmon_energy(self, x):
        plasmon_energy = self.plasmon_energy.value
        plasmon_linewidth = self.plasmon_linewidth.value
        intensity = self.intensity.value

        return np.where(x > 0,
            2 * x * plasmon_linewidth * plasmon_energy * (
            (x**4 + (x * plasmon_linewidth)**2 - plasmon_energy**4)
            / (x**4 + x**2 * (plasmon_linewidth**2 - 2 * 
            plasmon_energy**2) + plasmon_energy**4)**2)*intensity, 0)
        
    # Partial derivative with respect to the plasmon linewidth delta_E_p
    def grad_plasmon_linewidth(self, x):
        plasmon_energy = self.plasmon_energy.value
        plasmon_linewidth = self.plasmon_linewidth.value
        intensity = self.intensity.value

        return np.where(x > 0,
            x * plasmon_energy * ((x**4 - x**2 * (2 * plasmon_energy**2
            + plasmon_linewidth**2) + plasmon_energy**4) / (x**4 + x**2
            * (plasmon_linewidth**2 - 2 * plasmon_energy**2)
            + plasmon_energy**4)**2)*intensity,0)
            
    def grad_intensity(self, x):
        return self.function(x) / self.intensity.value
            
