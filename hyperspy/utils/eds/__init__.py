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
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>

import numpy as np
import math

from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.eds.elements import elements as elements_db
import hyperspy.misc.units_converter as units_converter

def xray_range(Xray_line,beam_energy,rho='auto'):
    '''Return the Anderson-Hasler X-ray range. 
    
    Return the maximum range of X-ray generation in a pure bulk material.
    
    Parameters
    ----------
    Xray_line: str
        The X-ray line, e.g. 'Al_Ka'
        
    beam_energy: float
        The energy of the beam in kV.
        
    rho: float | 'auto'
        The density of the material in g/cm3. If 'auto', the density 
        of the pure element is used.
    
    Returns
    -------
    X-ray range in micrometer.
    
    Notes
    -----
    From Anderson, C.A. and M.F. Hasler (1966). In proceedings of the
    4th international conference on X-ray optics and microanalysis.
    
    See also the textbook of Goldstein et al., Plenum publisher,
    third edition p 286
    
    '''
    element, line = utils_eds._get_element_and_line(Xray_line)
    if rho == 'auto':
        rho = elements_db[element]['density']
    Xray_energy = elements_db[element]['Xray_energy'][line]
    
    return 0.064/rho*(np.power(beam_energy,1.68)-
        np.power(Xray_energy,1.68))
        
def electron_range(element,beam_energy,rho='auto',tilt=0):
    '''Return the Kanaya-Okayama electron range 
    
    Return the maximum electron range in a pure bulk material.
    
    Parameters
    ----------
    element: str
        The abbreviation of the element, e.g. 'Al'
        
    beam_energy: float
        The energy of the beam in kV.
        
    rho: float | 'auto'
        The density of the material in g/cm3. If 'auto', the density of 
        the pure element is used.
        
    tilt: float (degree)
        the tilt of the sample.
        
    Returns
    -------
    Electron range in micrometer.
    
    Notes
    -----
    From Kanaya, K. and S. Okayama (1972). J. Phys. D. Appl. Phys. 5, p43
    
    See also the textbook of Goldstein et al., Plenum publisher,
    third edition p 72
    '''

    if rho == 'auto':
        rho = elements_db[element]['density']
    Z = elements_db[element]['Z']
    A = elements_db[element]['A']
    
    return (0.0276*A/np.power(Z,0.89)/rho*
        np.power(beam_energy,1.67)*math.cos(math.radians(tilt)))


def density_from_composition(elements,compositions,compo_unit='at'):
    """Return the density from the sample composition
    
    Parameters
    ----------
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']
        
    Compositions: list of float
        The atomic composition of the sample e.g. [0.2,0.8]. The composition
        is normalized.
        
    Returns
    -------
    
    The density in g/cm3
    
    """
    
    if compo_unit=='at':
        weights = units_converter.atomic_to_weight(elements,compositions)
    elif compo_unit=='wt':
        weights = np.array(compositions)/float(sum(compositions))
    density = 0
    for i, element in enumerate(elements):
        density = density + elements_db[element]['density']*weights[i]

    return density

def take_off_angle(tilt_stage,
    azimuth_angle,
    elevation_angle):
    """Calculate the take-off-angle (TOA).
    
    TOA is the angle with which the X-rays leave the surface towards 
    the detector. 

    Parameters
    ----------
    tilt_stage: float (Degree)
        The tilt of the stage. The sample is facing the detector when
        positively tilted. 

    azimuth_angle: float (Degree)
        The azimuth of the detector. 0 is perpendicular to the tilt 
        axis. 

    elevation_angle: float (Degree)
        The elevation of the detector.
                
    Returns
    -------
    take_off_angle: float (Degree)
    
    Notes
    -----
    Defined by M. Schaffer et al., Ultramicroscopy 107(8), pp 587-597 (2007)
    
    """        

        
    a = math.radians(90+tilt_stage)
    b = math.radians(azimuth_angle)
    c = math.radians(elevation_angle)
    
    return math.degrees( np.arcsin (-math.cos(a)*math.cos(b)*math.cos(c) \
    + math.sin(a)*math.sin(c)))
