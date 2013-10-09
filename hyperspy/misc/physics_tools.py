import numpy as np
import math

from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.eds.elements import elements as elements_db
import hyperspy.misc.units_converter as units_converter

def bragg_scattering_angle(d, E0=100):
    """Calculate the first order bragg diffraction semiangle.
    
    Parameters
    ----------
    d : float
        interplanar distance in m.
    E0 : float
        Incident energy in keV

    Returns
    -------
    float : Semiangle of scattering of the first order difracted beam. This is
    two times the bragg angle.
    
    """

    gamma = 1 + E0 / 511.
    v_rel = np.sqrt(1-1/gamma**2)
    e_lambda = 2*np.pi/(2590e9*(gamma*v_rel)) # m

    return e_lambda / d

def effective_Z(Z_list, exponent=2.94):
    """Effective atomic number of a compound or mixture.

    Exponent = 2.94 for X-ray absorption.

    Parameters
    ----------
    Z_list : list of tuples
        A list of tuples (f,Z) where f is the number of atoms of the element in
        the molecule and Z its atomic number

    Returns
    -------
    float
    
    """
    exponent = float(exponent)
    temp = 0
    total_e = 0
    for Z in Z_list:
        temp += Z[1]*Z[1]**exponent
        total_e += Z[0]*Z[1]
    return (temp/total_e)**(1/exponent)
    
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
        
def density_from_composition(elements,compositions):
    """Return the density from the sample composition
    
    Parameters
    ----------
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']
        
    Composition: list of float
        The atomic composition of the sample.
        
    Returns
    -------
    
    The density in g/cm3
    
    """
    density = 0
    weights = units_converter.atomic_to_weight(elements,compositions)
    for i, element in enumerate(elements):
        density = density + elements_db[element]['density']*weights[i]

    return density
