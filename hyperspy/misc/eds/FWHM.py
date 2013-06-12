import math

from hyperspy.misc.eds.elements import elements as elements_db

def FWHM_eds(energy_resolution_MnKa,E):
    """Calculates the FWHM of a peak at energy E.
    
    Parameters
    ----------
    energy_resolution_MnKa : float
        Energy resolution of Mn Ka in eV
    E : float
        Energy of the peak in keV
            
    Returns
    -------
    float : FWHM of the peak in keV
    
    Notes
    -----
    From the textbook of Goldstein et al., Plenum publisher, 
    third edition p 315
    
    """
    FWHM_ref = energy_resolution_MnKa
    E_ref = elements_db['Mn']['Xray_energy']['Ka']
    
    
    FWHM_e = 2.5*(E-E_ref)*1000 + FWHM_ref*FWHM_ref
   
    return math.sqrt(FWHM_e)/1000 # In mrad
