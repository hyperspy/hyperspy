import numpy as np

def _F(beam_energy):
    return (1 + beam_energy / 1022) / (1 + beam_energy / 511) ** 2

def _theta_E(density, beam_energy):
    gamma = 1 + beam_energy/511 # keV
    return 5.5 * density ** 0.3 / (gamma * _F(beam_energy) * beam_energy)

def iMFP(density, beam_energy):
    """Estimate electron inelastic mean free path from density

    Parameters:
    -----------
    density : float
        Material density in g/cm**3
    beam_energy : float
        Electron beam energy in keV
    
    Notes:
    ------
    From Iakoubovskii, K., K. Mitsuishi, Y. Nakayama, and K. Furuya.
    ‘Thickness Measurements with Electron Energy Loss Spectroscopy’.
    Microscopy Research and Technique 71, no. 8 (2008): 626–31.
    https://doi.org/10.1002/jemt.20597.
    """
    theta_C = 20 # mrad
    inv_lambda = 11 * density ** 0.3 / (200 * _F(beam_energy) * beam_energy) * np.log(theta_C ** 2 / _theta_E(density, beam_energy) ** 2)
    return 1 / inv_lambda

def iMFP_angular_correction(density, beam_energy, alpha, beta):
    """Estimate the effect of limited collection angle on EELS mean free path

    Parameters:
    -----------
    density : float
        Material density in g/cm**3
    beam_energy : float
        Electron beam energy in keV
    alpha, beta : float
        Convergence and collection angles in mrad.
    
    Notes:
    ------
    From Iakoubovskii, K., K. Mitsuishi, Y. Nakayama, and K. Furuya.
    ‘Thickness Measurements with Electron Energy Loss Spectroscopy’.
    Microscopy Research and Technique 71, no. 8 (2008): 626–31.
    https://doi.org/10.1002/jemt.20597.
    """
    theta_C = 20 # mrad
    A = alpha ** 2 + beta ** 2 + 2 * _theta_E(density, beam_energy) ** 2 + np.abs(alpha ** 2 - beta ** 2) 
    B = alpha ** 2 + beta ** 2 + 2 * theta_C ** 2 + np.abs(alpha ** 2 - beta ** 2)
    return np.log(theta_C ** 2 / _theta_E(density, beam_energy) ** 2) / np.log(A * theta_C ** 2 / B / _theta_E(density, beam_energy) ** 2)

