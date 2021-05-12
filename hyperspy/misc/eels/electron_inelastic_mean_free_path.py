# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import math

def _F(electron_energy):
    return (1 + electron_energy / 1022) / (1 + electron_energy / 511) ** 2

def _theta_E(density, electron_energy):
    return 5.5 * density ** 0.3 / (_F(electron_energy) * electron_energy)

def iMFP_Iakoubovskii(density, electron_energy):
    """Estimate electron inelastic mean free path from density

    Parameters
    ----------
    density : float
        Material density in g/cm**3
    beam_energy : float
        Electron beam energy in keV

    Notes
    -----
    For details see Equation 9 in reference [*]_.

    .. [*] Iakoubovskii, K., K. Mitsuishi, Y. Nakayama, and K. Furuya.
       ‘Thickness Measurements with Electron Energy Loss Spectroscopy’.
       Microscopy Research and Technique 71, no. 8 (2008): 626–31.
       https://doi.org/10.1002/jemt.20597

    Returns
    -------
    float
        Inelastic mean free path in nanometers
    """
    theta_C = 20 # mrad
    inv_lambda = 11 * density ** 0.3 / (200 * _F(electron_energy) * electron_energy) * np.log(theta_C ** 2 / _theta_E(density, electron_energy) ** 2)
    return 1 / inv_lambda


def iMFP_TPP2M(electron_energy, density, M, N_v, E_g):
    """Electron inelastic mean free path using TPP-2M

    Parameters
    ----------
    electron_energy : float
        Electron beam energy in keV
    density : float
        Material density in g/cm**3
    M : float
        Molar mass in g / mol
    N_v : int
        Number of valence electron
    E_g : float
        Band gap in eV

    Returns
    -------
    float
        Inelastic mean free path in nanometers

    Notes
    -----
    For details see reference [*]_.

    .. [*] Shinotsuka, H., S. Tanuma, C. J. Powell, and D. R. Penn. ‘Calculations
       of Electron Inelastic Mean Free Paths. X. Data for 41 Elemental Solids over
       the 50 EV to 200 KeV Range with the Relativistic Full Penn Algorithm:
       Calculations of Electron Inelastic Mean Free Paths. X’. Surface and
       Interface Analysis 47, no. 9 (September 2015): 871–88.
       https://doi.org/10.1002/sia.5789
    """
    E = electron_energy * 1e3
    rho = density
    alpha = (1 + E / 1021999.8) / (1 + E / 510998.9)**2
    E_p = 28.816 * math.sqrt(N_v * rho / M)
    gamma = 0.191 / math.sqrt(rho)
    U = (E_p / 28.816) ** 2
    C = 19.7 - 9.1 * U
    D = 534 - 208 * U
    beta = -1 + 9.44 / math.sqrt(E_p **2 + E_g**2) + 0.69 * rho ** 0.1
    iMFP = alpha * E / (E_p ** 2 * (beta * math.log(gamma * alpha * E) - C / E + D / E**2))
    return iMFP

def iMFP_angular_correction(density, beam_energy, alpha, beta):
    """Estimate the effect of limited collection angle on EELS mean free path

    Parameters
    ----------
    density : float
        Material density in g/cm**3
    beam_energy : float
        Electron beam energy in keV
    alpha, beta : float
        Convergence and collection angles in mrad.

    Notes
    -----
    For details see Equation 9 in reference [*]_.

    .. [*] Iakoubovskii, K., K. Mitsuishi, Y. Nakayama, and K. Furuya.
       ‘Thickness Measurements with Electron Energy Loss Spectroscopy’.
       Microscopy Research and Technique 71, no. 8 (2008): 626–31.
       https://doi.org/10.1002/jemt.20597
    """
    theta_C = 20 # mrad
    A = alpha ** 2 + beta ** 2 + 2 * _theta_E(density, beam_energy) ** 2 + np.abs(alpha ** 2 - beta ** 2)
    B = alpha ** 2 + beta ** 2 + 2 * theta_C ** 2 + np.abs(alpha ** 2 - beta ** 2)
    return np.log(theta_C ** 2 / _theta_E(density, beam_energy) ** 2) / np.log(A * theta_C ** 2 / B / _theta_E(density, beam_energy) ** 2)

