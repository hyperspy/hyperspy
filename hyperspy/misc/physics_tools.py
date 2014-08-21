import numpy as np


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
    v_rel = np.sqrt(1 - 1 / gamma ** 2)
    e_lambda = 2 * np.pi / (2590e9 * (gamma * v_rel))  # m

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
        temp += Z[1] * Z[1] ** exponent
        total_e += Z[0] * Z[1]
    return (temp / total_e) ** (1 / exponent)
