import numpy as np

from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.eds.ffast_mac import ffast_mac_db as ffast_mac
from hyperspy.misc.eds import utils as utils_eds


def weight_to_atomic(elements, weight_percent):
    """Convert weight percent (wt%) to atomic percent (at.%).

    Parameters
    ----------
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']

    weight_percent: list of float
        The weight fractions (composition) of the sample.

    Returns
    -------
    atomic_percent : list
        Composition in atomic percent.

    """
    atomic_weights = np.array(
        [elements_db[element]['General_properties']['atomic_weight'] for element in elements])
    atomic_percent = weight_percent / atomic_weights / (
        weight_percent / atomic_weights).sum() * 100
    return atomic_percent.tolist()


def atomic_to_weight(elements, atomic_percent):
    """Convert atomic percent to weight percent.

    Parameters
    ----------
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']

    atomic_percent: list of float
        The atomic fractions (composition) of the sample.

    Returns
    -------
    weight_percent : composition in weight percent.

    """

    atomic_weights = np.array(
        [elements_db[element]['General_properties']['atomic_weight'] for element in elements])

    weight_percent = atomic_percent * atomic_weights / (
        atomic_percent * atomic_weights).sum() * 100
    return weight_percent.tolist()


def density_of_mixture_of_pure_elements(elements, weight_percent):
    """Calculate the density a mixture of elements.

    The density of the elements is retrieved from an internal database. The
    calculation is only valid if there is no interaction between the
    components.

    Parameters
    ----------
    elements: list of str
        A list of element symbols, e.g. ['Al', 'Zn']
    weight_percent: list of float
        A list of weight percent for the different elements. If the total
        is not equal to 100, each weight percent is divided by the sum
        of the list (normalization).

    Returns
    -------
    density: The density in g/cm3.

    Examples
    --------

    Calculate the density of modern bronze given its weight percent:
    >>> utils.material.density_of_mixture_of_pure_elements(("Cu", "Sn"), (88, 12))
    8.6903187973131466

    """
    densities = np.array(
        [elements_db[element]['Physical_properties']['density (g/cm^3)'] for element in elements])
    density = (weight_percent / densities / sum(weight_percent)).sum() ** -1
    return density


def mass_absorption_coefficient(element, energies):
    """
    Get the mass absorption coefficient (mu/rho) of a X-ray(s)

    In a pure material for a Xray(s) of given energy(ies) or given name(s)

    Parameters
    ----------
    element: str
        The element symbol of the absorber, e.g. 'Al'.
    energies: float or list of float or str or list of str
        The energy or energies of the Xray in keV, or the name eg 'Al_Ka'

    Return
    ------
    mass absorption coefficient(s) in cm^2/g
    """

    energies_db = np.array(ffast_mac[element].energies_keV)
    macs = np.array(ffast_mac[element].mass_absorption_coefficient_cm2g)
    if isinstance(energies, str):
        energies = utils_eds._get_energy_xray_line(energies)
    elif hasattr(energies, '__iter__'):
        if isinstance(energies[0], str):
            for i, energy in enumerate(energies):
                energies[i] = utils_eds._get_energy_xray_line(energy)
    index = np.searchsorted(energies_db, energies)
    mac_res = np.exp(np.log(macs[index - 1])
                     + np.log(macs[index] / macs[index - 1])
                     * (np.log(energies / energies_db[index - 1])
                     / np.log(energies_db[index] / energies_db[index - 1])))
    return np.nan_to_num(mac_res)


def mass_absorption_coefficient_of_mixture_of_pure_elements(elements,
                                                            weight_fraction,
                                                            energies):
    """Calculate the mass absorption coefficients a mixture of elements.

    A compound is a mixture of pure elements

    Parameters
    ----------
    elements: list of str
        The list of element symbol of the absorber, e.g. ['Al','Zn'].
    weight_fraction: np.array
        dim = {el,z,y,x} The fraction of elements in the sample by weight
    energies: {float or list of float or str or list of str}
        The energy or energies of the Xray in keV, or the name eg 'Al_Ka'

    Examples
    --------
    >>> utils.material.mass_absorption_coefficient_of_mixture_of_pure_elements(
    >>>     ['Al','Zn'],[0.5,0.5],'Al_Ka')

    Return
    ------
    mass absorption coefficient(s) in cm^2/g

    See also
    --------
    utils.material.mass_absorption_coefficient
    """
    if len(elements) != len(weight_fraction):
        raise ValueError(
            "Elements and weight_fraction should have the same lenght")

    if hasattr(weight_fraction[0], '__iter__'):
        weight_fraction = np.array(weight_fraction)
        mac_res = np.zeros(weight_fraction.shape[1:])
        for element, weight in zip(elements, weight_fraction):
            mac_re = mass_absorption_coefficient(
                element, energies)
            mac_res += mac_re * weight
        return mac_res
    else:
        mac_res = np.array([mass_absorption_coefficient(
            el, energies) for el in elements])
        mac_res = np.dot(weight_fraction, mac_res)
        return mac_res
