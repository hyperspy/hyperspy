import numpy as np

from hyperspy.misc.eds.elements import elements as elements_db

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

    tot = 0
    for i, element in enumerate(elements):
        tot = tot + weight_percent[i] / elements_db[element]['A']
    atomic_percent = []
    for i, element in enumerate(elements):
        atomic_percent.append( 100 * weight_percent[i] / elements_db[element]['A'] / tot)

    return atomic_percent


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

    tot = 0
    for i, element in enumerate(elements):
        tot = tot + atomic_percent[i] * elements_db[element]['A']
    weight_percent = []
    for i, element in enumerate(elements):
        weight_percent.append(
            100 * atomic_percent[i] * elements_db[element]['A'] / tot)

    return weight_percent


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

    Returns
    -------
    density: The density in g/cm3.

    Examples
    --------

    Calculate the density of modern bronze given its weight percent:
    >>> density_of_mixture_of_pure_elements(("Cu", "Sn"), (88, 12))
    8.6903187973131466i

    """
    weights = np.asarray(atomic_percent) / 100.
    density = 0
    for i, element in enumerate(elements):
        density = density + weights[i] / elements_db[element]['density']
    density = density ** -1
    return density


