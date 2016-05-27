import numpy as np
import numbers
import warnings

from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.utils import stack


def _weight_to_atomic(weight_percent, elements):
    """Convert weight percent (wt%) to atomic percent (at.%).

    Parameters
    ----------
    weight_percent: array of float
        The weight fractions (composition) of the sample.
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']

    Returns
    -------
    atomic_percent : array of float
        Composition in atomic percent.

    Calculate the atomic percent of modern bronze given its weight percent:
    >>> hs.material.weight_to_atomic((88, 12), ("Cu", "Sn"))
    array([ 93.19698614,   6.80301386])

    """
    if len(elements) != len(weight_percent):
        raise ValueError(
            'The number of elements must match the size of the first axis'
            'of weight_percent.')
    atomic_weights = np.array(
        [elements_db[element]['General_properties']['atomic_weight']
            for element in elements])
    atomic_percent = np.array(
        list(map(np.divide, weight_percent, atomic_weights)))
    sum_weight = atomic_percent.sum(axis=0) / 100.
    for i, el in enumerate(elements):
        atomic_percent[i] /= sum_weight
        atomic_percent[i] = np.where(sum_weight == 0.0, 0.0, atomic_percent[i])
    return atomic_percent


def weight_to_atomic(weight_percent, elements='auto'):
    """Convert weight percent (wt%) to atomic percent (at.%).

    Parameters
    ----------
    weight_percent: list of float or list of signals
        The weight fractions (composition) of the sample.
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']. If elements is
        'auto', take the elements in en each signal metadata of th
        weight_percent list.

    Returns
    -------
    atomic_percent : as weight_percent
        Composition in atomic percent.

    Examples
    --------
    Calculate the atomic percent of modern bronze given its weight percent:
    >>> hs.material.weight_to_atomic((88, 12), ("Cu", "Sn"))
    array([ 93.19698614,   6.80301386])

    """
    from hyperspy.signals import BaseSignal
    elements = _elements_auto(weight_percent, elements)

    if isinstance(weight_percent[0], BaseSignal):
        atomic_percent = stack(weight_percent)
        atomic_percent.data = _weight_to_atomic(
            atomic_percent.data, elements)
        atomic_percent.data = np.nan_to_num(atomic_percent.data)
        atomic_percent = atomic_percent.split()
        return atomic_percent
    else:
        return _weight_to_atomic(weight_percent, elements)


def _atomic_to_weight(atomic_percent, elements):
    """Convert atomic percent to weight percent.

    Parameters
    ----------
    atomic_percent: array
        The atomic fractions (composition) of the sample.
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']

    Returns
    -------
    weight_percent : array of float
        composition in weight percent.

    Examples
    --------
    Calculate the weight percent of modern bronze given its atomic percent:
    >>> hs.material.atomic_to_weight([93.2, 6.8], ("Cu", "Sn"))
    array([ 88.00501989,  11.99498011])

    """
    if len(elements) != len(atomic_percent):
        raise ValueError(
            'The number of elements must match the size of the first axis'
            'of atomic_percent.')
    atomic_weights = np.array(
        [elements_db[element]['General_properties']['atomic_weight']
            for element in elements])
    weight_percent = np.array(
        list(map(np.multiply, atomic_percent, atomic_weights)))
    sum_atomic = weight_percent.sum(axis=0) / 100.
    for i, el in enumerate(elements):
        weight_percent[i] /= sum_atomic
        weight_percent[i] = np.where(sum_atomic == 0.0, 0.0, weight_percent[i])
    return weight_percent


def atomic_to_weight(atomic_percent, elements='auto'):
    """Convert atomic percent to weight percent.

    Parameters
    ----------
    atomic_percent: list of float or list of signals
        The atomic fractions (composition) of the sample.
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']. If elements is
        'auto', take the elements in en each signal metadata of the
        atomic_percent list.

    Returns
    -------
    weight_percent : as atomic_percent
        composition in weight percent.

    Examples
    --------
    Calculate the weight percent of modern bronze given its atomic percent:
    >>> hs.material.atomic_to_weight([93.2, 6.8], ("Cu", "Sn"))
    array([ 88.00501989,  11.99498011])

    """
    from hyperspy.signals import Signal
    elements = _elements_auto(atomic_percent, elements)
    if isinstance(atomic_percent[0], Signal):
        weight_percent = stack(atomic_percent)
        weight_percent.data = _atomic_to_weight(
            weight_percent.data, elements)
        weight_percent = weight_percent.split()
        return weight_percent
    else:
        return _atomic_to_weight(atomic_percent, elements)


def _density_of_mixture_of_pure_elements(weight_percent,
                                         elements,
                                         mean='harmonic'):
    """Calculate the density a mixture of elements.

    The density of the elements is retrieved from an internal database. The
    calculation is only valid if there is no interaction between the
    components.

    Parameters
    ----------
    weight_percent: array
        A list of weight percent for the different elements. If the total
        is not equal to 100, each weight percent is divided by the sum
        of the list (normalization).
    elements: list of str
        A list of element symbols, e.g. ['Al', 'Zn']
    mean: 'harmonic' or 'weighted'
        The type of mean use to estimate the density

    Returns
    -------
    density: The density in g/cm3.

    Examples
    --------
    Calculate the density of modern bronze given its weight percent:
    >>> hs.material.density_of_mixture_of_pure_elements(
            (88, 12),("Cu", "Sn"))
    8.6903187973131466

    """
    if len(elements) != len(weight_percent):
        raise ValueError(
            'The number of elements must match the size of the first axis'
            'of weight_percent.')
    densities = np.array(
        [elements_db[element]['Physical_properties']['density (g/cm^3)']
            for element in elements])
    sum_densities = np.zeros_like(weight_percent, dtype='float')
    if mean == 'harmonic':
        for i, weight in enumerate(weight_percent):
            sum_densities[i] = weight / densities[i]
        sum_densities = sum_densities.sum(axis=0)
        density = np.sum(weight_percent, axis=0) / sum_densities
        return np.where(sum_densities == 0.0, 0.0, density)
    elif mean == 'weighted':
        for i, weight in enumerate(weight_percent):
            sum_densities[i] = weight * densities[i]
        sum_densities = sum_densities.sum(axis=0)
        sum_weight = np.sum(weight_percent, axis=0)
        density = sum_densities / sum_weight
        return np.where(sum_weight == 0.0, 0.0, density)


def density_of_mixture_of_pure_elements(weight_percent,
                                        elements='auto',
                                        mean='harmonic'):
    """Calculate the density of a mixture of elements.

    The density of the elements is retrieved from an internal database. The
    calculation is only valid if there is no interaction between the
    components.

    Parameters
    ----------
    weight_percent: list of float or list of signals
        A list of weight percent for the different elements. If the total
        is not equal to 100, each weight percent is divided by the sum
        of the list (normalization).
    elements: list of str
        A list of element symbols, e.g. ['Al', 'Zn']. If elements is 'auto',
        take the elements in en each signal metadata of the weight_percent
        list.
    mean: 'harmonic' or 'weighted'
        The type of mean use to estimate the density

    Returns
    -------
    density: The density in g/cm3.

    Examples
    --------
    Calculate the density of modern bronze given its weight percent:
    >>> hs.material.density_of_mixture_of_pure_elements(
            (88, 12),("Cu", "Sn"))
    8.6903187973131466

    """
    from hyperspy.signals import Signal
    elements = _elements_auto(weight_percent, elements)
    if isinstance(weight_percent[0], Signal):
        density = weight_percent[0]._deepcopy_with_new_data(
            _density_of_mixture_of_pure_elements(
                stack(weight_percent).data, elements, mean=mean))
        return density
    else:
        return _density_of_mixture_of_pure_elements(weight_percent,
                                                    elements, mean=mean)


def _elements_auto(composition, elements):
    if isinstance(composition[0], numbers.Number):
        if isinstance(elements, str):
            if elements == 'auto':
                raise ValueError("The elements needs to be provided.")
    else:
        if isinstance(elements, str):
            if elements == 'auto':
                elements = []
                for compo in composition:
                    if len(compo.metadata.Sample.elements) > 1:
                        raise ValueError(
                            "The signal %s contains more than one element "
                            "but this function requires only one element "
                            "per signal." % compo.metadata.General.title)
                    else:
                        elements.append(compo.metadata.Sample.elements[0])
    return elements
