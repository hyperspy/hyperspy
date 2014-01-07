from hyperspy.misc.eds.elements import elements as elements_db


def weight_to_atomic(elements, compo_wt):
    """Convert weight percent (wt%) to atomic percent (at.%).

    Parameters
    ----------
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']

    compo_wt: list of float
        The weight fractions (composition) of the sample.

    Returns
    -------
    compo_at : list
        Composition in atomic percent.

    """

    tot = 0
    for i, element in enumerate(elements):
        tot = tot + compo_wt[i] / elements_db[element]['A']
    compo_at = []
    for i, element in enumerate(elements):
        compo_at.append(compo_wt[i] / elements_db[element]['A'] / tot)

    return compo_at


def atomic_to_weight(elements, compo_at):
    """Convert atomic percent to weight percent.

    Parameters
    ----------
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']

    compo_at: list of float
        The atomic fractions (composition) of the sample.

    Returns
    -------
    compo_wt : composition in weight percent.

    """

    tot = 0
    for i, element in enumerate(elements):
        tot = tot + compo_at[i] * elements_db[element]['A']
    compo_wt = []
    for i, element in enumerate(elements):
        compo_wt.append(compo_at[i] * elements_db[element]['A'] / tot)

    return compo_wt


def density_of_mixture(elements, compositions, compo_unit='at'):
    """Calculate the density a solution from its components.

    Parameters
    ----------
    elements: list of str
        A list of element abbreviations, e.g. ['Al', 'Zn']
    compositions: list of float
        The atomic composition of the sample e.g. [0.2, 0.8]. The composition
        is normalized.

    Returns
    -------
    density: The density in g/cm3.

    """

    if compo_unit == 'at':
        weights = atomic_to_weight(elements, compositions)
    elif compo_unit == 'wt':
        weights = np.array(compositions) / float(sum(compositions))
    density = 0
    for i, element in enumerate(elements):
        density = density + elements_db[element]['density'] * weights[i]

    return density


