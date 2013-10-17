from hyperspy.misc.eds.elements import elements as elements_db

def weigth_to_atomic(elements,compo_wt):
    """Convert weigth percent in atomic percent
    
    Parameters
    ----------
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']
        
    compo_wt: list of float
        The weight fractions (composition) of the sample.
        
    Returns
    -------    
    The atomic fractions (composition)
    """
    tot = 0
    for i, element in enumerate(elements):
        tot = tot + compo_wt[i]/elements_db[element]['A']
    compo_at = []
    for i, element in enumerate(elements):
        compo_at.append(compo_wt[i]/elements_db[element]['A']/tot)
        
    return compo_at
    
def atomic_to_weight(elements,compo_at):
    """Convert atomic percent in weigth percent
    
    Parameters
    ----------
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']
        
    compo_at: list of float
        The atomic fractions (composition) of the sample.
        
    Returns
    -------    
    The weight fractions (composition)
    """
    tot = 0
    for i, element in enumerate(elements):
        tot = tot + compo_at[i]*elements_db[element]['A']
    compo_wt = []
    for i, element in enumerate(elements):
        compo_wt.append(compo_at[i]*elements_db[element]['A']/tot)
        
    return compo_wt
