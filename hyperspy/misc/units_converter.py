from hyperspy.misc.eds.elements import elements as elements_db

def weigth_to_atomic(elements,compositions):
    """Convert weigth percent in atomic percent
    
    Parameters
    ----------
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']
        
    Composition: list of float
        The weight composition of the sample.
        
    Returns
    -------    
    The atomic composition
    """
    tot = 0
    for i, element in enumerate(elements):
        tot = tot + compositions[i]/elements_db[element]['A']
    atomic_compositions = []
    for i, element in enumerate(elements):
        atomic_compositions.append(compositions[i]/elements_db[element]['A']/tot)
        
    return atomic_compositions 
    
def atomic_to_weight(elements,compositions):
    """Convert atomic percent in weigth percent
    
    Parameters
    ----------
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']
        
    Composition: list of float
        The atomic composition of the sample.
        
    Returns
    -------    
    The weight composition
    """
    tot = 0
    for i, element in enumerate(elements):
        tot = tot + compositions[i]*elements_db[element]['A']
    weight_compositions = []
    for i, element in enumerate(elements):
        weight_compositions.append(compositions[i]*elements_db[element]['A']/tot)
        
    return weight_compositions
