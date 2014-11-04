from operator import attrgetter

import numpy as np


def _slice_arrays(arr, arr_slices=None, nav=True):
    """ Slices arrays while keeping them as arrays.

    Parameters
    ----------
        arr : numpy.ndarray
            the array to be sliced
        arr_slices : slice
        nav : Bool
            if True, the array will be slices, otherwise passed as-is (i.e. if slicing the navigation or signal axes)
    Returns
    -------
        out : numpy.ndarray
    """
    if nav:
        return np.atleast_1d(arr[tuple(arr_slices[:-1])])
    else:
        return arr


def _slice_all_whitelist(_from, _to, array_slices, isNav):
    """ Shallow-copies all attributes mentioned in a whitelist from one object to another

    Parameters
    ----------
        _from : object
            object from which to copy, must have _from._whitelist dictionary, which keys will be used as references
            what to copy
        _to : object
            object where to copy
        array_slices : slice
            slices how to slice any found arrays if required
        isNav : bool
            if True, found arrays will be sliced, else passed as-is
    """
    for key in _from._whitelist.keys():
        if key.startswith('_init_') or key == '_id_':
            pass
        elif key.startswith('_fn_'):
            attrsetter(_to, key[4:], attrgetter(key[4:])(_from))
        else:
            if isinstance(attrgetter(key)(_from), np.ndarray):
                attrsetter(
                    _to,
                    key,
                    deal_with_arrays(
                        attrgetter(key)(_from),
                        array_slices,
                        isNav))
            else:
                attrsetter(_to, key, attrgetter(key)(_from))

# Will be imported once it is merged


def attrsetter(target, attrs, value):
    """ Like operator.attrgetter, but for setattr - supports "nested" attributes.

        Parameters
        ----------
            target : object
            attrs : string
            value : object

    """
    where = attrs.rfind('.')
    if where != -1:
        target = attrgetter(attrs[:where])(target)
    setattr(target, attrs[where + 1:], value)
