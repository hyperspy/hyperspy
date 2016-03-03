try:
    from collections import OrderedDict
    ordict = True
except ImportError:
    # happens with Python < 2.7
    ordict = False

import numpy as np


def get_array_memory_size_in_GiB(shape, dtype):
    """Given the size and dtype returns the amount of memory that such
    an array needs to allocate

    Parameters
    ----------
    shape: tuple
    dtype : data-type
        The desired data-type for the array.
    """
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)
    return np.array(shape).cumprod()[-1] * dtype.itemsize / 2. ** 30


def are_aligned(shape1, shape2):
    """Check if two numpy arrays are aligned.

    Parameters
    ----------
    shape1, shape2 : iterable

    Returns
    -------
    isaligned : bool

    """
    isaligned = True
    shape1 = list(shape1)
    shape2 = list(shape2)
    try:
        while isaligned is True:
            dim1 = shape1.pop()
            dim2 = shape2.pop()
            if dim1 != dim2 and (dim1 != 1 and dim2 != 1):
                isaligned = False
    except IndexError:
        return isaligned
    return isaligned


def homogenize_ndim(*args):
    """Given any number of arrays returns the same arrays
    reshaped by adding facing dimensions of size 1.

    """

    max_len = max([len(ary.shape) for ary in args])

    return [ary.reshape((1,) * (max_len - len(ary.shape)) + ary.shape)
            for ary in args]


def rebin(a, new_shape):
    """Rebin array.

    rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4
    rows can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.

    Parameters
    ----------
    a : numpy array
    new_shape : tuple
        shape after binning

    Returns
    -------
    numpy array

    Examples
    --------
    >>> a=rand(6,4); b=rebin(a,(3,2))
    >>> a=rand(6); b=rebin(a,(2,))

    Notes
    -----
    Adapted from scipy cookbook

    """
    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape) / np.asarray(new_shape)
    evList = ['a.reshape('] + \
             ['new_shape[%d],factor[%d],' % (i, i) for i in range(lenShape)] + \
             [')'] + ['.sum(%d)' % (i + 1) for i in range(lenShape)]
    return eval(''.join(evList))


def sarray2dict(sarray, dictionary=None):
    """Converts a struct array to an ordered dictionary

    Parameters
    ----------
    sarray: struct array
    dictionary: None or dic
        If dictionary is not None the content of sarray will be appended to the
        given dictonary

    Returns
    -------
    Ordered dictionary

    """
    if dictionary is None:
        if ordict:
            dictionary = OrderedDict()
        else:
            print("\nWARNING:")
            print("sarray2dict")
            print(
                "OrderedDict is not available, using a standard dictionary.\n")
            dictionary = {}
    for name in sarray.dtype.names:
        dictionary[name] = sarray[name][0] if len(sarray[name]) == 1 \
            else sarray[name]
    return dictionary
