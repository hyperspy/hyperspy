try:
    from collections import OrderedDict
    ordict = True
except ImportError:
    # happens with Python < 2.7
    ordict = False

import warnings

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
    # ensure the new shape is integers
    new_shape = tuple(int(ns) for ns in new_shape)
    factor = np.asarray(shape) // np.asarray(new_shape)
    evList = ['a.reshape('] + \
             ['new_shape[%d],factor[%d],' % (i, i) for i in range(lenShape)] + \
             [')'] + ['.sum(%d)' % (i + 1) for i in range(lenShape)]
    return eval(''.join(evList))


def linear_bin(originalSpectrum, scale):

    """
    Binning of the spectrum image by a non-integer pixel value.

    Parameters
    ----------
    originalSpectrum: numpy.array
        the original spectrum
    step: a list of floats for each dimension specify the new:old pixel
    ratio
    e.g. a ratio of 1 is no binning
         a ratio of 2 means that each pixel in the new spectrum is
         twice the size of the pixels in the old spectrum.

    Return
    ------
    numpy.array of the spectrum with new dimensions width/step.
    """

    shape = originalSpectrum.data.shape
    if len(shape) != len(scale):
        raise ValueError(
           'The list of bins must match the number of dimensions, including the energy dimension.'
            'In order to not bin in any of these dimensions specifically, simply set the value in shape to 1')

    spectrum = originalSpectrum.deepcopy()
    s = spectrum.data

    for k, step in enumerate(scale):
        shape = s.shape
        newSpectrum = np.zeros((math.ceil(shape[0]/step), shape[1], shape[2]), dtype='float')
        if k != 0:
            s = np.swapaxes(s, 0, k)
            shape = s.shape
            newSpectrum = np.zeros((math.ceil(shape[0]/step), shape[1], shape[2]), dtype='float')
        for j in range(0, math.ceil(shape[0]/step)):
            bottomPos = (j*step)
            topPos = ((1 + j) * step)
            if topPos > shape[0]:
                topPos = shape[0]
            while (topPos - bottomPos) >= 1:
                if math.ceil(bottomPos) - bottomPos != 0:
                    newSpectrum[j] = newSpectrum[j] + s[math.floor(bottomPos)] * (math.ceil(bottomPos) - bottomPos)
                    bottomPos = math.ceil(bottomPos)
                else:
                    newSpectrum[j] = newSpectrum[j] + s[bottomPos]
                    bottomPos += 1
            if topPos != bottomPos:
                newSpectrum[j] = newSpectrum[j] + s[math.floor(bottomPos)] *(topPos - bottomPos)
        if k != 0:
            newSpectrum = np.swapaxes(newSpectrum, 0, k)
        spectrum = spectrum._deepcopy_with_new_data(newSpectrum)
        s = spectrum.data

    spectrum.get_dimensions_from_data()
    print(scale)
    for s, step in zip(spectrum.axes_manager._axes, scale):
        s.scale /= step

    return spectrum


def sarray2dict(sarray, dictionary=None):
    """Converts a struct array to an ordered dictionary

    Parameters
    ----------
    sarray: struct array
    dictionary: None or dict
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
            warnings.warn(
                "OrderedDict is not available, using a standard dictionary.")
            dictionary = {}
    for name in sarray.dtype.names:
        dictionary[name] = sarray[name][0] if len(sarray[name]) == 1 \
            else sarray[name]
    return dictionary


def dict2sarray(dictionary, sarray=None, dtype=None):
    """Populates a struct array from a dictionary

    Parameters
    ----------
    dictionary: dict
    sarray: struct array or None
        Either sarray or dtype must be given. If sarray is given, it is
        populated from the dictionary.
    dtype: None, numpy dtype or dtype list
        If sarray is None, dtype must be given. If so, a new struct array
        is created according to the dtype, which is then populated.

    Returns
    -------
    Structure array

    """
    if sarray is None:
        if dtype is None:
            raise ValueError("Either sarray or dtype need to be specified.")
        sarray = np.zeros((1,), dtype=dtype)
    for name in set(sarray.dtype.names).intersection(set(dictionary.keys())):
        if len(sarray[name]) == 1:
            sarray[name][0] = dictionary[name]
        else:
            sarray[name] = dictionary[name]
    return sarray
