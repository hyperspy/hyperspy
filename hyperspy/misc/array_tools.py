try:
    from collections import OrderedDict
    ordict = True
except ImportError:
    # happens with Python < 2.7
    ordict = False

import warnings

import numpy as np
import math as math


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
             ['new_shape[%d],factor[%d],' % (i, i) for i in range(lenShape)] +\
             [')'] + ['.sum(%d)' % (i + 1) for i in range(lenShape)]
    return eval(''.join(evList))


def _linear_bin(s, scale,
                crop='on'):

    """
    Binning of the spectrum image by a non-integer pixel value.

    Parameters
    ----------
    originalSpectrum: numpy.array, or the s.data, where s is a signal array.

    scale: a list of floats for each dimension specify the new:old pixel ratio
        e.g. a ratio of 1 is no binning
             a ratio of 2 means that each pixel in the new spectrum is
             twice the size of the pixels in the old spectrum.

    crop: when binning by a non-integer number of pixels it is likely that
         the final row in each dimension contains less than the full quota to
         fill one pixel.
         e.g. 5*5 array binned by 2.1 will produce two rows containing 2.1
         pixels and one row containing only 0.8 pixels worth. Selection of
         crop = 'True' or crop = 'False' determines whether or not this 'black'
         line is cropped from the final binned array or not.

    *Please note that if crop = 'False' is used:the final row in each
    dimension may appear black, if a fractional number of pixels are left
    over. It can be removed but has been left to preserve total counts
    before and after binning.*

    Return
    ------
    An np.array with new dimensions width/scale for each
    dimension in the data.

    """
    def string_to_bool(text):
        return text.lower() in {'yes', 'on', 'true'}
    crop = string_to_bool(crop)

    if len(s.shape) != len(scale):
        raise ValueError(
           'The list of bins must match the number of dimensions, including the\
            energy dimension.\
            In order to not bin in any of these dimensions specifically, \
            simply set the value in shape to 1')

    newSpectrum = s[:]
    for dimension_number, step in enumerate(scale):

        # Is this the same as the original copy operation
        latestSpectrum = np.copy(newSpectrum)
        if dimension_number != 0:
            latestSpectrum = np.swapaxes(s, 0, dimension_number)

        def get_dimension(i, size):
            if i != 0:
                new_size = size
            elif crop:
                new_size = math.floor(size / step)
            else:
                new_size = math.ceil(size / step)
            return new_size

        newSpectrum = np.zeros([get_dimension(i, dimension_size)
                                for i, dimension_size in enumerate(s.shape)],
                               dtype="float")
        k = newSpectrum.shape[0]
        for j in range(k):
            bottomPos = j*step
            topPos = min((1+j)*step, latestSpectrum.shape[0])
            updatedValue = newSpectrum[j]
            while (topPos - bottomPos) >= 1:
                if math.ceil(bottomPos) - bottomPos != 0:
                    updatedValue += latestSpectrum[math.floor(bottomPos)] * \
                               (math.ceil(bottomPos) - bottomPos)
                    bottomPos = math.ceil(bottomPos)
                else:
                    updatedValue += latestSpectrum[int(bottomPos)]
                    bottomPos += 1
            if topPos != bottomPos:
                updatedValue += latestSpectrum[
                                math.floor(bottomPos)]*(topPos-bottomPos)
            # Update new_spectrum
            newSpectrum[j] = updatedValue

        if dimension_number != 0:
            newSpectrum = np.swapaxes(newSpectrum, 0, dimension_number)

    return newSpectrum


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
