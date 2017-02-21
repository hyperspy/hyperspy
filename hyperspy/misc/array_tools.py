try:
    from collections import OrderedDict
    ordict = True
except ImportError:
    # happens with Python < 2.7
    ordict = False

import warnings
import math as math
import logging
import numbers

import numpy as np

_logger = logging.getLogger(__name__)


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
    lenShape = len(a.shape)
    # ensure the new shape is integers
    new_shape = tuple(int(ns) for ns in new_shape)
    factor = np.asarray(a.shape) // np.asarray(new_shape)
    if factor.max() < 2:
        return a.copy()
    if isinstance(a, np.ndarray):
        # most of the operations will fall here and dask is not imported
        rshape = ()
        for athing in zip(new_shape, factor):
            rshape += athing
        return a.reshape(rshape).sum(axis=tuple(
            2 * i + 1 for i in range(lenShape)))
    else:
        import dask.array as da
        try:
            return da.coarsen(np.sum, a, {i: int(f)
                                          for i, f in enumerate(factor)})
        # we provide slightly better error message in hypersy context
        except ValueError:
            raise ValueError("Rebinning does not allign with data dask chunks."
                             " Rebin fewer dimensions at a time to avoid this"
                             " error")


def jit_ifnumba(func):
    try:
        import numba
        return numba.jit(func)
    except ImportError:
        return func


@jit_ifnumba
def _linear_bin_loop(result, data, scale):
    for j in range(result.shape[0]):
        x1 = j * scale
        x2 = min((1 + j) * scale, data.shape[0])
        value = result[j:j + 1]
        if (x2 - x1) >= 1:
            cx1 = math.ceil(x1)
            rem = cx1 - x1
            value += data[math.floor(x1)] * rem
            x1 = cx1
            while (x2 - x1) >= 1:
                value += data[cx1]
                x1 += 1
            if x2 != x1:
                value+= data[math.floor(x1)]*(x2-x1)
        else:
            fx1 = math.floor(x1)
            cx1 = math.ceil(x1)
            if scale > (cx1 - x1) > 0:
                value += data[fx1] * (cx1 - x1)
                x1 = cx1
            if x1 <= (result.shape[0] - 1):
                value += data[math.floor(x1)]*(x2-x1)

def _linear_bin(dat, scale, crop=True):
    """
    Binning of the spectrum image by a non-integer pixel value.
    Parameters
    ----------
    originalSpectrum: numpy.array, or the s.data, where s is a signal array.
    scale: a list of floats for each dimension specify the new:old pixel ratio
        e.g. a ratio of 1 is no binning
             a ratio of 2 means that each pixel in the new spectrum is
             twice the size of the pixels in the old spectrum.
    crop_str: when binning by a non-integer number of pixels it is likely that
         the final row in each dimension contains less than the full quota to
         fill one pixel.
         e.g. 5*5 array binned by 2.1 will produce two rows containing 2.1
         pixels and one row containing only 0.8 pixels worth. Selection of
         crop_str = 'True' or crop = 'False' determines whether or not this
         'black' line is cropped from the final binned array or not.
        *Please note that if crop=False is used, the final row in each
    dimension may appear black, if a fractional number of pixels are left
    over. It can be removed but has been left to preserve total counts
    before and after binning.*
    Return
    ------
    An np.array with new dimensions width/scale for each
    dimension in the data.
    """
    if len(dat.shape) != len(scale):
        raise ValueError(
            'The list of bins must match the number of dimensions, including the\
            energy dimension.\
            In order to not bin in any of these dimensions specifically, \
            simply set the value in shape to 1')

    if not hasattr(_linear_bin_loop, "__numba__"):
        _logger.warning("Install numba to speed up the computation of `rebin`")

    all_integer = np.all([isinstance(n, numbers.Integral) for n in scale])
    dtype = (dat.dtype if (all_integer or "complex" in dat.dtype.name)
             else "float")

    for axis, s in enumerate(scale):
        dat = np.swapaxes(dat, 0, axis)
        dim = (math.floor(dat.shape[0] / s) if crop
               else math.ceil(dat.shape[0] / s))
        result = np.zeros((dim,) + dat.shape[1:], dtype=dtype)
        _linear_bin_loop(result=result, data=dat, scale=s)
        result = result.swapaxes(0, axis)
        dat = result

    return result


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
