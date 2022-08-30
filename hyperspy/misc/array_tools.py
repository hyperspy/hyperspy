# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>


from collections import OrderedDict
import math
import logging

import dask.array as da
import numpy as np
from numba import njit


from hyperspy.misc.math_tools import anyfloatin
from hyperspy.docstrings.utils import REBIN_ARGS


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
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)
    return np.array(shape).cumprod()[-1] * dtype.itemsize / 2.0 ** 30


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

    return [ary.reshape((1,) * (max_len - len(ary.shape)) + ary.shape) for ary in args]


def _requires_linear_rebin(arr, scale):
    """Returns True if linear_rebin is required.
    Parameters
    ----------
    arr: array
        numpy array to rebin
    scale: tuple
        rebinning factors
    """

    return (np.asarray(arr.shape) % np.asarray(scale)).any() or anyfloatin(scale)


def rebin(a, new_shape=None, scale=None, crop=True, dtype=None):
    """Rebin data into a smaller or larger array based on a linear
    interpolation. Specify either a new_shape or a scale. Scale of 1 means no
    binning. Scale less than one results in up-sampling.

    Parameters
    ----------
    a : numpy array
        The array to rebin.
    %s

    Returns
    -------
    numpy array

    Examples
    --------
    >>> a=rand(6,4); b=rebin(a,scale=(3,2))
    >>> a=rand(6); b=rebin(a,scale=(2,))

    Notes
    -----
    Fast ``re_bin`` function Adapted from scipy cookbook
    If rebin function fails with error stating that the function is 'not binned
    and therefore cannot be rebinned', add binned to axes parameters with:
    >>> s.axes_manager[axis].is_binned = True

    """
    # Series of if statements to check that only one out of new_shape or scale
    # has been given. New_shape is then converted to scale. If both or neither
    # are given the function raises and error and wont run.
    if new_shape is None and scale is None:
        raise ValueError("One of new_shape, or scale must be specified")
    elif new_shape is not None and scale is not None:
        raise ValueError("Only one out of new_shape or scale should be specified")
    elif new_shape is not None:
        scale = []
        for i, _ in enumerate(a.shape):
            scale.append(a.shape[i] / new_shape[i])
    if isinstance(dtype, str) and dtype != 'same':
        raise ValueError(
            '`dtype` argument needs to be None, a numpy dtype or '
            'the string "same".'
            )

    # check whether or not interpolation is needed.
    if _requires_linear_rebin(arr=a, scale=scale):
        _logger.debug("Using linear_bin")
        return _linear_bin(a, scale, crop, dtype=dtype)
    else:
        if dtype == 'same':
            dtype = a.dtype
        _logger.debug("Using standard rebin with lazy support")
        # if interpolation is not needed run fast re_bin function.
        # Adapted from scipy cookbook.
        lenShape = len(a.shape)
        new_shape = np.asarray(a.shape) // np.asarray(scale)
        # ensure the new shape is integers
        new_shape = tuple(int(ns) for ns in new_shape)
        # check function wont bin to zero.
        for item in new_shape:
            if item == 0:
                raise ValueError(
                    "One of your dimensions collapses to zero. "
                    "Re-adjust your scale values or run code with "
                    "crop=False to avoid this error."
                )
        scale = np.asarray(a.shape) // np.asarray(new_shape)
        if scale.max() < 2:
            return a.copy()

        if isinstance(a, np.ndarray):
            # most of the operations will fall here and dask is not imported
            rshape = ()
            for athing in zip(new_shape, scale):
                rshape += athing
            return a.reshape(rshape).sum(axis=tuple(
                2 * i + 1 for i in range(lenShape)), dtype=dtype)
        else:
            try:
                return da.coarsen(np.sum, a,
                                  {i: int(f) for i, f in enumerate(scale)},
                                  dtype=dtype)
            # we provide slightly better error message in hyperspy context
            except ValueError:
                raise ValueError(
                    "Rebinning does not align with data dask chunks. "
                    "Rebin fewer dimensions at a time to avoid this error"
                )

# Replacing space is necessary to get the correct indentation
rebin.__doc__ %= REBIN_ARGS.replace("        ", "    ")


@njit(cache=True)
def _linear_bin_loop(result, data, scale):  # pragma: no cover
    for j in range(result.shape[0]):
        # Begin by determining the upper and lower limits of a given new pixel.
        x1 = j * scale
        x2 = min((1 + j) * scale, data.shape[0])
        value = result[j : j + 1]

        if (x2 - x1) >= 1:
            # When binning, the first part is to deal with the fractional pixel
            # left over from it being non-integer binning e.g. when x1=1.4
            cx1 = math.ceil(x1)
            rem = cx1 - x1
            # This will add a value of fractional pixel to the bin, eg if x1=1.4,
            # the fist step will be to add 0.6*data[1]
            value += data[math.floor(x1)] * rem
            # Update x1 to remove the part of the bin we have just added.
            x1 = cx1
            while (x2 - x1) >= 1:
                # Main binning function to add full pixel values to the data.
                value += data[int(x1)]
                # Update x1 each time.
                x1 += 1
            if x2 > x1:
                # Finally take into account the fractional pixel left over.
                value += data[math.floor(x1)] * (x2 - x1)
        else:
            # When step < 1, so we are upsampling
            fx1 = math.floor(x1)
            cx1 = math.ceil(x1)
            if scale > (cx1 - x1) > 0:
                # If our step is smaller than rounding up to the nearest whole
                # number.
                value += data[fx1] * (cx1 - x1)
                x1 = cx1  # This step is needed when this particular bin straddes
                # two neighbouring pixels.
            if x1 < x2:
                # The standard upsampling function where each new pixel is a
                # fraction of the original pixel.
                value += data[math.floor(x1)] * (x2 - x1)


def _linear_bin(dat, scale, crop=True, dtype=None):
    """Binning of the spectrum image by a non-integer pixel value.

    Parameters
    ----------
    originalSpectrum : numpy.array
    scale : a list of floats
        For each dimension specify the new:old pixel ratio,
        e.g. a ratio of 1 is no binning; a ratio of 2 means that each pixel in
        the new spectrum is twice the size of the pixels in the old spectrum.
        The length of the list should match the dimensions of the data.
    crop : bool, default True
        When binning by a non-integer number of pixels it is likely that
        the final row in each dimension contains less than the full quota to
        fill one pixel.
        e.g. 5*5 array binned by 2.1 will produce two rows containing 2.1
        pixels and one row containing only 0.8 pixels worth. Selection of
        crop='True' or crop='False' determines whether or not this 'black'
        line is cropped from the final binned array or not.

        *Please note that if crop=False is used, the final row in each
        dimension may appear black, if a fractional number of pixels are left
        over. It can be removed but has been left optional to preserve total
        counts before and after binning.*

    Returns
    -------
    np.array
        with new dimensions width/scale for each dimension in the data.
    """
    if len(dat.shape) != len(scale):
        raise ValueError(
            "The list of bins must match the number of dimensions, including "
            "the energy dimension. In order to not bin in any of these "
            "dimensions specifically, simply set the value in shape to 1."
        )

    # Unsuported dtype value argument
    dtype_str_same_integer = (isinstance(dtype, str) and dtype == 'same' and
                              np.issubdtype(dat.dtype, np.integer))
    dtype_interger = (not isinstance(dtype, str) and
                      np.issubdtype(dtype, np.integer))

    if dtype_str_same_integer or dtype_interger:
        raise ValueError(
            "Linear interpolation requires float dtype, change the "
            "dtype argument."
            )

    if np.issubdtype(dat.dtype, np.integer):
        # The _linear_bin function below requires a float dtype
        # because of the default numpy casting rule ('same_kind').
        dtype = float
    # Make sure that native endian is used as required by numba.jit
    elif not dat.dtype.isnative:
        dtype = dat.dtype.type

    # If dtype is not None, it means that we need to change it
    if dtype is not None:
        dat = dat.astype(dtype, casting="safe", copy=False)

    for axis, s in enumerate(scale):
        # For each iteration of linear_bin the axis being interated over has to
        # be switched to axis[0] in order to carry out the interation loop.
        dat = np.swapaxes(dat, 0, axis)
        # The new dimension size is old_size/step, this is rounded down normally
        # but if crop is switched off it is rounded up to the nearest whole
        # number.
        if not np.issubdtype(s, np.floating):
            s = float(s)

        dim = math.floor(dat.shape[0] / s) if crop else math.ceil(dat.shape[0] / s)
        # check function wont bin to zero.
        if dim == 0:
            raise ValueError(
                "One of your dimensions collapses to zero. "
                "Re-adjust your scale values or run code with "
                "crop=False to avoid this error."
            )

        # Make sure that native endian is used
        if dtype is None:
            dtype = dat.dtype.type
        # Set up the result np.array to have a new axis[0] size for after
        # cropping.
        result = np.zeros((dim,) + dat.shape[1:], dtype=dtype)

        # Carry out binning over axis[0]
        _linear_bin_loop(result=result, data=dat, scale=s)
        # Swap axis[0] back to the original axis location.
        result = result.swapaxes(0, axis)
        # Update the np.array reading of iterating over the next axis.
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
        dictionary = OrderedDict()
    for name in sarray.dtype.names:
        dictionary[name] = sarray[name][0] if len(sarray[name]) == 1 else sarray[name]
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


def numba_histogram(data, bins, ranges):
    """
    Parameters
    ----------
    data : numpy array
        Input data. The histogram is computed over the flattened array.
    bins : int
        Number of bins
    ranges : (float, float)
        The lower and upper range of the bins.

    Returns
    -------
    hist : array
        The values of the histogram.
    """
    # Make sure that native endian is used
    if not data.dtype.isnative:
        data = data.astype(data.dtype.type)
    return _numba_histogram(data, bins, ranges)


@njit(cache=True)
def _numba_histogram(data, bins, ranges):
    """
    Numba histogram computation requiring native endian datatype.
    """
    # Adapted from https://iscinumpy.gitlab.io/post/histogram-speeds-in-python/
    hist = np.zeros((bins,), dtype=np.intp)
    delta = 1 / ((ranges[1] - ranges[0]) / bins)

    for x in data.flat:
        i = (x - ranges[0]) * delta
        if 0 <= i < bins:
            hist[int(i)] += 1

    return hist


def get_signal_chunk_slice(index, chunks):
    """
    Convenience function returning the chunk slice in signal space containing
    the specified index.

    Parameters
    ----------
    index : int or tuple of int
        Index determining the wanted chunk.
    chunks : tuple
        Dask array chunks.

    Returns
    -------
    slice
        Slice containing the index x,y.

    """
    if not isinstance(index, (list, tuple)):
        index = tuple(index)

    chunk_slice_raw_list = da.core.slices_from_chunks(chunks[-len(index) :])
    chunk_slice_list = []
    for chunk_slice_raw in chunk_slice_raw_list:
        chunk_slice_list.append(list(chunk_slice_raw)[::-1])

    for chunk_slice in chunk_slice_list:
        _slice = chunk_slice
        if _slice[0].start <= index[0] < _slice[0].stop:
            if len(_slice) == 1:
                return chunk_slice
            elif _slice[1].start <= index[1] < _slice[1].stop:
                return chunk_slice
    raise ValueError("Index out of signal range.")


@njit(cache=True)
def numba_closest_index_round(axis_array, value_array):
    """For each value in value_array, find the closest value in axis_array and
    return the result as a numpy array of the same shape as value_array.
    Use round half towards zero strategy for rounding float to interger.

    Parameters
    ----------
    axis_array : numpy array
    value_array : numpy array

    Returns
    -------
    numpy array

    """
    # initialise the index same dimension as input, force type to int
    index_array = np.empty_like(value_array, dtype="uint")
    # assign on flat, iterate on flat.
    rtol = 1e-12
    machineepsilon = np.min(np.abs(np.diff(axis_array))) * rtol
    for i, v in enumerate(value_array.flat):
        index_array.flat[i] = np.abs(axis_array - v + np.sign(v) * machineepsilon).argmin()
    return index_array


@njit(cache=True)
def numba_closest_index_floor(axis_array, value_array):  # pragma: no cover
    """For each value in value_array, find the closest smaller value in
    axis_array and return the result as a numpy array of the same shape
    as value_array.

    Parameters
    ----------
    axis_array : numpy array
    value_array : numpy array

    Returns
    -------
    numpy array

    """
    # initialise the index same dimension as input, force type to int
    index_array = np.empty_like(value_array, dtype="uint")
    # assign on flat, iterate on flat.
    for i, v in enumerate(value_array.flat):
        x = axis_array - v
        index_array.flat[i] = np.where(x > 0, -np.inf, x).argmax()

    return index_array


@njit(cache=True)
def numba_closest_index_ceil(axis_array, value_array):  # pragma: no cover
    """For each value in value_array, find the closest larger value in
    axis_array and return the result as a numpy array of the same shape
    as value_array.

    Parameters
    ----------
    axis_array : numpy array
    value_array : numpy array

    Returns
    -------
    numpy array
    """
    # initialise the index same dimension as input, force type to int
    index_array = np.empty_like(value_array, dtype="uint")
    # assign on flat, iterate on flat.
    for i, v in enumerate(value_array.flat):
        x = axis_array - v
        index_array.flat[i] = np.where(x < 0, +np.inf, x).argmin()

    return index_array


@njit(cache=True)
def round_half_towards_zero(array, decimals=0):  # pragma: no cover
    """
    Round input array using "half towards zero" strategy.

    Parameters
    ----------
    array : ndarray
        Input array.

    decimals : int, optional
        Number of decimal places to round to (default: 0).

    Returns
    -------
    rounded_array : ndarray
        An array of the same type as a, containing the rounded values.
    """
    multiplier = 10 ** decimals

    return np.where(array >= 0,
                    np.ceil(array * multiplier - 0.5) / multiplier,
                    np.floor(array * multiplier + 0.5) / multiplier
                    )


@njit(cache=True)
def round_half_away_from_zero(array, decimals=0):  # pragma: no cover
    """
    Round input array using "half away from zero" strategy.

    Parameters
    ----------
    array : ndarray
        Input array.

    decimals : int, optional
        Number of decimal places to round to (default: 0).

    Returns
    -------
    rounded_array : ndarray
        An array of the same type as a, containing the rounded values.
    """
    multiplier = 10 ** decimals

    return np.where(array >= 0,
                    np.floor(array * multiplier + 0.5) / multiplier,
                    np.ceil(array * multiplier - 0.5) / multiplier
                    )
