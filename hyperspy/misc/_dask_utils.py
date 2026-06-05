# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import dask.array as da
import numpy as np
from tqdm.dask import TqdmCallback

from hyperspy.defaults_parser import preferences
from hyperspy.misc.utils import dummy_context_manager


def process_function_blockwise(
    data,
    *args,
    function,
    nav_indexes=None,
    output_signal_size=None,
    output_dtype=None,
    arg_keys=None,
    **kwargs,
):
    """
    Convenience function for processing a function blockwise. By design, its
    output is used as an argument of the dask ``map_blocks`` so that the
    function only gets applied to the signal axes.

    Parameters
    ----------
    data : np.ndarray
        The data for one chunk
    *args : tuple
        Any signal that is iterated alongside the input data. In the form
        ((key1, value1), (key2, value2))
    function : function
        The function to be applied to the signal axis
    nav_indexes : tuple
        The indexes of the navigation axes for the dataset.
    output_signal_size: tuple
        The shape of the output signal. For a ragged signal, this is equal to 1
    output_dtype : dtype
        The data type for the output.
    arg_keys : tuple
        The list of keys for the passed arguments (args).  Together this makes
        a set of key:value pairs to be passed to the function.
    **kwargs : dict
        Any additional key value pairs to be used by the function
        (Note that these are the constants that are applied.)

    """
    # Both of these values need to be passed in
    dtype = output_dtype
    chunk_nav_shape = tuple([data.shape[i] for i in sorted(nav_indexes)])
    output_shape = chunk_nav_shape + tuple(output_signal_size)
    # Pre-allocating the output array
    output_array = np.empty(output_shape, dtype=dtype, like=data)
    if len(args) == 0:
        # There aren't any BaseSignals for iterating
        for nav_index in np.ndindex(chunk_nav_shape):
            islice = np.s_[nav_index]
            output_array[islice] = function(data[islice], **kwargs)
    else:
        # There are BaseSignals which iterate alongside the data
        for index in np.ndindex(chunk_nav_shape):
            islice = np.s_[index]
            iter_dict = {}
            for key, a in zip(arg_keys, args):
                arg_i = np.squeeze(a[islice])
                # Some functions do not handle 0-dimension NumPy arrays
                if hasattr(arg_i, "shape") and arg_i.shape == ():
                    arg_i = arg_i[()]
                iter_dict[key] = arg_i
            output_array[islice] = function(data[islice], **iter_dict, **kwargs)
    if not (chunk_nav_shape == output_array.shape):
        try:
            output_array = output_array.squeeze(-1)
        except ValueError:
            pass
    return output_array


def _get_block_pattern(args, output_shape):
    """Returns the block pattern used by the `blockwise` function for a
    set of arguments given a resulting output_shape

    Parameters
    ----------
    args: list
        A list of all the arguments which are used for `da.blockwise`
    output_shape: tuple
        The output shape for the function passed to `da.blockwise` given args
    """
    arg_patterns = tuple(tuple(range(a.ndim)) for a in args)
    arg_shapes = tuple(a.shape for a in args)
    output_pattern = tuple(range(len(output_shape)))
    all_ind = arg_shapes + (output_shape,)
    max_len = max((len(i) for i in all_ind))  # max number of dimensions
    max_arg_len = max((len(i) for i in arg_shapes))
    adjust_chunks = {}
    new_axis = {}
    output_shape = output_shape + (0,) * (max_len - len(output_shape))
    for i in range(max_len):
        shapes = np.array(
            [s[i] if len(s) > i else -1 for s in (output_shape,) + arg_shapes]
        )
        is_equal_shape = shapes == shapes[0]  # if in shapes == output shapes
        if not all(is_equal_shape):
            if i > max_arg_len - 1:  # output shape is a new axis
                new_axis[i] = output_shape[i]
            else:  # output shape is an existing axis
                adjust_chunks[i] = output_shape[i]  # adjusting chunks based on output
    arg_pairs = [(a, p) for a, p in zip(args, arg_patterns)]
    return arg_pairs, adjust_chunks, new_axis, output_pattern


def guess_output_signal_size(test_data, function, ragged, **kwargs):
    """This function is for guessing the output signal shape and size.
    It will attempt to apply the function to some test data and then output
    the resulting signal shape and datatype.

    Parameters
    ----------
    test_data : NumPy array
        Data from a test signal for the function to be applied to.
        The data must be from a signal with 0 navigation dimensions.
    function : function
        The function to be applied to the data
    ragged : bool
        If the data is ragged then the output signal size is () and the
        data type is 'object'
    **kwargs : dict
        Any other keyword arguments passed to the function.
    """
    if ragged:
        output_dtype = object
        output_signal_size = ()
    else:
        output = function(test_data, **kwargs)
        try:
            output_dtype = output.dtype
            output_signal_size = output.shape
        except AttributeError:
            output = np.asarray(output)
            output_dtype = output.dtype
            output_signal_size = output.shape
    return output_signal_size, output_dtype


def _compute(arrays, store_to=None, show_progressbar=None, **kwargs):
    """Compute a dask array with optional progressbar and storing to disk.

    Parameters
    ----------
    arrays : dask.array or list of dask.array
        The dask array(s) to compute.
    store_to : str or list of str or None
        If not None, the location to store the dask array to disk.
    show_progressbar : bool or None
        Whether to show a progressbar during computation. If None, the value
        from preferences.General.show_progressbar is used.
    **kwargs : dict
        Additional keyword arguments passed to dask.array.compute or

    Returns
    -------
    computed_array : numpy.ndarray or list of numpy.ndarray
    """
    if show_progressbar is None:
        show_progressbar = preferences.General.show_progressbar
    # this isn't compatible with distributed scheduler
    # https://docs.dask.org/en/stable/diagnostics-distributed.html#progress-bar
    cm = TqdmCallback if show_progressbar else dummy_context_manager

    with cm():
        if store_to is not None:
            da.store(arrays, store_to, compute=True, lock=False, **kwargs)
        else:
            return da.compute(arrays, **kwargs)[0]


def _get_navigation_dimension_chunk_slice(navigation_indices, chunks):
    """Get the slice necessary to get the dask data chunk containing the
    navigation indices.

    Parameters
    ----------
    navigation_indices : iterable
    chunks : iterable

    Returns
    -------
    chunk_slice : list of slices

    Examples
    --------
    Making all the variables


    >>> from hyperspy._signals.lazy import _get_navigation_dimension_chunk_slice
    >>> data = da.random.random((128, 128, 256, 256), chunks=(32, 32, 32, 32))
    >>> s = hs.signals.Signal2D(data).as_lazy()

    >>> sig_dim = s.axes_manager.signal_dimension
    >>> nav_chunks = s.data.chunks[:-sig_dim]
    >>> navigation_indices = s.axes_manager._getitem_tuple[:-sig_dim]

    The navigation index here is (0, 0), giving us the slice which contains
    this index.

    >>> chunk_slice = _get_navigation_dimension_chunk_slice(navigation_indices, nav_chunks)
    >>> print(chunk_slice)
    (slice(0, 32, None), slice(0, 32, None))
    >>> data_chunk = data[chunk_slice]

    Moving the navigator to a new position, by directly setting the indices.
    Normally, this is done by moving the navigator while plotting the data.
    Note the "inversion" of the axes here: the indices is given in (x, y),
    while the chunk_slice is given in (y, x).

    >>> s.axes_manager.indices = (127, 70)
    >>> navigation_indices = s.axes_manager._getitem_tuple[:-sig_dim]
    >>> chunk_slice = _get_navigation_dimension_chunk_slice(navigation_indices, nav_chunks)
    >>> print(chunk_slice)
    (slice(64, 96, None), slice(96, 128, None))
    >>> data_chunk = data[chunk_slice]

    """
    chunk_slice_list = da.core.slices_from_chunks(chunks)
    for chunk_slice in chunk_slice_list:
        is_slice = True
        for index_nav in range(len(navigation_indices)):
            temp_slice = chunk_slice[index_nav]
            nav = navigation_indices[index_nav]
            if not (temp_slice.start <= nav < temp_slice.stop):
                is_slice = False
                break
        if is_slice:
            return chunk_slice

    return False


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


def get_chunk_slice(
    shape,
    signal_dimension,
    chunks="auto",
    block_size_limit=None,
    dtype=None,
):
    """
    Takes a shape and chunks and returns an array of the slices to be used with
    :func:`dask.array.map_blocks`.

    Parameters
    ----------
    shape : tuple
        Shape of the data.
    signal_dimension : int
        The signal dimension of the signal.
    chunks : "auto", "dask_auto" or tuple
        If ``"auto"``, no chunking is created in the signal dimension. If ``"dask_auto"``, the
        dask "auto" chunking is used - see :func:`dask.array.core.normalize_chunks` for more
        information. The default is "auto".
    block_size_limit : int, optional
        Maximum size of a block in bytes. The default is None. This is passed
        to the :func:`dask.array.core.normalize_chunks` function when chunks == "auto".
    dtype : numpy.dtype, optional
        Data type. The default is None. This is passed to the
        :func:`dask.array.core.normalize_chunks` function when chunks == "auto".

    Returns
    -------
    numpy.ndarray of slices
        Dask array of the slices.
    tuple
        Tuple of the chunks.

    Note
    ----
    Adapted from https://github.com/hyperspy/rosettasciio/blob/main/rsciio/utils/distributed.py
    """
    if chunks == "auto":
        # no chunking along signal_dimension
        chunks = ("auto",) * len(shape[:-signal_dimension]) + (-1,)
    elif chunks == "dask_auto":
        # Use dask auto
        chunks = "auto"

    chunks = da.core.normalize_chunks(
        chunks=chunks, shape=shape, limit=block_size_limit, dtype=dtype
    )
    chunks_shape = tuple([len(c) for c in chunks])
    slices = np.empty(
        shape=chunks_shape + (len(chunks_shape), 2),
        dtype=int,
    )
    for ind in np.ndindex(chunks_shape):
        current_chunk = [chunk[i] for i, chunk in zip(ind, chunks)]
        starts = [int(np.sum(chunk[:i])) for i, chunk in zip(ind, chunks)]
        stops = [s + c for s, c in zip(starts, current_chunk)]
        slices[ind] = [[start, stop] for start, stop in zip(starts, stops)]

    return slices, chunks
