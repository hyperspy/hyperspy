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

import importlib
import logging
import os
from functools import partial
from itertools import product

import numpy as np
import scipy
from dask.widgets import TEMPLATE_PATHS
from rsciio.utils import file

from hyperspy import signals
from hyperspy.docstrings.signal import (
    LAZYSIGNAL_DOC,
    MANY_AXIS_PARAMETER,
    RECHUNK_ARG,
    SHOW_PROGRESSBAR_ARG,
)
from hyperspy.external.progressbar import progressbar
from hyperspy.misc import array_tools, dask_utils, utils
from hyperspy.misc.hist_tools import _set_histogram_metadata, histogram_dask

SKLEARN_INSTALLED = importlib.util.find_spec("sklearn") is not None

_logger = logging.getLogger(__name__)

lazyerror = NotImplementedError("This method is not available in lazy signals")

templates_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "misc", "dask_widgets"
)
TEMPLATE_PATHS.append(templates_path)


def _get():
    import dask

    try:
        get = dask.threaded.get
    except AttributeError:  # pragma: no cover
        # For pyodide
        get = dask.get
        _logger.warning(
            "Dask scheduler with threads is not available in this environment. "
            "Falling back to synchronous scheduler (single-threaded)."
        )
    return get


def to_array(thing, chunks=None):
    """Accepts BaseSignal, dask or numpy arrays and always produces either
    numpy or dask array.

    Parameters
    ----------
    thing : {BaseSignal, dask.array.Array, numpy.ndarray}
        the thing to be converted
    chunks : {None, tuple of tuples}
        If None, the returned value is a numpy array. Otherwise returns dask
        array with the chunks as specified.

    Returns
    -------
    res : {numpy.ndarray, dask.array.Array}
    """
    if thing is None:
        return None
    if isinstance(thing, signals.BaseSignal):
        thing = thing.data
    if chunks is None:
        if utils.is_dask_array(thing):
            thing = thing.compute()
        if isinstance(thing, np.ndarray):
            return thing
        else:
            raise ValueError
    else:
        if isinstance(thing, np.ndarray):
            import dask.array as da

            thing = da.from_array(thing, chunks=chunks)
        if utils.is_dask_array(thing):
            if thing.chunks != chunks:
                thing = thing.rechunk(chunks)
            return thing
        else:
            raise ValueError


class LazySignal(signals.BaseSignal):
    """Lazy general signal class."""

    _lazy = True
    __doc__ += LAZYSIGNAL_DOC.replace("__BASECLASS__", "BaseSignal")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The _cache_dask_chunk and _cache_dask_chunk_slice attributes are
        # used to temporarily cache data contained in one chunk, when
        # self.__call__ is used. Typically done when using plot or fitting.
        # _cache_dask_chunk has the NumPy array itself, while
        # _cache_dask_chunk_slice has the navigation dimension chunk which
        # the NumPy array originates from.
        self._cache_dask_chunk = None
        self._cache_dask_chunk_slice = None
        if self._clear_cache_dask_data not in self.events.data_changed.connected:
            self.events.data_changed.connect(self._clear_cache_dask_data)

    __init__.__doc__ = signals.BaseSignal.__init__.__doc__.replace(
        ":class:`numpy.ndarray`", ":class:`dask.array.Array`"
    )

    def _repr_html_(self):
        try:
            from dask import config
            from dask.array.svg import svg
            from dask.utils import format_bytes
            from dask.widgets import get_template

            nav_chunks = self.get_chunk_size(self.axes_manager.navigation_axes)
            sig_chunks = self.get_chunk_size(self.axes_manager.signal_axes)
            if nav_chunks == ():
                nav_grid = ""
            else:
                nav_grid = svg(
                    chunks=nav_chunks, size=config.get("array.svg.size", 160)
                )
            if sig_chunks == ():
                sig_grid = ""
            else:
                sig_grid = svg(
                    chunks=sig_chunks, size=config.get("array.svg.size", 160)
                )
            nbytes = format_bytes(self.data.nbytes)
            cbytes = format_bytes(
                np.prod(self.data.chunksize) * self.data.dtype.itemsize
            )
            return get_template("lazy_signal.html.j2").render(
                nav_grid=nav_grid,
                sig_grid=sig_grid,
                dim=self.axes_manager._get_dimension_str(),
                chunks=self._get_chunk_string(),
                array=self.data,
                signal_type=self._signal_type,
                nbytes=nbytes,
                cbytes=cbytes,
                title=self.metadata.General.title,
            )

        except ModuleNotFoundError:
            return self

    def _get_chunk_string(self):
        nav_chunks = self.data.chunksize[: len(self.axes_manager.navigation_shape)][
            ::-1
        ]
        string = "("
        for chunks, axis in zip(nav_chunks, self.axes_manager.navigation_shape):
            if chunks == axis:
                string += "<b>" + str(chunks) + "</b>,"
            else:
                string += str(chunks) + ","
        string = string.rstrip(",")
        string += "|"

        sig_chunks = self.data.chunksize[len(self.axes_manager.navigation_shape) :][
            ::-1
        ]
        for chunks, axis in zip(sig_chunks, self.axes_manager.signal_shape):
            if chunks == axis:
                string += "<b>" + str(chunks) + "</b>,"
            else:
                string += str(chunks) + ","
        string = string.rstrip(",")
        string += ")"
        return string

    def compute(self, close_file=False, show_progressbar=None, **kwargs):
        """
        Attempt to store the full signal in memory.

        Parameters
        ----------
        close_file : bool, default False
            If True, attempt to close the file associated with the dask
            array data if any. Note that closing the file will make all other
            associated lazy signals inoperative.
        %s
        **kwargs : dict
            Any other keyword arguments for :meth:`dask.array.Array.compute`.
            For example `scheduler` or `num_workers`.

        Returns
        -------
        None

        Notes
        -----
        For alternative ways to set the compute settings see
        https://docs.dask.org/en/stable/scheduling.html#configuration

        Examples
        --------
        >>> import dask.array as da
        >>> data = da.zeros((100, 100, 100), chunks=(10, 20, 20))
        >>> s = hs.signals.Signal2D(data).as_lazy()

        With default parameters

        >>> s1 = s.deepcopy()
        >>> s1.compute()

        Using 2 workers, which can reduce the memory usage (depending on
        the data and your computer hardware). Note that `num_workers` only
        work for the 'threads' and 'processes' `scheduler`.

        >>> s2 = s.deepcopy()
        >>> s2.compute(num_workers=2)

        Using a single threaded scheduler, which is useful for debugging

        >>> s3 = s.deepcopy()
        >>> s3.compute(scheduler='single-threaded')

        """
        self.data = dask_utils._compute(
            self.data, show_progressbar=show_progressbar, **kwargs
        )
        if close_file:
            self.close_file()

        self._lazy = False
        self._assign_subclass()

    compute.__doc__ %= SHOW_PROGRESSBAR_ARG

    def rechunk(self, nav_chunks="auto", sig_chunks=-1, inplace=True, **kwargs):
        """
        Rechunks the data using the same rechunking formula from Dask
        expect that the navigation and signal chunks are defined seperately.
        Note, for most functions sig_chunks should remain ``None`` so that it
        spans the entire signal axes.

        Parameters
        ----------
        nav_chunks : {tuple, int, "auto"}
            The navigation block dimensions to create.
            -1 indicates the full size of the corresponding dimension.
            Default is “auto” which automatically determines chunk sizes.
        sig_chunks : {tuple, int, "auto"}
            The signal block dimensions to create.
            -1 indicates the full size of the corresponding dimension.
            Default is -1 which automatically spans the full signal dimension
        **kwargs : dict
            Any other keyword arguments for :func:`dask.array.rechunk`.
        """
        if not isinstance(sig_chunks, tuple):
            sig_chunks = (sig_chunks,) * len(self.axes_manager.signal_shape)
        if not isinstance(nav_chunks, tuple):
            nav_chunks = (nav_chunks,) * len(self.axes_manager.navigation_shape)

        data_ = self.data.rechunk(nav_chunks + sig_chunks, **kwargs)
        if inplace:
            self.data = data_
        else:
            return self._deepcopy_with_new_data(data_)

    def close_file(self):
        """Closes the associated data file if any.

        Currently it only supports closing the file associated with a dask
        array created from an h5py DataSet (default HyperSpy hdf5 reader).

        """
        try:
            file.get_file_handle(self.data).close()
        except AttributeError:
            _logger.warning("Failed to close lazy signal file")

    def _clear_cache_dask_data(self, obj=None):
        self._cache_dask_chunk = None
        self._cache_dask_chunk_slice = None

    def _get_dask_chunks(self, axis=None, dtype=None):
        """Returns dask chunks.

        Aims:
            - Have at least one signal (or specified axis) in a single chunk,
              or as many as fit in memory

        Parameters
        ----------
        axis : {int, string, None, axis, tuple}
            If axis is None (default), returns chunks for current data shape so
            that at least one signal is in the chunk. If an axis is specified,
            only that particular axis is guaranteed to be "not sliced".
        dtype : {string, np.dtype}
            The dtype of target chunks.

        Returns
        -------
        Tuple of tuples, dask chunks
        """
        dc = self.data
        dcshape = dc.shape
        for _axis in self.axes_manager._axes:
            if _axis.index_in_array < len(dcshape):
                _axis.size = int(dcshape[_axis.index_in_array])

        if axis is not None:
            need_axes = self.axes_manager[axis]
            if not np.iterable(need_axes):
                need_axes = [
                    need_axes,
                ]
        else:
            need_axes = self.axes_manager.signal_axes

        if dtype is None:
            dtype = dc.dtype
        elif not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        typesize = max(dtype.itemsize, dc.dtype.itemsize)
        want_to_keep = utils.multiply([ax.size for ax in need_axes]) * typesize

        # @mrocklin reccomends to have around 100MB chunks, so we do that:
        num_that_fit = int(100.0 * 2.0**20 / want_to_keep)

        # want to have at least one "signal" per chunk
        if num_that_fit < 2:
            chunks = [tuple(1 for _ in range(i)) for i in dc.shape]
            for ax in need_axes:
                chunks[ax.index_in_array] = (dc.shape[ax.index_in_array],)
            return tuple(chunks)

        sizes = [ax.size for ax in self.axes_manager._axes if ax not in need_axes]
        indices = [
            ax.index_in_array for ax in self.axes_manager._axes if ax not in need_axes
        ]

        while True:
            if utils.multiply(sizes) <= num_that_fit:
                break

            i = np.argmax(sizes)
            sizes[i] = np.floor(sizes[i] / 2)
        chunks = []
        ndim = len(dc.shape)
        for i in range(ndim):
            if i in indices:
                size = float(dc.shape[i])
                split_array = np.array_split(
                    np.arange(size), np.ceil(size / sizes[indices.index(i)])
                )
                chunks.append(tuple(len(sp) for sp in split_array))
            else:
                chunks.append((dc.shape[i],))
        return tuple(chunks)

    def get_chunk_size(self, axes=None):
        """
        Returns the chunk size as tuple for a set of given axes. The order
        of the returned tuple follows the order of the dask array.

        Parameters
        ----------
        axes : %s

        Examples
        --------
        >>> import dask.array as da
        >>> data = da.random.random((10, 200, 300))
        >>> data.chunksize
        (10, 200, 300)
        >>> s = hs.signals.Signal1D(data).as_lazy()
        >>> s.get_chunk_size() # All navigation axes
        ((10,), (200,))
        >>> s.get_chunk_size(0) # The first navigation axis
        ((200,),)
        """
        if axes is None:
            axes = self.axes_manager.navigation_axes

        axes = self.axes_manager[axes]

        if not np.iterable(axes):
            axes = (axes,)

        axes = tuple([axis.index_in_array for axis in axes])
        ax_chunks = tuple([self.data.chunks[i] for i in sorted(axes)])

        return ax_chunks

    get_chunk_size.__doc__ %= MANY_AXIS_PARAMETER

    def _lazy_data(self, axis=None, rechunk=False, dtype=None):
        """
        Return the data as a dask array, rechunked if necessary.

        Parameters
        ----------
        axis : None, :class:`~.axes.DataAxis` or tuple of data axes
            The data axis that must not be broken into chunks when `rechunk`
            is ``True``. If None, it defaults to the current signal axes.
        %s
        dtype : numpy.dtype
            The array dtype used to calculate chunking.

        Returns
        -------
        dask.array
            The data as dask array and rechunked if necessary.
        """
        import dask.array as da

        if rechunk == "dask_auto":
            new_chunks = "auto"
        elif isinstance(rechunk, tuple):
            new_chunks = rechunk
        elif isinstance(rechunk, bool) or rechunk == "auto":
            # when rechunk is False, still need new_chunks
            # da.from_array call in case of numpy array
            new_chunks = self._get_dask_chunks(axis=axis, dtype=dtype)
        else:
            raise ValueError(
                "`rechunk` argument must be a tuple, a boolean or "
                "a str ('auto' or 'dask_auto') "
            )
        if utils.is_dask_array(self.data):
            res = self.data
            # rechunk when necessary when rechunk is True, "auto" or "dask_auto"
            if rechunk and res.chunks != new_chunks:
                _logger.info("Rechunking.\nOriginal chunks: %s." % str(res.chunks))
                res = self.data.rechunk(new_chunks)
                _logger.info("Final chunks: %s." % str(res.chunks))
        else:
            if isinstance(self.data, np.ma.masked_array):
                data = np.where(self.data.mask, np.nan, self.data)
            else:
                data = self.data
            res = da.from_array(data, chunks=new_chunks)

        return res

    _lazy_data.__doc__ %= RECHUNK_ARG

    def _apply_function_on_data_and_remove_axis(
        self, function, axes, out=None, rechunk=False
    ):
        import dask.array as da

        def get_dask_function(numpy_name):
            # Translate from the default numpy to dask functions
            translations = {"amax": "max", "amin": "min"}
            if numpy_name in translations:
                numpy_name = translations[numpy_name]
            return getattr(da, numpy_name)

        function = get_dask_function(function.__name__)
        axes = self.axes_manager[axes]
        if not np.iterable(axes):
            axes = (axes,)
        ar_axes = tuple(ax.index_in_array for ax in axes)
        if len(ar_axes) == 1:
            ar_axes = ar_axes[0]
        # For reduce operations the actual signal and navigation
        # axes configuration does not matter. Hence we leave
        # dask guess the chunks
        if rechunk is True:
            rechunk = "dask_auto"
        current_data = self._lazy_data(rechunk=rechunk)
        # Apply reducing function
        new_data = function(current_data, axis=ar_axes)
        if not new_data.ndim:
            new_data = new_data.reshape((1,))
        if out:
            if out.data.shape == new_data.shape:
                out.data = new_data
                out.events.data_changed.trigger(obj=out)
            else:
                raise ValueError(
                    "The output shape %s does not match  the shape of "
                    "`out` %s" % (new_data.shape, out.data.shape)
                )
        else:
            s = self._deepcopy_with_new_data(new_data)
            s._remove_axis([ax.index_in_axes_manager for ax in axes])
            return s

    def _get_cache_dask_chunk(self, indices):
        """Method for handling caching of dask chunks, when using __call__.

        When accessing data in a chunked HDF5 file, the whole chunks needs
        to be loaded into memory. So even if you only want to access a single
        index in the navigation dimension, the whole chunk in the navigation
        dimension needs to be loaded into memory. This method keeps (caches)
        this chunk in memory after loading it, so moving to a different
        position with the same chunk will be much faster, reducing amount of
        data which needs be read from the disk.

        If a navigation index (via the indices parameter) in a different chunk
        is asked for, the currently cached chunk is discarded, and the new
        chunk is loaded into memory.

        This only works for functions using self.__call__, for example
        plot and fitting functions. This will not work with the region of
        interest functionality.

        The cached chunk is stored in the attribute s._cache_dask_chunk,
        and the slice needed to extract this chunk is in
        s._cache_dask_chunk_slice. To these, use s._clear_cache_dask_data()

        Parameters
        ----------
        indices : tuple
            Must be the same length as navigation dimensions in self.

        Returns
        -------
        value : NumPy array
            Same shape as the signal shape of self.

        Examples
        --------
        >>> import dask.array as da
        >>> s = hs.signals.Signal2D(da.ones((5, 10, 20, 30, 40))).as_lazy()
        >>> value = s._get_cache_dask_chunk((3, 6, 2))
        >>> cached_chunk = s._cache_dask_chunk # Cached array
        >>> cached_chunk_slice = s._cache_dask_chunk_slice # Slice of chunk
        >>> s._clear_cache_dask_data() # Clearing both of these

        """

        sig_dim = self.axes_manager.signal_dimension
        chunks = self.get_chunk_size(self.axes_manager.navigation_axes)
        navigation_indices = indices[:-sig_dim]
        chunk_slice = dask_utils._get_navigation_dimension_chunk_slice(
            navigation_indices, chunks
        )

        if (
            chunk_slice != self._cache_dask_chunk_slice
            or self._cache_dask_chunk is None
        ):
            self._cache_dask_chunk = self.data.__getitem__(chunk_slice).compute()
            self._cache_dask_chunk_slice = chunk_slice

        indices = list(indices)
        for i, temp_slice in enumerate(chunk_slice):
            indices[i] -= temp_slice.start
        indices = tuple(indices)
        value = self._cache_dask_chunk[indices]
        return value

    def rebin(
        self,
        new_shape=None,
        scale=None,
        crop=False,
        dtype=None,
        out=None,
        rechunk=False,
    ):
        factors = self._validate_rebin_args_and_get_factors(
            new_shape=new_shape, scale=scale
        )
        if array_tools._requires_linear_rebin(arr=self.data, scale=factors):
            if new_shape:
                raise NotImplementedError(
                    "Lazy rebin requires that the new shape is a divisor "
                    "of the original signal shape e.g. if original shape "
                    "(10| 6), new_shape=(5| 3) is valid, (3 | 4) is not."
                )
            else:
                raise NotImplementedError(
                    "Lazy rebin requires scale to be integer and divisor of the "
                    "original signal shape"
                )
        axis = {ax.index_in_array: ax for ax in self.axes_manager._axes}[
            factors.argmax()
        ]
        self.data = self._lazy_data(axis=axis, rechunk=rechunk)
        return super().rebin(
            new_shape=new_shape, scale=scale, crop=crop, dtype=dtype, out=out
        )

    rebin.__doc__ = signals.BaseSignal.rebin.__doc__

    def __array__(self, dtype=None, copy=None):
        return self.data.__array__(dtype=dtype, copy=copy)

    def _make_sure_data_is_contiguous(self):
        self.data = self._lazy_data(rechunk=True)

    def diff(self, axis, order=1, out=None, rechunk=False):
        if not self.axes_manager[axis].is_uniform:
            raise NotImplementedError(
                "Performing a numerical difference on a non-uniform axis "
                "is not implemented. Consider using `derivative` instead."
            )
        arr_axis = self.axes_manager[axis].index_in_array

        def dask_diff(arr, n, axis):
            # assume arr is da.Array already
            n = int(n)
            if n == 0:
                return arr
            if n < 0:
                raise ValueError("order must be positive")
            nd = len(arr.shape)
            slice1 = [slice(None)] * nd
            slice2 = [slice(None)] * nd
            slice1[axis] = slice(1, None)
            slice2[axis] = slice(None, -1)
            slice1 = tuple(slice1)
            slice2 = tuple(slice2)
            if n > 1:
                return dask_diff(arr[slice1] - arr[slice2], n - 1, axis=axis)
            else:
                return arr[slice1] - arr[slice2]

        current_data = self._lazy_data(axis=axis, rechunk=rechunk)
        new_data = dask_diff(current_data, order, arr_axis)
        if not new_data.ndim:
            new_data = new_data.reshape((1,))

        s = out or self._deepcopy_with_new_data(new_data)
        if out:
            if out.data.shape == new_data.shape:
                out.data = new_data
            else:
                raise ValueError(
                    "The output shape %s does not match  the shape of "
                    "`out` %s" % (new_data.shape, out.data.shape)
                )
        axis2 = s.axes_manager[axis]
        new_offset = self.axes_manager[axis].offset + (order * axis2.scale / 2)
        axis2.offset = new_offset
        s.get_dimensions_from_data()
        if out is None:
            return s
        else:
            out.events.data_changed.trigger(obj=out)

    diff.__doc__ = signals.BaseSignal.diff.__doc__

    def integrate_simpson(self, axis, out=None, rechunk=False):
        axis = self.axes_manager[axis]

        axis = self.axes_manager[axis]
        data = self._lazy_data(axis=axis, rechunk=rechunk)
        new_data = data.map_blocks(
            scipy.integrate.simpson,
            x=axis.axis,
            axis=axis.index_in_array,
            drop_axis=axis.index_in_array,
            dtype=data.dtype,
        )
        s = out or self._deepcopy_with_new_data(new_data)
        if out:
            if out.data.shape == new_data.shape:
                out.data = new_data
                out.events.data_changed.trigger(obj=out)
            else:
                raise ValueError(
                    "The output shape %s does not match  the shape of "
                    "`out` %s" % (new_data.shape, out.data.shape)
                )
        else:
            s._remove_axis(axis.index_in_axes_manager)
            return s

    integrate_simpson.__doc__ = signals.BaseSignal.integrate_simpson.__doc__

    def valuemax(self, axis, out=None, rechunk=False):
        idx = self.indexmax(axis, rechunk=rechunk)
        old_data = idx.data
        data = old_data.map_blocks(lambda x: self.axes_manager[axis].index2value(x))
        if out is None:
            idx.data = data
            return idx
        else:
            out.data = data
            out.events.data_changed.trigger(obj=out)

    valuemax.__doc__ = signals.BaseSignal.valuemax.__doc__

    def valuemin(self, axis, out=None, rechunk=False):
        idx = self.indexmin(axis, rechunk=rechunk)
        old_data = idx.data
        data = old_data.map_blocks(lambda x: self.axes_manager[axis].index2value(x))
        if out is None:
            idx.data = data
            return idx
        else:
            out.data = data
            out.events.data_changed.trigger(obj=out)

    valuemin.__doc__ = signals.BaseSignal.valuemin.__doc__

    def get_histogram(
        self, bins="fd", range_bins=None, out=None, rechunk=False, **kwargs
    ):
        data = self._lazy_data(rechunk=rechunk).flatten()
        hist, bin_edges = histogram_dask(data, bins=bins, range=range_bins, **kwargs)
        if out is None:
            hist_spec = signals.Signal1D(hist)
            hist_spec._lazy = True
            hist_spec._assign_subclass()
        else:
            hist_spec = out
            # we always overwrite the data because the computation is lazy ->
            # the result signal is lazy. Assume that the `out` is already lazy
            hist_spec.data = hist

        hist_spec.axes_manager[0].scale = bin_edges[1] - bin_edges[0]
        hist_spec.axes_manager[0].offset = bin_edges[0]
        hist_spec.axes_manager[0].size = hist.shape[-1]

        _set_histogram_metadata(self, hist_spec, **kwargs)

        if out is None:
            return hist_spec
        else:
            out.events.data_changed.trigger(obj=out)

    get_histogram.__doc__ = signals.BaseSignal.get_histogram.__doc__

    @staticmethod
    def _estimate_poissonian_noise_variance(
        dc, gain_factor, gain_offset, correlation_factor
    ):
        import dask.array as da

        variance = (dc * gain_factor + gain_offset) * correlation_factor
        # The lower bound of the variance is the gaussian noise.
        variance = da.clip(variance, gain_offset * correlation_factor, np.inf)
        return variance

    # def _get_navigation_signal(self, data=None, dtype=None):
    # return super()._get_navigation_signal(data=data, dtype=dtype).as_lazy()

    # _get_navigation_signal.__doc__ = signals.BaseSignal._get_navigation_signal.__doc__

    # def _get_signal_signal(self, data=None, dtype=None):
    #     return super()._get_signal_signal(data=data, dtype=dtype).as_lazy()

    # _get_signal_signal.__doc__ = signals.BaseSignal._get_signal_signal.__doc__

    def _calculate_summary_statistics(self, rechunk=False):
        import dask.array as da

        if rechunk is True:
            # Use dask auto rechunk instead of HyperSpy's one, what should be
            # better for these operations
            rechunk = "dask_auto"
        data = self._lazy_data(rechunk=rechunk)
        _raveled = data.ravel()
        _mean, _std, _min, _q1, _q2, _q3, _max = da.compute(
            da.nanmean(data),
            da.nanstd(data),
            da.nanmin(data),
            da.percentile(
                _raveled,
                [
                    25,
                ],
            ),
            da.percentile(
                _raveled,
                [
                    50,
                ],
            ),
            da.percentile(
                _raveled,
                [
                    75,
                ],
            ),
            da.nanmax(data),
        )
        # unlike np.percentile, da.percentile returns array
        _q1 = _q1 if np.isscalar(_q1) else _q1[0]
        _q2 = _q2 if np.isscalar(_q2) else _q2[0]
        _q3 = _q3 if np.isscalar(_q3) else _q3[0]
        return _mean, _std, _min, _q1, _q2, _q3, _max

    def _block_iterator(
        self, flat_signal=True, get=None, navigation_mask=None, signal_mask=None
    ):
        """A function that allows iterating lazy signal data by blocks,
        defining the dask.Array.

        Parameters
        ----------
        flat_signal: bool
            returns each block flattened, such that the shape (for the
            particular block) is (navigation_size, signal_size), with
            optionally masked elements missing. If false, returns
            the equivalent of s.inav[{blocks}].data, where masked elements are
            set to np.nan or 0.
        get : dask scheduler or None
            The dask scheduler to use for computations. If ``None``,
            ``dask.threaded.get` will be used if possible, otherwise
            ``dask.get`` will be used, for example in pyodide interpreter.
        navigation_mask : {BaseSignal, numpy array, dask array}
            The navigation locations marked as True are not returned (flat) or
            set to NaN or 0.
        signal_mask : {BaseSignal, numpy array, dask array}
            The signal locations marked as True are not returned (flat) or set
            to NaN or 0.

        """
        import dask.array as da

        if get is None:
            get = _get()
        data = self._data_aligned_with_axes
        nav_chunks = data.chunks[: self.axes_manager.navigation_dimension]
        indices = product(*[range(len(c)) for c in nav_chunks])
        signalsize = self.axes_manager.signal_size
        sig_reshape = (signalsize,) if signalsize else ()
        data = data.reshape((self.axes_manager.navigation_shape[::-1] + sig_reshape))

        if signal_mask is None:
            signal_mask = (
                slice(None)
                if flat_signal
                else np.zeros(self.axes_manager.signal_size, dtype="bool")
            )
        else:
            try:
                signal_mask = to_array(signal_mask).ravel()
            except ValueError:
                # re-raise with a message
                raise ValueError(
                    "signal_mask has to be a signal, numpy or"
                    " dask array, but "
                    "{} was given".format(type(signal_mask))
                )
            if flat_signal:
                signal_mask = ~signal_mask

        if navigation_mask is None:
            nav_mask = da.zeros(
                self.axes_manager.navigation_shape[::-1],
                chunks=nav_chunks,
                dtype="bool",
            )
        else:
            try:
                nav_mask = to_array(navigation_mask, chunks=nav_chunks)
            except ValueError:
                # re-raise with a message
                raise ValueError(
                    "navigation_mask has to be a signal, numpy or"
                    " dask array, but "
                    "{} was given".format(type(navigation_mask))
                )
        if flat_signal:
            nav_mask = ~nav_mask
        for ind in indices:
            chunk = get(data.dask, (data.name,) + ind + (0,) * bool(signalsize))
            n_mask = get(nav_mask.dask, (nav_mask.name,) + ind)
            if flat_signal:
                yield chunk[n_mask, ...][..., signal_mask]
            else:
                chunk = chunk.copy()
                value = np.nan if np.can_cast("float", chunk.dtype) else 0
                chunk[n_mask, ...] = value
                chunk[..., signal_mask] = value
                yield chunk.reshape(
                    chunk.shape[:-1] + self.axes_manager.signal_shape[::-1]
                )

    def normalize_poissonian_noise(self, navigation_mask=None, signal_mask=None):
        """Normalize the signal under the assumption of Poisson noise.

        Scales the signal to normalise Poisson noise for subsequent
        decomposition analysis as described in [Keenan2004]_.  The scaling
        is performed lazily so the full dataset is never loaded into memory.

        The Keenan-Kotula scaling computes::

            D_scaled[i, j] = D[i, j] / (sqrt(aG[i]) * sqrt(bH[j]))

        where ``aG[i]`` is the total counts for navigation position ``i``
        (summed over unmasked signal channels) and ``bH[j]`` is the total
        counts for signal channel ``j`` (summed over unmasked navigation
        positions).

        The square-root arrays ``sqrt(aG)`` and ``sqrt(bH)`` are stored as
        ``self._root_aG`` and ``self._root_bH`` so that
        :py:meth:`decomposition` can rescale the factors and loadings back to
        the original data space after decomposition.

        Parameters
        ----------
        navigation_mask : {None, boolean numpy array or BaseSignal}, default None
            Navigation positions marked as ``True`` are excluded from the
            computation of the scaling coefficients and are not scaled.
        signal_mask : {None, boolean numpy array or BaseSignal}, default None
            Signal channels marked as ``True`` are excluded from the
            computation of the scaling coefficients and are not scaled.

        Raises
        ------
        ValueError
            If all data points are masked or if negative values are found
            in the (unmasked) data.

        References
        ----------
        .. [Keenan2004] M. Keenan and P. Kotula, "Accounting for Poisson noise
            in the multivariate analysis of ToF-SIMS spectrum images", Surf.
            Interface Anal 36(3) (2004): 203-212.

        See Also
        --------
        :py:meth:`~.learn._mva.MVA.normalize_poissonian_noise` for the
        non-lazy equivalent.
        """
        import dask.array as da

        _logger.info("Scaling the data to normalize Poissonian noise")

        data = self._data_aligned_with_axes
        ndim = self.axes_manager.navigation_dimension
        sdim = self.axes_manager.signal_dimension
        nav_chunks = data.chunks[:ndim]
        sig_chunks = data.chunks[ndim:]

        # Build boolean keep-masks (True = use this position)
        if navigation_mask is None:
            nm = da.ones(
                self.axes_manager.navigation_shape[::-1],
                chunks=nav_chunks,
                dtype=bool,
            )
        else:
            nm = da.logical_not(to_array(navigation_mask, chunks=nav_chunks))

        if signal_mask is None:
            sm = da.ones(
                self.axes_manager.signal_shape[::-1],
                chunks=sig_chunks,
                dtype=bool,
            )
        else:
            sm = da.logical_not(to_array(signal_mask, chunks=sig_chunks))

        # Zero out masked entries before summing so that masked positions
        # do not contribute to aG or bH — matching the non-lazy behaviour.
        # Broadcasting: data is (nav..., sig...), nm is (nav...,), sm is (sig...,)
        nm_broadcast = nm[(...,) + (None,) * sdim]  # (nav..., 1...) for sig dims
        sm_broadcast = sm[(None,) * ndim + (...,)]  # (1..., sig...) for nav dims
        combined_mask = nm_broadcast & sm_broadcast
        masked_data = da.where(combined_mask, data, 0.0)

        # Check for negative values in the unmasked region
        min_val = masked_data.min().compute()
        if min_val < 0.0:
            raise ValueError(
                "Negative values found in data!\n"
                "Are you sure that the data follow a Poisson distribution?"
            )

        nav_axes = tuple(range(ndim, ndim + sdim))  # axes to sum over for aG
        sig_axes = tuple(range(ndim))  # axes to sum over for bH

        aG, bH = da.compute(
            masked_data.sum(axis=nav_axes),  # shape: navigation_shape[::-1]
            masked_data.sum(axis=sig_axes),  # shape: signal_shape[::-1]
        )
        aG = da.from_array(aG)
        bH = da.from_array(bH)

        # Check that not everything is masked
        if float(aG.sum().compute()) == 0.0:
            raise ValueError("All the data are masked, change the mask.")

        # Replace zeros with 1 to avoid division by zero (masked positions
        # already contribute 0 to the sum so their sqrt would be 0)
        aG = da.where(aG == 0, 1, aG)
        bH = da.where(bH == 0, 1, bH)

        self._root_aG = da.sqrt(aG)  # shape: navigation_shape[::-1]
        self._root_bH = da.sqrt(bH)  # shape: signal_shape[::-1]

        # Build the full scaling coefficient via broadcasting and divide
        coeff = (
            self._root_aG[(...,) + (None,) * sdim]
            * self._root_bH[(None,) * ndim + (...,)]
        )
        self.data = da.where(combined_mask, data / coeff, data)

    def decomposition(
        self,
        normalize_poissonian_noise=False,
        algorithm="SVD",
        output_dimension=None,
        centre=None,
        signal_mask=None,
        navigation_mask=None,
        get=None,
        num_chunks=None,
        reproject=True,
        print_info=True,
        **kwargs,
    ):
        """Perform Incremental (Batch) decomposition on the data.

        The results are stored in the
        :attr:`~.api.signals.BaseSignal.learning_results`
        attribute.

        Read more in the :ref:`User Guide <big_data.decomposition>`.

        Parameters
        ----------
        normalize_poissonian_noise : bool, default False
            If True, scale the signal to normalize Poissonian noise using
            the approach described in [KeenanKotula2004]_.
        algorithm : {'SVD', 'PCA', 'ORPCA', 'ORNMF'}, default 'SVD'
            The decomposition algorithm to use.
        output_dimension : int or None, default None
            Number of components to keep/calculate. Required for all
            algorithms including 'SVD'.
        centre : {None, 'navigation', 'signal'}, default None
            Subtract the mean along the 'navigation' or 'signal' axis
            before decomposition. Only used for the 'SVD' algorithm;
            incompatible with ``normalize_poissonian_noise=True``.
        get : dask scheduler or None
            The dask scheduler to use for computations. If ``None``,
            ``dask.threaded.get` will be used if possible, otherwise
            ``dask.get`` will be used, for example in pyodide interpreter.
        num_chunks : int or None, default None
            the number of dask chunks to pass to the decomposition model.
            More chunks require more memory, but should run faster. Will be
            increased to contain at least ``output_dimension`` signals.
        navigation_mask : :class:~.api.signals.BaseSignal, numpy.ndarray or dask.array.Array
            The navigation locations marked as True are not used in the
            decomposition.
        signal_mask : :class:~.api.signals.BaseSignal, numpy.ndarray or dask.array.Array
            The signal locations marked as True are not used in the
            decomposition.
        reproject : bool, default True
            Reproject data on the learnt components (factors) after learning.
        print_info : bool, default True
            If True, print information about the decomposition being performed.
            In the case of sklearn.decomposition objects, this includes the
            values of all arguments of the chosen sklearn algorithm.
        **kwargs
            passed to the partial_fit/fit functions.

        References
        ----------
        .. [KeenanKotula2004] M. Keenan and P. Kotula, "Accounting for Poisson noise
            in the multivariate analysis of ToF-SIMS spectrum images", Surf.
            Interface Anal 36(3) (2004): 203-212.

        See Also
        --------
        hyperspy.learn.incremental_svd.ISVD, sklearn.decomposition.IncrementalPCA,
        hyperspy.learn.orpca, hyperspy.learn.ornmf

        """
        if get is None:
            get = _get()
        # Check algorithms requiring output_dimension
        algorithms_require_dimension = ["PCA", "ORPCA", "ORNMF", "SVD"]
        if algorithm in algorithms_require_dimension and output_dimension is None:
            raise ValueError(
                "`output_dimension` must be specified for '{}'".format(algorithm)
            )

        if centre not in (None, "navigation", "signal"):
            raise ValueError(
                f"`centre` must be None, 'navigation' or 'signal', not {centre!r}"
            )

        explained_variance = None
        explained_variance_ratio = None

        _al_data = self._data_aligned_with_axes
        nav_chunks = _al_data.chunks[: self.axes_manager.navigation_dimension]

        num_chunks = 1 if num_chunks is None else num_chunks
        blocksize = np.min([utils.multiply(ar) for ar in product(*nav_chunks)])
        nblocks = utils.multiply([len(c) for c in nav_chunks])

        if output_dimension and blocksize / output_dimension < num_chunks:
            num_chunks = np.ceil(blocksize / output_dimension)

        blocksize *= num_chunks

        # Initialize print_info
        to_print = [
            "Decomposition info:",
            f"  normalize_poissonian_noise={normalize_poissonian_noise}",
            f"  algorithm={algorithm}",
            f"  output_dimension={output_dimension}",
            f"  centre={centre}",
        ]

        # LEARN
        if algorithm == "PCA":
            if not SKLEARN_INSTALLED:
                raise ImportError("algorithm='PCA' requires scikit-learn")

            import sklearn

            obj = sklearn.decomposition.IncrementalPCA(n_components=output_dimension)
            method = partial(obj.partial_fit, **kwargs)
            reproject = True
            to_print.extend(["scikit-learn estimator:", obj])

        elif algorithm == "ORPCA":
            from hyperspy.learn._rpca import ORPCA

            batch_size = kwargs.pop("batch_size", None)
            obj = ORPCA(output_dimension, **kwargs)
            method = partial(obj.fit, batch_size=batch_size)

        elif algorithm == "ORNMF":
            from hyperspy.learn._ornmf import ORNMF

            batch_size = kwargs.pop("batch_size", None)
            obj = ORNMF(output_dimension, **kwargs)
            method = partial(obj.fit, batch_size=batch_size)

        elif algorithm != "SVD":
            raise ValueError("'algorithm' not recognised")

        original_data = self.data
        try:
            _logger.info("Performing decomposition analysis")

            if normalize_poissonian_noise:
                if centre is not None:
                    raise ValueError(
                        "normalize_poissonian_noise=True is only compatible "
                        f"with centre=None, not centre={centre!r}."
                    )
                self.normalize_poissonian_noise(
                    navigation_mask=navigation_mask,
                    signal_mask=signal_mask,
                )

            # LEARN
            if algorithm == "SVD":
                from hyperspy.learn.incremental_svd import ISVD

                reproject = False
                try:
                    self._unfolded4decomposition = self.unfold()

                    obj = ISVD(n_components=output_dimension)

                    # Apply centring by temporarily modifying self.data
                    # (which is restored via original_data in the outer
                    # finally block).
                    if centre == "navigation":
                        mean = self.data.mean(axis=0, keepdims=True).compute()
                        self.data = self.data - mean
                    elif centre == "signal":
                        mean = self.data.mean(axis=1, keepdims=True).compute()
                        self.data = self.data - mean
                    else:
                        mean = None

                    for chunk in progressbar(
                        self._block_iterator(
                            flat_signal=True,
                            get=get,
                            signal_mask=signal_mask,
                            navigation_mask=navigation_mask,
                        ),
                        total=nblocks,
                        leave=True,
                        desc="Learn",
                    ):
                        obj.partial_fit(chunk)

                    factors = obj.components_.T  # (n_sig, n_components)
                    explained_variance = obj.explained_variance_
                    explained_variance_ratio = obj.explained_variance_ratio_

                    # Reproject to get loadings
                    H = []
                    for chunk in progressbar(
                        self._block_iterator(
                            flat_signal=True,
                            get=get,
                            signal_mask=signal_mask,
                            navigation_mask=navigation_mask,
                        ),
                        total=nblocks,
                        leave=True,
                        desc="Project",
                    ):
                        H.append(obj.transform(chunk))
                    loadings = np.concatenate(H, axis=0)
                finally:
                    if self._unfolded4decomposition is True:
                        self.fold()
                        self._unfolded4decomposition = False
            else:
                self._check_navigation_mask(navigation_mask)
                self._check_signal_mask(signal_mask)
                this_data = []
                try:
                    for chunk in progressbar(
                        self._block_iterator(
                            flat_signal=True,
                            get=get,
                            signal_mask=signal_mask,
                            navigation_mask=navigation_mask,
                        ),
                        total=nblocks,
                        leave=True,
                        desc="Learn",
                    ):
                        this_data.append(chunk)
                        if len(this_data) == num_chunks:
                            thedata = np.concatenate(this_data, axis=0)
                            method(thedata)
                            this_data = []
                    if len(this_data):
                        thedata = np.concatenate(this_data, axis=0)
                        method(thedata)
                except KeyboardInterrupt:  # pragma: no cover
                    pass

            # GET ALREADY CALCULATED RESULTS
            if algorithm == "PCA":
                explained_variance = obj.explained_variance_
                explained_variance_ratio = obj.explained_variance_ratio_
                factors = obj.components_.T

            elif algorithm == "ORPCA":
                factors, loadings = obj.finish()
                loadings = loadings.T

            elif algorithm == "ORNMF":
                factors, loadings = obj.finish()
                loadings = loadings.T

            # REPROJECT
            if reproject:
                if algorithm == "PCA":
                    method = obj.transform

                    def post(a):
                        return np.concatenate(a, axis=0)

                elif algorithm == "ORPCA":
                    method = obj.project

                    def post(a):
                        return np.concatenate(a, axis=1).T

                elif algorithm == "ORNMF":
                    method = obj.project

                    def post(a):
                        return np.concatenate(a, axis=1).T

                _map = map(
                    lambda thing: method(thing),
                    self._block_iterator(
                        flat_signal=True,
                        get=get,
                        signal_mask=signal_mask,
                        navigation_mask=navigation_mask,
                    ),
                )
                H = []
                try:
                    for thing in progressbar(_map, total=nblocks, desc="Project"):
                        H.append(thing)
                except KeyboardInterrupt:  # pragma: no cover
                    pass
                loadings = post(H)

            if explained_variance is not None and explained_variance_ratio is None:
                explained_variance_ratio = explained_variance / explained_variance.sum()

            # RESHUFFLE "blocked" LOADINGS
            ndim = self.axes_manager.navigation_dimension
            if algorithm != "SVD":  # Only needed for online algorithms
                try:
                    loadings = _reshuffle_mixed_blocks(
                        loadings, ndim, (output_dimension,), nav_chunks
                    ).reshape((-1, output_dimension))
                except ValueError:
                    # In case the projection step was not finished, it's left
                    # as scrambled
                    pass
        finally:
            self.data = original_data

        target = self.learning_results
        target.decomposition_algorithm = algorithm
        target.output_dimension = output_dimension
        if algorithm != "SVD":
            target._object = obj
        target.factors = factors
        target.loadings = loadings
        target.explained_variance = explained_variance
        target.explained_variance_ratio = explained_variance_ratio
        if algorithm == "SVD":
            target.mean = mean
            target.centre = centre

        # Rescale the results if the noise was normalized
        if normalize_poissonian_noise is True:
            target.factors = (
                target.factors * self._root_bH.ravel().compute()[:, np.newaxis]
            )
            target.loadings = (
                target.loadings * self._root_aG.ravel().compute()[:, np.newaxis]
            )

        # Print details about the decomposition we just performed
        if print_info:
            print("\n".join([str(pr) for pr in to_print]))

    def plot(self, navigator="auto", **kwargs):
        if self.axes_manager.ragged:
            raise RuntimeError("Plotting ragged signal is not supported.")
        if isinstance(navigator, str):
            if navigator == "spectrum":
                # We don't support the 'spectrum' option to keep it simple
                _logger.warning(
                    "The `navigator='spectrum'` option is not "
                    "supported for lazy signals, 'auto' is used "
                    "instead."
                )
                navigator = "auto"
            if navigator == "auto":
                if self.navigator is None:
                    self.compute_navigator()
                navigator = self.navigator
        super().plot(navigator=navigator, **kwargs)

    def compute_navigator(self, index=None, chunks_number=None, show_progressbar=None):
        """
        Compute the navigator by taking the sum over a single chunk contained
        the specified coordinate. Taking the sum over a single chunk is a
        computationally efficient approach to compute the navigator. The data
        can be rechunk by specifying the ``chunks_number`` argument.

        Parameters
        ----------
        index : (int, float, None) or iterable, optional
            Specified where to take the sum, follows HyperSpy indexing syntax
            for integer and float. If None, the index is the centre of the
            signal_space
        chunks_number : (int, None) or iterable, optional
            Define the number of chunks in the signal space used for rechunk
            the when calculating of the navigator. Useful to define the range
            over which the sum is calculated.
            If None, the existing chunking will be considered when picking the
            chunk used in the navigator calculation.
        %s

        Returns
        -------
        None.

        Notes
        -----
        The number of chunks will affect where the sum is taken. If the sum
        needs to be taken in the centre of the signal space (for example, in
        the case of diffraction pattern), the number of chunk needs to be an
        odd number, so that the middle is centered.

        """
        import dask.array as da

        signal_shape = self.axes_manager.signal_shape

        if index is None:
            index = [round(shape / 2) for shape in signal_shape]
        else:
            if not utils.isiterable(index):
                index = [index] * len(signal_shape)
            index = [
                axis._get_index(_idx)
                for _idx, axis in zip(index, self.axes_manager.signal_axes)
            ]
        _logger.info(f"Using index: {index}")

        if chunks_number is None:
            chunks = self.data.chunks
        else:
            if not utils.isiterable(chunks_number):
                chunks_number = [chunks_number] * len(signal_shape)
            # Determine the chunk size
            signal_chunks = da.core.normalize_chunks(
                [int(size / cn) for cn, size in zip(chunks_number, signal_shape)],
                shape=signal_shape,
            )
            # Needs to reverse the chunks list to match dask chunking order
            signal_chunks = list(signal_chunks)[::-1]
            navigation_chunks = ["auto"] * len(self.axes_manager.navigation_shape)
            chunks = self.data.rechunk(
                [*navigation_chunks, *signal_chunks],
                balance=True,
            ).chunks

        # Get the slice of the corresponding chunk
        signal_size = len(signal_shape)
        signal_chunks = tuple(chunks[i - signal_size] for i in range(signal_size))
        _logger.info(f"Signal chunks: {signal_chunks}")
        isig_slice = dask_utils.get_signal_chunk_slice(index, chunks)

        _logger.info(f"Computing sum over signal dimension: {isig_slice}")
        axes = [axis.index_in_array for axis in self.axes_manager.signal_axes]
        navigator = self.isig[isig_slice].sum(axes)
        navigator.compute(show_progressbar=show_progressbar)
        navigator.original_metadata.set_item("sum_from", str(isig_slice))

        self.navigator = navigator.T

    compute_navigator.__doc__ %= SHOW_PROGRESSBAR_ARG


def _reshuffle_mixed_blocks(array, ndim, sshape, nav_chunks):
    """Reshuffles dask block-shuffled array

    Parameters
    ----------
    array : np.ndarray
        the array to reshuffle
    ndim : int
        the number of navigation (shuffled) dimensions
    sshape : tuple of ints
        The shape
    """
    splits = np.cumsum(
        [utils.multiply(ar) for ar in product(*nav_chunks)][:-1]
    ).tolist()
    if splits:
        all_chunks = [
            ar.reshape(shape + sshape)
            for shape, ar in zip(product(*nav_chunks), np.split(array, splits))
        ]

        def split_stack_list(what, step, axis):
            total = len(what)
            if total != step:
                return [
                    np.concatenate(what[i : i + step], axis=axis)
                    for i in range(0, total, step)
                ]
            else:
                return np.concatenate(what, axis=axis)

        for chunks, axis in zip(nav_chunks[::-1], range(ndim - 1, -1, -1)):
            step = len(chunks)
            all_chunks = split_stack_list(all_chunks, step, axis)
        return all_chunks
    else:
        return array
