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
    DECOMP_MASK_DOC,
    DECOMP_NORMALIZE_POISSONIAN_NOISE_DOC,
    DECOMP_PRINT_INFO_DOC,
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
        if signalsize:
            # Ensure the signal dimension is a single chunk so that the
            # index appended in the loop below retrieves the full signal
            # vector rather than only the first chunk.  This matters when
            # the on-disk chunk size is smaller than the signal size
            # (e.g. after unfold()).
            data = data.rechunk({-1: -1})

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
        :meth:`~hyperspy.api.signals.BaseSignal.normalize_poissonian_noise` :
            Non-lazy equivalent.
        """
        import dask.array as da

        _logger.info("Scaling the data to normalize Poissonian noise")

        # Convert masks from HyperSpy navigation_shape / signal_shape order to
        # underlying array axis order.  BaseSignal masks expose .data already
        # in array order; numpy/dask masks are provided in navigation_shape
        # order (HyperSpy convention) and must be transposed.
        if isinstance(navigation_mask, signals.BaseSignal):
            navigation_mask = navigation_mask.data
        elif navigation_mask is not None and hasattr(navigation_mask, "T"):
            navigation_mask = navigation_mask.T

        if isinstance(signal_mask, signals.BaseSignal):
            signal_mask = signal_mask.data

        data = self._data_aligned_with_axes
        ndim = self.axes_manager.navigation_dimension
        sdim = self.axes_manager.signal_dimension

        # Ensure masks are dask arrays with matching chunks so the shared
        # function can infer the backend correctly.
        nav_chunks = data.chunks[:ndim]
        sig_chunks = data.chunks[ndim:]

        if navigation_mask is not None and not isinstance(navigation_mask, da.Array):
            navigation_mask = da.from_array(navigation_mask, chunks=nav_chunks)
        if signal_mask is not None and not isinstance(signal_mask, da.Array):
            signal_mask = da.from_array(signal_mask, chunks=sig_chunks)

        from hyperspy.learn._mva import _keenan_kotula_scale

        self.data, self._root_aG, self._root_bH = _keenan_kotula_scale(
            data, navigation_mask, signal_mask, ndim, sdim
        )

    def _decomposition_svd_matrix(
        self, svd_solver, centre, output_dimension, navigation_mask, signal_mask
    ):
        """Run SVD (full or randomized) on the unfolded 2-D data matrix.

        Unfolds the signal, resolves navigation and signal masks to 1-D
        boolean arrays, applies centring if requested, computes the SVD
        via either ``da.linalg.svd`` (full) or ``da.linalg.svd_compressed``
        (randomized), and folds the signal back.

        Parameters
        ----------
        svd_solver : str
            ``"full"`` or ``"randomized"``.
        centre : str or None
            Centring strategy (``"navigation"``, ``"signal"``, or ``None``).
        output_dimension : int or None
            Number of components; required for ``"randomized"``.
        navigation_mask : various or None
            Navigation mask; may be mutated to array order.
        signal_mask : various or None
            Signal mask; may be mutated to array order.

        Returns
        -------
        dict
            Keys: ``loadings``, ``factors``, ``explained_variance``, ``mean``,
            ``nav_mask_1d``, ``sig_mask_1d``, ``navigation_mask``,
            ``signal_mask``, ``_navigation_mask_for_reproject``,
            ``_D_unfolded``.
        """
        import dask
        import dask.array as da

        # Initialise all variables that are set inside the try block so
        # the finally / return blocks never see an UnboundLocalError.
        loadings = None
        factors = None
        explained_variance = None
        mean = None
        nav_mask_1d = None
        sig_mask_1d = None
        _navigation_mask_for_reproject = navigation_mask
        _D_unfolded = None

        try:
            self._unfolded4decomposition = self.unfold()

            # Resolve navigation mask to a 1-D boolean numpy array.
            nav_mask_1d = None
            if navigation_mask is not None:
                if isinstance(navigation_mask, signals.BaseSignal):
                    _nm = navigation_mask.data
                elif hasattr(navigation_mask, "T"):
                    _nm = navigation_mask.T
                else:
                    _nm = navigation_mask
                _navigation_mask_for_reproject = _nm
                if isinstance(_nm, da.Array):
                    nav_mask_1d = _nm.ravel().compute().astype(bool)
                else:
                    nav_mask_1d = np.asarray(_nm).ravel().astype(bool)
                _navigation_mask_for_reproject = nav_mask_1d
                # Update navigation_mask to array order so downstream
                # _to_flat_bool calls get the correct ravel order for 2-D
                # navigation spaces.
                navigation_mask = _nm

            # Resolve signal mask to a 1-D boolean numpy array.
            sig_mask_1d = None
            if signal_mask is not None:
                if isinstance(signal_mask, signals.BaseSignal):
                    _sm = signal_mask.data
                else:
                    _sm = signal_mask
                if isinstance(_sm, da.Array):
                    sig_mask_1d = _sm.ravel().compute().astype(bool)
                else:
                    sig_mask_1d = np.asarray(_sm).ravel().astype(bool)
                signal_mask = _sm

            # Build the data matrix, applying masks if present.
            # After unfold() self.data is 2-D: (nav, sig).
            D = self.data  # dask array (nav, sig)
            # Keep an unmasked reference for reproject use (after fold
            # self.data is N-D again, so we capture it here).
            _D_unfolded = self.data
            if nav_mask_1d is not None:
                D = D[~nav_mask_1d, :]
            if sig_mask_1d is not None:
                D = D[:, ~sig_mask_1d]

            if svd_solver == "full":
                import warnings

                from hyperspy.exceptions import VisibleDeprecationWarning

                warnings.warn(
                    "svd_solver='full' is deprecated and will be removed in "
                    "HyperSpy 3.0.  Use svd_solver='randomized' instead, "
                    "which gives identical results for truncated SVD with "
                    "substantially lower memory usage.",
                    VisibleDeprecationWarning,
                    stacklevel=2,
                )
                if centre == "navigation":
                    mean = D.mean(axis=0, keepdims=True).compute()
                    D = D - mean
                elif centre == "signal":
                    mean = D.mean(axis=1, keepdims=True).compute()
                    D = D - mean
                else:
                    mean = None

                if D.numblocks[1] > 1:
                    D = D.rechunk({1: -1})
                U, S, V = da.linalg.svd(D)
                if output_dimension is not None:
                    U = U[:, :output_dimension]
                    S = S[:output_dimension]
                    V = V[:output_dimension]
                factors = V.T
                explained_variance = S**2 / D.shape[0]
                loadings = U * S
            else:  # randomized
                if centre == "navigation":
                    mean = D.mean(axis=0, keepdims=True).compute()
                    D = D - mean
                elif centre == "signal":
                    mean = D.mean(axis=1, keepdims=True).compute()
                    D = D - mean
                else:
                    mean = None

                # Use the synchronous scheduler for svd_compressed to
                # avoid materialising the full dataset in memory.  The
                # default threaded scheduler fires all rows of D @ Omega
                # simultaneously, producing a peak-memory spike equal to
                # the full data size.  Synchronous processes one row at a
                # time and is actually faster here because there is no
                # I/O to overlap — every chunk is a CPU-bound matmul.
                with dask.config.set(scheduler="synchronous"):
                    U, S, V = da.linalg.svd_compressed(D, k=output_dimension)
                    U, S, V = dask.compute(U, S, V)

                factors = V.T  # (n_unmasked_sig, output_dimension)
                explained_variance = S**2 / D.shape[0]
                loadings = U * S
        finally:
            if self._unfolded4decomposition is True:
                self.fold()
                # BUGFIX: was ``is False`` (identity comparison, always
                # no-op); must be ``= False`` assignment to clear the flag
                # so the signal does not remain permanently unfolded.
                self._unfolded4decomposition = False

        return {
            "loadings": loadings,
            "factors": factors,
            "explained_variance": explained_variance,
            "mean": mean,
            "nav_mask_1d": nav_mask_1d,
            "sig_mask_1d": sig_mask_1d,
            "navigation_mask": navigation_mask,
            "signal_mask": signal_mask,
            "_navigation_mask_for_reproject": _navigation_mask_for_reproject,
            "_D_unfolded": _D_unfolded,
        }

    def _decomposition_reproject_signal(
        self,
        algorithm,
        svd_solver,
        reproject,
        loadings,
        factors,
        mean,
        centre,
        _D_unfolded,
        nav_mask_1d,
        _navigation_mask_for_reproject,
        _flat_sig_mask,
        _flat_nav_mask,
        get,
        nblocks,
    ):
        """Reproject factors over the full (unmasked) signal space.

        Recomputes *factors* by projecting the full signal data through
        the pseudo-inverse of the (unmasked-navigation) loadings.
        Supports both dask-based (``svd_solver='full'``) and eager
        (chunk-loop) paths.

        Parameters
        ----------
        algorithm : str
            Decomposition algorithm name.
        svd_solver : str
            SVD backend (``"full"``, ``"randomized"``, ``"incremental"``).
        reproject : str
            Must be ``"signal"`` or ``"both"``.
        loadings : ndarray or dask Array
            Current loadings (unmasked navigation rows).
        factors : ndarray or dask Array
            Current factors (may be overwritten).
        mean : ndarray or None
            Per-channel or per-pixel mean from centring.
        centre : str or None
            Centring strategy.
        _D_unfolded : dask Array
            Unfolded (nav, sig) data matrix, captured before fold().
        nav_mask_1d : ndarray or None
            Flat boolean navigation mask (True = excluded).
        _navigation_mask_for_reproject : various or None
            Navigation mask in a form suitable for _block_iterator.
        _flat_sig_mask : ndarray or None
            Flat boolean signal mask (True = excluded).
        _flat_nav_mask : ndarray or None
            Flat boolean navigation mask for result indexing.
        get : dask scheduler
        nblocks : int
            Number of navigation blocks.

        Returns
        -------
        factors : ndarray
            Re-projected factors covering the full signal space.
        """
        if algorithm == "SVD" and svd_solver == "full":
            import dask.array as da

            D_sig = _D_unfolded  # (nav, sig)
            if nav_mask_1d is not None:
                D_sig = D_sig[~nav_mask_1d, :]
            if mean is not None and centre == "navigation":
                D_sig = D_sig - mean
            if reproject == "both":
                # loadings covers all nav after nav-reproject; restrict
                # to unmasked rows before computing pinv.
                if _flat_nav_mask is not None:
                    L = loadings[~_flat_nav_mask, :]
                else:
                    L = loadings
            else:
                L = loadings  # already unmasked-nav only
            # pinv(L) is (k, n_unmasked_nav) — small; compute eagerly.
            pinv_L = np.linalg.pinv(L.compute() if isinstance(L, da.Array) else L)
            # (k, n_unmasked_nav) @ (n_unmasked_nav, sig) = (k, sig)
            # Dask streams over nav chunks; result is small.
            factors = (da.from_array(pinv_L) @ D_sig).T.compute()
        else:
            from hyperspy.external.progressbar import progressbar
            from hyperspy.learn._mva import _reproject_signal_factors

            # Collect all navigation-unmasked rows with the full signal
            # (no signal mask), then solve: factors = pinv(L) @ D_full
            D_chunks = []
            for chunk in progressbar(
                self._block_iterator(
                    flat_signal=True,
                    get=get,
                    signal_mask=None,
                    navigation_mask=_navigation_mask_for_reproject,
                ),
                total=nblocks,
                leave=True,
                desc="Reproject signal",
            ):
                D_chunks.append(chunk)
            D = np.concatenate(D_chunks, axis=0)  # (n_unmasked_nav, sig_size)
            if mean is not None:
                # mean may be 2-D (keepdims=True from centre='navigation');
                # ravel to 1-D so length and boolean-index assignment work.
                mean_1d = np.asarray(mean).ravel()
                # mean_1d was computed over unmasked signal channels only;
                # expand to full signal size (zeros at masked positions)
                # so it can be broadcast against D which covers all channels.
                if _flat_sig_mask is not None and len(mean_1d) < D.shape[1]:
                    mean_full = np.zeros(D.shape[1], dtype=mean_1d.dtype)
                    mean_full[~_flat_sig_mask] = mean_1d
                    D = D - mean_full
                else:
                    D = D - mean_1d
            if reproject == "both":
                if _flat_nav_mask is not None:
                    L = loadings[~_flat_nav_mask, :]
                else:
                    L = loadings
            else:
                L = loadings  # already unmasked-nav only
            factors = _reproject_signal_factors(D, L)
        return factors

    def _project_loadings(self, obj, desc, navigation_mask, signal_mask, get, nblocks):
        """Project data chunks through *obj.transform* and concatenate.

        Iterates over navigation blocks via :meth:`_block_iterator`, calls
        ``obj.transform()`` on each chunk, and returns the concatenated
        loadings array.  Used by ISVD, PCA, NMF, ORPCA, ORNMF, and custom
        estimators after fitting.

        Parameters
        ----------
        obj : estimator
            Fitted estimator with a ``transform`` method.
        desc : str
            Progress-bar label.
        navigation_mask : various or None
            Navigation mask for :meth:`_block_iterator`.
        signal_mask : various or None
            Signal mask for :meth:`_block_iterator`.
        get : dask scheduler
        nblocks : int
            Number of navigation blocks.

        Returns
        -------
        ndarray
            Concatenated loadings, shape ``(n_nav, n_components)``.
        """
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
            desc=desc,
        ):
            H.append(obj.transform(chunk))
        return np.concatenate(H, axis=0)

    def _decomposition_reproject_navigation(
        self,
        reproject,
        algorithm,
        svd_solver,
        loadings,
        factors,
        mean,
        centre,
        _D_unfolded,
        sig_mask_1d,
        obj,
        signal_mask,
        get,
        nblocks,
        _navigation_mask_for_reproject,
    ):
        """Reproject loadings over the full (unmasked) navigation space.

        Recomputes *loadings* by projecting the full navigation data through
        the learned factors.  Supports three paths: dask matmul for
        ``svd_solver='full'``, dask matmul for ``svd_solver='randomized'``,
        and a ``obj.transform`` chunk-loop for other algorithms.  Also
        handles the ``reproject=None`` default (project for non-SVD).

        Parameters
        ----------
        reproject : str or None
            ``"navigation"``, ``"both"``, or ``None``.
        algorithm : str
            Decomposition algorithm name.
        svd_solver : str
            SVD backend.
        loadings : ndarray or dask Array
            Current loadings (may be overwritten with full-nav loadings).
        factors : ndarray or dask Array
            Learned factors (signal × components).
        mean : ndarray or None
            Per-channel mean from centring.
        centre : str or None
            Centring strategy.
        _D_unfolded : dask Array or None
            Unfolded (nav, sig) data matrix, captured before fold().
        sig_mask_1d : ndarray or None
            Flat boolean signal mask (True = excluded).
        obj : estimator or None
            Fitted estimator with a ``transform`` method.
        signal_mask : various or None
            Signal mask for ``_project_loadings``.
        get : dask scheduler
        nblocks : int
            Number of navigation blocks.
        _navigation_mask_for_reproject : various or None
            Navigation mask for the default projection pass.

        Returns
        -------
        loadings : ndarray or dask Array
            Loadings covering the full navigation space.
        _nav_reprojected : bool
            ``True`` if navigation reprojection was performed.
        """
        _nav_reprojected = False
        if reproject in ("navigation", "both"):
            if algorithm == "SVD" and svd_solver == "full":
                import dask.array as da

                from hyperspy.learn._mva import _reproject_navigation_loadings

                D_nav = _D_unfolded  # (nav, sig)
                if sig_mask_1d is not None:
                    D_nav = D_nav[:, ~sig_mask_1d]
                if mean is not None and centre == "navigation":
                    D_nav = D_nav - mean
                _factors_da = (
                    factors if isinstance(factors, da.Array) else da.from_array(factors)
                )
                # Dask matmul — streams over nav chunks without
                # materialising the full matrix.  .compute() only
                # materialises the small (nav × k) loadings array.
                loadings = _reproject_navigation_loadings(D_nav, _factors_da).compute()
            elif algorithm == "SVD" and svd_solver == "randomized":
                import dask.array as da

                from hyperspy.learn._mva import _reproject_navigation_loadings

                D_nav = _D_unfolded  # (nav, sig)
                if sig_mask_1d is not None:
                    D_nav = D_nav[:, ~sig_mask_1d]
                if mean is not None and centre == "navigation":
                    D_nav = D_nav - mean
                _factors_da = (
                    factors if isinstance(factors, da.Array) else da.from_array(factors)
                )
                # Same least-squares formula as full SVD; only the factor
                # matrix was computed by a different solver.
                loadings = _reproject_navigation_loadings(D_nav, _factors_da).compute()
            else:
                # Non-SVD algorithms (PCA, NMF, ORPCA, ORNMF, custom):
                # use the estimator's transform() on each chunk.
                try:
                    loadings = self._project_loadings(
                        obj, "Reproject", None, signal_mask, get, nblocks
                    )
                except KeyboardInterrupt:  # pragma: no cover
                    pass
            _nav_reprojected = True
        elif reproject is None:
            # Default behaviour: for non-SVD algorithms, project to get
            # loadings (preserves the pre-existing default of
            # reproject=True).  SVD already computed loadings during the
            # learn pass, so nothing extra is needed.
            if algorithm != "SVD":
                try:
                    loadings = self._project_loadings(
                        obj,
                        "Project",
                        _navigation_mask_for_reproject,
                        signal_mask,
                        get,
                        nblocks,
                    )
                except KeyboardInterrupt:  # pragma: no cover
                    pass
        return loadings, _nav_reprojected

    def _store_decomposition_results(
        self,
        explained_variance,
        explained_variance_ratio,
        factors,
        loadings,
        mean,
        centre,
        navigation_mask,
        signal_mask,
        algorithm,
        svd_solver,
        obj,
        output_dimension,
        normalize_poissonian_noise,
        _flat_nav_mask,
        _flat_sig_mask,
        _nav_reprojected,
        _signal_reprojected,
        print_info,
        to_print,
        return_info,
    ):
        """Store decomposition results in ``learning_results`` and finalise.

        Computes explained variance ratio and elbow, normalises masks,
        stores all attributes in ``self.learning_results``, applies
        Poisson-noise rescaling, NaN-fills excluded positions, prints
        decomposition info, and returns the ``return_info`` object.

        Parameters
        ----------
        explained_variance : ndarray or None
        explained_variance_ratio : ndarray or None
        factors : ndarray or dask Array
        loadings : ndarray or dask Array
        mean : ndarray or None
        centre : str or None
        navigation_mask : various or None
        signal_mask : various or None
        algorithm : str or object
        svd_solver : str
        obj : estimator or None
        output_dimension : int or None
        normalize_poissonian_noise : bool
        _flat_nav_mask : ndarray or None
        _flat_sig_mask : ndarray or None
        _nav_reprojected : bool
        _signal_reprojected : bool
        print_info : bool
        to_print : list of str
        return_info : bool

        Returns
        -------
        object or None
            The fitted estimator if ``return_info`` is ``True`` and the
            algorithm supports it, else ``None``.
        """
        from hyperspy.learn._mva import _nan_expand_rows, _to_flat_bool

        target = self.learning_results

        # explained variance ratio and elbow estimate
        if explained_variance is not None and explained_variance_ratio is None:
            (
                explained_variance_ratio,
                number_significant_components,
            ) = self._compute_explained_variance_ratio(explained_variance)
        elif explained_variance_ratio is not None:
            number_significant_components = int(
                self.estimate_elbow_position(explained_variance_ratio) + 1
            )
        else:
            number_significant_components = None

        # normalise masks to flat bool arrays
        nav_size = self.axes_manager.navigation_size
        sig_size = self.axes_manager.signal_size
        flat_nav_mask = _to_flat_bool(navigation_mask)
        flat_sig_mask = _to_flat_bool(signal_mask)

        # store core results
        target.decomposition_algorithm = algorithm
        _stored_output_dim = (
            output_dimension
            if output_dimension is not None
            else (factors.shape[1] if factors is not None else None)
        )
        target.output_dimension = _stored_output_dim
        target.poissonian_noise_normalized = normalize_poissonian_noise
        target.explained_variance = explained_variance
        target.explained_variance_ratio = explained_variance_ratio
        target.number_significant_components = number_significant_components
        target.centre = centre
        target.mean = mean
        target.unmixing_matrix = None
        target.bss_algorithm = None
        if algorithm != "SVD":
            target._object = obj

        # rescale if Poisson noise was normalised
        if normalize_poissonian_noise:
            root_bH_flat = self._root_bH.ravel().compute()
            if _flat_sig_mask is not None and not _signal_reprojected:
                root_bH_flat = root_bH_flat[~_flat_sig_mask]
            factors = factors * root_bH_flat[:, np.newaxis]
            root_aG_flat = self._root_aG.ravel().compute()
            if _flat_nav_mask is not None and not _nav_reprojected:
                root_aG_flat = root_aG_flat[~_flat_nav_mask]
            loadings = loadings * root_aG_flat[:, np.newaxis]

        # store masks and NaN-fill excluded positions
        if flat_sig_mask is not None:
            target.signal_mask = flat_sig_mask.reshape(
                self.axes_manager._signal_shape_in_array
            )
            if not _signal_reprojected:
                factors = _nan_expand_rows(factors, flat_sig_mask, sig_size)

        if flat_nav_mask is not None:
            target.navigation_mask = flat_nav_mask.reshape(
                self.axes_manager._navigation_shape_in_array
            )
            if not _nav_reprojected:
                loadings = _nan_expand_rows(loadings, flat_nav_mask, nav_size)

        target.factors = factors
        target.loadings = loadings

        if print_info:
            print("\n".join([str(pr) for pr in to_print]))

        if return_info:
            return obj if algorithm != "SVD" or svd_solver == "incremental" else None

    def decomposition(
        self,
        normalize_poissonian_noise=False,
        algorithm="SVD",
        output_dimension=None,
        centre=None,
        auto_transpose=True,
        signal_mask=None,
        navigation_mask=None,
        get=None,
        num_chunks=None,
        reproject=None,
        return_info=False,
        print_info=True,
        svd_solver="randomized",
        **kwargs,
    ):
        """Apply decomposition to a lazy dataset.

        The results are stored in the
        :attr:`~.api.signals.BaseSignal.learning_results`
        attribute.

        Read more in the :ref:`User Guide <big_data.decomposition>`.

        Parameters
        ----------
        %s
        algorithm : {'SVD', 'PCA', 'ORPCA', 'ORNMF', 'NMF'} or object, default 'SVD'
            The decomposition algorithm to use. In addition to the named
            algorithms, any object that implements ``partial_fit`` (and
            ``transform`` or ``fit_transform``) can be passed directly and
            will be used as an out-of-core estimator; objects that only
            implement ``fit`` / ``fit_transform`` (without ``partial_fit``)
            are also accepted but all data will be collected into memory
            before calling ``fit_transform``.  After fitting, the estimator
            must expose a ``components_`` attribute (rows = components) to
            supply the factors.

            For ``'SVD'``, the specific backend is chosen via ``svd_solver``
            (see below).
        output_dimension : int or None, default None
            Number of components to keep/calculate. Required for all
            algorithms and for ``svd_solver='randomized'`` and
            ``svd_solver='incremental'``.  Optional for
            ``svd_solver='full'``, in which case all components up to
            ``min(nav_size, sig_size)`` are returned as a lazy dask array
            without triggering any computation.
        centre : {None, 'navigation', 'signal'}, default None
            Subtract the mean along the 'navigation' or 'signal' axis
            before decomposition. Only used for the ``'SVD'`` and ``'PCA'``
            algorithms; incompatible with ``normalize_poissonian_noise=True``.
        auto_transpose : bool, default True
            Deprecated and has no effect. Kept for API compatibility.
        get : dask scheduler or None
            The dask scheduler to use for computations. If ``None``,
            ``dask.threaded.get`` will be used if possible, otherwise
            ``dask.get`` will be used, for example in pyodide interpreter.
        num_chunks : int or None, default None
            the number of dask chunks to pass to the decomposition model.
            More chunks require more memory, but should run faster. Will be
            increased to contain at least ``output_dimension`` signals.
            Not used for ``'SVD'`` with ``svd_solver='randomized'``.
        %s
        %s
        reproject : {None, "navigation", "signal", "both"}, default None
            If not None, the decomposition results will be projected onto the
            full (unmasked) data after learning:

            * ``None``: use the default for the chosen algorithm.
              For ``"PCA"``, ``"NMF"``, ``"ORPCA"`` and ``"ORNMF"`` this is equivalent
              to ``"navigation"``; for ``"SVD"`` loadings are computed during
              the learn pass (reprojection is a no-op).
            * ``"navigation"``: reproject onto navigation space to get full
              loadings (useful when a navigation mask was applied).
            * ``"signal"``: reproject onto signal space to get full factors
              (useful when a signal mask was applied).
            * ``"both"``: perform both reprojections.
        return_info : bool, default False
            The result of the decomposition is stored internally. However,
            some algorithms generate extra information that is not stored. If
            True, return any extra information if available. For sklearn-based
            algorithms (``"PCA"``, ``"NMF"``, ``"ORPCA"``, ``"ORNMF"``, a custom
            estimator object, or ``svd_solver="incremental"``), this is the
            fitted estimator object.  For ``svd_solver`` values that do not
            use an sklearn-like estimator (``"full"``, ``"randomized"``),
            ``None`` is returned.
        %s
        svd_solver : {'randomized', 'incremental', 'full'}, default 'randomized'
            Selects the SVD backend when ``algorithm='SVD'``.  Ignored for
            all other algorithms.

            * ``'randomized'`` (default): randomised truncated SVD via
              ``dask.array.linalg.svd_compressed``.  Builds a single dask
              task graph and materialises only the top-*k* singular vectors.
              **Fastest of the three solvers** by a substantial margin
              (the computation is CPU-bound and benefits from being executed
              sequentially within a single graph).  ``output_dimension``
              is required.  Supports ``centre``, navigation/signal masks,
              and ``reproject``.  Works with arrays chunked in one or both
              dimensions.

              *Advantages*: fastest solver; graph-based scheduling avoids
              per-chunk overhead; supports masking and centring.

              *Disadvantages*: randomised algorithm — results differ
              slightly between runs and from exact SVD; ``output_dimension``
              must be set.

            * ``'incremental'``: exact incremental SVD via
              ``ISVD`` (a subclass of
              ``sklearn.decomposition.IncrementalPCA`` with centering
              disabled).  Streams the data one mini-batch at a time, with
              steady-state memory proportional to the chunk size rather
              than the full dataset.  ``output_dimension`` is required.

              *Advantages*: lowest steady-state memory — scales to datasets
              larger than RAM; deterministic result; supports ``centre``,
              masks, and all ``reproject`` modes.

              *Disadvantages*: significantly slower than ``'randomized'``
              (each chunk is processed serially rather than through a single
              task graph); requires scikit-learn.

            * ``'full'``: exact full SVD via ``dask.array.linalg.svd``
              (TSQR algorithm).

              .. deprecated:: 2.5
                 ``svd_solver='full'`` is deprecated and will be removed
                 in HyperSpy 3.0.  Use ``svd_solver='randomized'`` instead,
                 which gives identical results for truncated SVD with
                 substantially lower memory usage.
        **kwargs
            passed to the partial_fit/fit functions.

        References
        ----------
        .. [Keenan2004] M. Keenan and P. Kotula, "Accounting for Poisson noise
            in the multivariate analysis of ToF-SIMS spectrum images", Surf.
            Interface Anal 36(3) (2004): 203-212.

        Notes
        -----
        **Array types stored in** ``learning_results``

        After decomposition, ``learning_results.factors`` and
        ``learning_results.loadings`` hold either **numpy** or **dask** arrays
        depending on the algorithm and solver:

        .. list-table::
           :header-rows: 1
           :widths: 30 35 35

           * - Algorithm / solver
             - ``factors``
             - ``loadings``
           * - ``'SVD'``, ``svd_solver='randomized'``
             - numpy (computed)
             - numpy (computed)
           * - ``'SVD'``, ``svd_solver='incremental'``
             - numpy (computed)
             - numpy (computed)
           * - ``'SVD'``, ``svd_solver='full'`` (no ``reproject``)
             - **dask** (lazy)
             - **dask** (lazy)
           * - ``'SVD'``, ``svd_solver='full'``, ``reproject='navigation'``
             - **dask** (lazy)
             - numpy (computed)
           * - ``'SVD'``, ``svd_solver='full'``, ``reproject='signal'``
             - numpy (computed)
             - **dask** (lazy)
           * - ``'SVD'``, ``svd_solver='full'``, ``reproject='both'``
             - numpy (computed)
             - numpy (computed)

        **Fully lazy pipeline with** ``svd_solver='full'``

        ``svd_solver='full'`` keeps the entire pipeline lazy — including when
        ``reproject`` is used.  Reproject steps are performed with dask
        matmuls that stream over chunks; only the *produced* array (small:
        ``nav × k`` or ``sig × k``) is computed eagerly.  The unrequested
        array remains lazy::

             s.decomposition(algorithm="SVD", svd_solver="full", output_dimension=3)
             # learning_results.factors and .loadings are dask arrays

             model = s.get_decomposition_model()
             # model is a LazySignal; model.data is a dask array

             model.save("model.hspy")
             # triggers computation chunk by chunk while writing to disk

             # With reproject: factors stay lazy (only loadings are computed)
             s.decomposition(algorithm="SVD", svd_solver="full",
                             output_dimension=3, reproject="navigation")
             model = s.get_decomposition_model()  # still lazy
             model.save("model_reprojected.hspy")

        See Also
        --------
        hyperspy.learn.incremental_svd.ISVD :
            Incremental SVD backend.
        sklearn.decomposition.IncrementalPCA :
            Used by the ``'PCA'`` algorithm.
        sklearn.decomposition.MiniBatchNMF :
            Used by the ``'NMF'`` algorithm.
        hyperspy.learn.orpca :
            Online robust PCA.
        hyperspy.learn.ornmf :
            Online robust NMF.

        """
        from hyperspy.learn._mva import _to_flat_bool

        if get is None:
            get = _get()
        # Check algorithms requiring output_dimension.
        algorithms_require_dimension = ["PCA", "ORPCA", "ORNMF", "NMF"]
        if algorithm in algorithms_require_dimension and output_dimension is None:
            raise ValueError(
                "`output_dimension` must be specified for '{}'".format(algorithm)
            )
        # Detect custom sklearn-like estimator objects
        _is_custom_sklearn_like = not isinstance(algorithm, str) and (
            hasattr(algorithm, "fit_transform")
            or (hasattr(algorithm, "fit") and hasattr(algorithm, "transform"))
        )
        if (
            not _is_custom_sklearn_like
            and isinstance(algorithm, str)
            and algorithm not in ("SVD", "PCA", "ORPCA", "ORNMF", "NMF")
        ):
            _lazy_unsupported = {
                "MLPCA",
                "RPCA",
                "sklearn_pca",
                "sparse_pca",
                "mini_batch_sparse_pca",
            }
            if algorithm in _lazy_unsupported:
                raise NotImplementedError(
                    f"algorithm={algorithm!r} is not supported for lazy signals. "
                    "Supported algorithms are: 'SVD', 'PCA', 'ORPCA', 'ORNMF', 'NMF', "
                    "or a custom object with fit_transform() or fit()+transform()."
                )
            raise ValueError(
                f"'algorithm' {algorithm!r} not recognised. "
                "Expected one of: 'SVD', 'PCA', 'ORPCA', 'ORNMF', 'NMF', "
                "or a custom object with fit_transform() or fit()+transform()."
            )

        if kwargs.get("var_array") is not None or kwargs.get("var_func") is not None:
            raise NotImplementedError(
                "`var_array` and `var_func` are only used by the 'MLPCA' algorithm, "
                "which is not supported for lazy signals."
            )

        if algorithm == "SVD" and svd_solver not in (
            "randomized",
            "incremental",
            "full",
        ):
            raise ValueError(
                f"svd_solver={svd_solver!r} not recognised. "
                "Expected one of: 'randomized', 'incremental', 'full'."
            )

        # ── input validation (mirrors non-lazy MVA.decomposition) ────────────
        # Only pass svd_solver for the SVD path so solver-specific
        # output_dimension constraints do not leak into PCA or custom
        # sklearn-like estimators.
        self._validate_decomposition_inputs(
            output_dimension,
            centre,
            reproject,
            svd_solver=svd_solver if algorithm == "SVD" else None,
        )

        self._check_navigation_mask(navigation_mask)
        self._check_signal_mask(signal_mask)
        # ─────────────────────────────────────────────────────────────────────

        explained_variance = None
        explained_variance_ratio = None
        mean = None
        loadings = None
        factors = None
        _D_unfolded = None
        nav_mask_1d = None
        sig_mask_1d = None
        obj = None  # set below for non-SVD algorithms and svd_solver='incremental'

        _al_data = self._data_aligned_with_axes
        nav_chunks = _al_data.chunks[: self.axes_manager.navigation_dimension]

        if num_chunks is not None and (
            not isinstance(num_chunks, (int, np.integer))
            or isinstance(num_chunks, bool)
            or num_chunks <= 0
        ):
            raise ValueError(
                f"`num_chunks` must be a positive integer, got {num_chunks!r}."
            )
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
            to_print.extend(["scikit-learn estimator:", obj])

        elif algorithm == "NMF":
            if not SKLEARN_INSTALLED:
                raise ImportError("algorithm='NMF' requires scikit-learn")

            import sklearn.decomposition as _skd

            # MiniBatchNMF (sklearn >= 1.1) supports incremental partial_fit.
            # Fall back to NMF (loads all data) if MiniBatchNMF is not available.
            if hasattr(_skd, "MiniBatchNMF"):
                obj = _skd.MiniBatchNMF(n_components=output_dimension, **kwargs)
                method = partial(obj.partial_fit)
            else:  # pragma: no cover
                obj = _skd.NMF(n_components=output_dimension, **kwargs)
                # Will be called once with all data collected; method is not
                # used in batch mode for this fallback — handled below.
                method = None
            to_print.extend(["scikit-learn estimator:", obj])

        elif algorithm == "ORPCA":
            from hyperspy.learn._rpca import ORPCA

            batch_size = kwargs.pop("batch_size", None)
            obj = ORPCA(output_dimension, **kwargs)
            method = partial(obj.partial_fit, batch_size=batch_size)

        elif algorithm == "ORNMF":
            from hyperspy.learn._ornmf import ORNMF

            batch_size = kwargs.pop("batch_size", None)
            obj = ORNMF(output_dimension, **kwargs)
            method = partial(obj.partial_fit, batch_size=batch_size)

        elif _is_custom_sklearn_like:
            obj = algorithm
            if hasattr(obj, "partial_fit"):
                method = partial(obj.partial_fit)
            else:
                # No incremental fitting; fall back to collecting all data
                # and calling fit_transform / fit+transform once.
                method = None
            to_print.extend(["Custom sklearn-like estimator:", obj])

        elif algorithm == "SVD" and svd_solver == "incremental":
            from hyperspy.learn.incremental_svd import ISVD

            obj = ISVD(n_components=output_dimension)
            method = partial(obj.partial_fit)

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
            # For non-SVD algorithms _navigation_mask_for_reproject stays equal
            # to navigation_mask (no unfolding/ravelling occurs).  For SVD it
            # is updated below after the BaseSignal unwrap but before ravel.
            _navigation_mask_for_reproject = navigation_mask
            if (
                algorithm == "SVD"
                and svd_solver == "incremental"
                and centre is not None
            ):
                import dask.array as _da

                _nav_size = self.axes_manager.navigation_size
                _sig_size = self.axes_manager.signal_size
                _D_flat = self._data_aligned_with_axes.reshape((_nav_size, _sig_size))
                if centre == "navigation":
                    if navigation_mask is not None:
                        _nm = navigation_mask
                        if isinstance(_nm, signals.BaseSignal):
                            _nm = _nm.data
                        if isinstance(_nm, _da.Array):
                            _nm = _nm.compute()
                        _nm_1d = np.asarray(_nm, dtype=bool).ravel()
                        mean = _D_flat[~_nm_1d, :].mean(axis=0, keepdims=True).compute()
                    else:
                        mean = _D_flat.mean(axis=0, keepdims=True).compute()
                else:
                    mean = (
                        _D_flat.mean(axis=1, keepdims=True)
                        .compute()
                        .reshape(self._data_aligned_with_axes.shape[:-1] + (1,))
                    )
                self.data = self.data - mean
            elif algorithm != "SVD" or svd_solver != "incremental":
                mean = None

            # For ISVD, normalise navigation_mask to array-axis order so that
            # _block_iterator (which expects array-axis-order masks) can accept
            # it.  numpy/dask masks arrive in HyperSpy navigation_shape order
            # (axes reversed relative to the underlying array), so they must be
            # transposed.  BaseSignal masks have .data already in array order.
            if (
                algorithm == "SVD"
                and svd_solver == "incremental"
                and navigation_mask is not None
            ):
                if isinstance(navigation_mask, signals.BaseSignal):
                    navigation_mask = navigation_mask.data
                elif hasattr(navigation_mask, "T"):
                    navigation_mask = navigation_mask.T
                _navigation_mask_for_reproject = navigation_mask

            if algorithm == "SVD" and svd_solver != "incremental":
                _svd_result = self._decomposition_svd_matrix(
                    svd_solver,
                    centre,
                    output_dimension,
                    navigation_mask,
                    signal_mask,
                )
                loadings = _svd_result["loadings"]
                factors = _svd_result["factors"]
                explained_variance = _svd_result["explained_variance"]
                mean = _svd_result["mean"]
                nav_mask_1d = _svd_result["nav_mask_1d"]
                sig_mask_1d = _svd_result["sig_mask_1d"]
                navigation_mask = _svd_result["navigation_mask"]
                signal_mask = _svd_result["signal_mask"]
                _navigation_mask_for_reproject = _svd_result[
                    "_navigation_mask_for_reproject"
                ]
                _D_unfolded = _svd_result["_D_unfolded"]
            else:
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
                        if method is not None and len(this_data) == num_chunks:
                            thedata = np.concatenate(this_data, axis=0)
                            method(thedata)
                            this_data = []
                    if len(this_data):
                        if method is not None:
                            thedata = np.concatenate(this_data, axis=0)
                            method(thedata)
                        # else: method is None (NMF fallback or custom w/o
                        # partial_fit); all data is now in this_data for
                        # fit_transform below.
                except KeyboardInterrupt:  # pragma: no cover
                    pass

                # NMF fallback (sklearn < 1.1) and custom objects without
                # partial_fit: collect all chunks and call fit_transform once.
                if method is None:
                    all_data = np.concatenate(this_data, axis=0)
                    if hasattr(obj, "fit_transform"):
                        loadings = obj.fit_transform(all_data)
                    else:
                        obj.fit(all_data)
                        loadings = obj.transform(all_data)

            # GET ALREADY CALCULATED RESULTS
            if algorithm == "SVD" and svd_solver == "incremental":
                # ISVD inherits explained_variance_ / explained_variance_ratio_
                # from sklearn's IncrementalPCA, which computes them assuming
                # centred PCA (S²/(N-1) for variance, S²/Σ(col_var·N) for
                # ratio).  For plain SVD without centring, the correct formulas
                # are S²/N (matching every other decomposition path in
                # HyperSpy) and the ratio is computed downstream by
                # _store_decomposition_results via _compute_explained_variance_ratio.
                S = obj.singular_values_
                n_total = obj.n_samples_seen_
                explained_variance = S**2 / n_total
                explained_variance_ratio = (
                    None  # computed by _store_decomposition_results
                )
                factors = obj.components_.T
                if centre is None:
                    mean = None
                loadings = self._project_loadings(
                    obj, "Project", navigation_mask, signal_mask, get, nblocks
                )

            elif algorithm == "PCA":
                explained_variance = obj.explained_variance_
                explained_variance_ratio = obj.explained_variance_ratio_
                factors = obj.components_.T
                mean = obj.mean_

            elif algorithm in ("NMF", "ORPCA", "ORNMF") or _is_custom_sklearn_like:
                if not hasattr(obj, "components_"):
                    raise AttributeError(
                        f"Fitted estimator {obj!r} has no attribute 'components_'"
                    )
                factors = obj.components_.T
                if hasattr(obj, "explained_variance_"):
                    explained_variance = obj.explained_variance_
                if hasattr(obj, "mean_"):
                    mean = obj.mean_
                else:
                    mean = None
                # Compute loadings via transform if not already set
                # (batch-only objects set loadings above during fit_transform;
                # incremental objects need a project pass now).
                if loadings is None:
                    loadings = self._project_loadings(
                        obj, "Project", navigation_mask, signal_mask, get, nblocks
                    )

            # Pre-compute flat boolean masks needed by the reproject blocks
            # below.  (The same masks are recomputed later outside the try
            # block for storing in learning_results; that duplication is
            # intentional to keep the two concerns separate.)
            _flat_nav_mask = _to_flat_bool(navigation_mask)
            _flat_sig_mask = _to_flat_bool(signal_mask)

            # REPROJECT NAVIGATION (recompute loadings over full nav)
            loadings, _nav_reprojected = self._decomposition_reproject_navigation(
                reproject,
                algorithm,
                svd_solver,
                loadings,
                factors,
                mean,
                centre,
                _D_unfolded,
                sig_mask_1d,
                obj,
                signal_mask,
                get,
                nblocks,
                _navigation_mask_for_reproject,
            )

            # For reproject='signal', non-SVD algorithms need loadings computed
            # first (over masked nav + masked signal), which mirrors
            # reproject=None.  SVD already computed loadings in the learn pass.
            if reproject == "signal" and algorithm != "SVD" and loadings is None:
                try:
                    loadings = self._project_loadings(
                        obj,
                        "Project",
                        _navigation_mask_for_reproject,
                        signal_mask,
                        get,
                        nblocks,
                    )
                except KeyboardInterrupt:  # pragma: no cover
                    pass

            # REPROJECT SIGNAL (recompute factors over full signal)
            # All algorithms support signal reprojection via the pseudo-
            # inverse: factors = pinv(loadings) @ D_full_signal.
            # This mirrors the non-lazy SVD path in _mva.py.
            _signal_reprojected = False
            if reproject in ("signal", "both"):
                factors = self._decomposition_reproject_signal(
                    algorithm,
                    svd_solver,
                    reproject,
                    loadings,
                    factors,
                    mean,
                    centre,
                    _D_unfolded,
                    nav_mask_1d,
                    _navigation_mask_for_reproject,
                    _flat_sig_mask,
                    _flat_nav_mask,
                    get,
                    nblocks,
                )
                _signal_reprojected = True

            # RESHUFFLE "blocked" LOADINGS
            ndim = self.axes_manager.navigation_dimension
            _n_comp = (
                output_dimension
                if output_dimension is not None
                else (factors.shape[1] if factors is not None else None)
            )
            if algorithm != "SVD" and loadings is not None and _n_comp is not None:
                try:
                    loadings = _reshuffle_mixed_blocks(
                        loadings, ndim, (_n_comp,), nav_chunks
                    ).reshape((-1, _n_comp))
                except ValueError:
                    # In case the projection step was not finished, it's left
                    # as scrambled
                    pass
        finally:
            self.data = original_data

        _return_value = self._store_decomposition_results(
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
            factors=factors,
            loadings=loadings,
            mean=mean,
            centre=centre,
            navigation_mask=navigation_mask,
            signal_mask=signal_mask,
            algorithm=algorithm,
            svd_solver=svd_solver,
            obj=obj,
            output_dimension=output_dimension,
            normalize_poissonian_noise=normalize_poissonian_noise,
            _flat_nav_mask=_flat_nav_mask,
            _flat_sig_mask=_flat_sig_mask,
            _nav_reprojected=_nav_reprojected,
            _signal_reprojected=_signal_reprojected,
            print_info=print_info,
            to_print=to_print,
            return_info=return_info,
        )
        if return_info:
            return _return_value

    decomposition.__doc__ %= (
        DECOMP_NORMALIZE_POISSONIAN_NOISE_DOC,
        DECOMP_MASK_DOC
        % (
            "navigation_mask",
            ":class:`~.api.signals.BaseSignal`, numpy.ndarray or dask.array.Array",
            "navigation",
        ),
        DECOMP_MASK_DOC
        % (
            "signal_mask",
            ":class:`~.api.signals.BaseSignal`, numpy.ndarray or dask.array.Array",
            "signal",
        ),
        DECOMP_PRINT_INFO_DOC,
    )

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
