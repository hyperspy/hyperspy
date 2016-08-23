# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import logging

import numpy as np
import dask.array as da
import dask.delayed as dd
from dask.delayed import Delayed as dDelayed

from hyperspy.signal import BaseSignal
from hyperspy.misc.utils import underline
from hyperspy.external.progressbar import progressbar
from hyperspy.external.astroML.histtools import dasky_histogram
from hyperspy.defaults_parser import preferences
from hyperspy.docstrings.signal import (ONE_AXIS_PARAMETER, OUT_ARG)

_logger = logging.getLogger(__name__)

lazyerror = NotImplementedError('This method is not available in lazy signals')


class LazySignal(BaseSignal):
    """A Lazy Signal instance that delays computation until explicitly saved
    (assuming storing the full result of computation in memory is not feasible)
    """
    _lazy = True

    def __init__(self, data, **kwds):
        super().__init__(data, **kwds)
        self.metadata.Signal.lazy = True

    def _compute(self):
        """Only for testing, when able to store the result in memory.."""
        self.data = self.data.compute()
        self.metadata.Signal.lazy = False
        self._assign_subclass()

    def _get_dask_chunks(self, axis=None):
        """Returns dask chunks
        Aims:
            - Have at least one signal (or specified axis) in a single chunk,
            or as many as fit in memory
        Parameters
        ----------
        axis : {int, string, None, axis, tuple}
            If axis is None (default), returns chunks for current data shape so
            that at least one signal is in the chunk. If an axis is specified,
            only that particular axis is guaranteed to be "not sliced".
        Returns
        -------
        Tuple of tuples, dask chunks
        """
        dc = self.data
        for axis in self.axes_manager._axes:
            axis.size = int(dc.shape[axis.index_in_array])

        if axis is not None:
            need_axes = self.axes_manager[axis]
            if not np.iterable(need_axes):
                need_axes = [need_axes, ]
        else:
            need_axes = self.axes_manager.signal_axes

        typesize = dc.dtype.itemsize
        want_to_keep = np.product([ax.size for ax in need_axes]) * typesize

        # @mrocklin reccomends to have around 100MB chunks, so we do that:
        num_that_fit = int(100. * 2.**20 / want_to_keep)

        # want to have at least one "signal" per chunk
        if num_that_fit < 2:
            chunks = [tuple(1 for _ in range(i)) for i in dc.shape]
            for ax in need_axes:
                chunks[ax.index_in_array] = dc.shape[ax.index_in_array],
            return tuple(chunks)

        sizes = [ax.size for ax in self.axes_manager._axes
                 if ax not in need_axes]
        indices = [ax.index_in_array for ax in self.axes_manager._axes
                   if ax not in need_axes]

        while True:
            if np.product(sizes) <= num_that_fit:
                break

            i = np.argmax(sizes)
            sizes[i] = np.floor(sizes[i] / 2)
        chunks = []
        ndim = len(dc.shape)
        for i in range(ndim):
            if i in indices:
                size = float(dc.shape[i])
                split_array = np.array_split(np.arange(size),
                                             np.ceil(size /
                                                     sizes[indices.index(i)]))
                chunks.append(tuple(len(sp) for sp in split_array))
            else:
                chunks.append((dc.shape[i],))
        return tuple(chunks)

    def _make_lazy(self, axis=None):
        new_chunks = self._get_dask_chunks(axis=axis)
        if isinstance(self.data, da.Array):
            if self.data.chunks != new_chunks:
                self.data = self.data.rechunk(new_chunks)
        else:
            self.data = da.from_array(self.data,
                                      chunks=new_chunks)
        self.metadata.Signal.lazy = True

    def _lazy_data(self, axis=None):
        self._make_lazy(axis=axis)
        return self.data

    def _apply_function_on_data_and_remove_axis(self, function, axes,
                                                out=None):
        self._make_lazy()

        def get_dask_function(numpy_name):
            # Translate from the default numpy to dask functions
            translations = {'amax': 'max',
                            'amin': 'min'}
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
        current_data = self._lazy_data(axis=axes)
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
                    "`out` %s" % (new_data.shape, out.data.shape))
        else:
            s = self._deepcopy_with_new_data(new_data)
            s._remove_axis([ax.index_in_axes_manager for ax in axes])
            return s

    def swap_axes(self, *args):
        raise lazyerror

    def rebin(self, new_shape, out=None):
        if len(new_shape) != len(self.data.shape):
            raise ValueError("Wrong shape size")
        new_shape_in_array = []
        for axis in self.axes_manager._axes:
            new_shape_in_array.append(
                new_shape[axis.index_in_axes_manager])
        factors = (np.array(self.data.shape) /
                   np.array(new_shape_in_array))
        axis = {ax.index_in_array: ax for ax in
                self.axes_manager._axes}[factors.argmax()]
        self._make_lazy(axis=axis)
        return super().rebin(new_shape, out=out)
    rebin.__doc__ = BaseSignal.rebin.__doc__

    def __array__(self, dtype=None):
        return self.data.__array__(dtype=dtype)

    def _unfold(self, *args):
        raise lazyerror

    def _make_sure_data_is_contiguous(self, log=None):
        self._make_lazy()

    def diff(self, axis, order=1, out=None):
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
                return dask_diff(
                    arr[slice1] - arr[slice2], n - 1, axis=axis)
            else:
                return arr[slice1] - arr[slice2]

        current_data = self._lazy_data(axis=axis)
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
                    "`out` %s" % (new_data.shape, out.data.shape))
        axis2 = s.axes_manager[axis]
        new_offset = self.axes_manager[axis].offset + (order * axis2.scale / 2)
        axis2.offset = new_offset
        s.get_dimensions_from_data()
        if out is None:
            return s
        else:
            out.events.data_changed.trigger(obj=out)
    diff.__doc__ = BaseSignal.diff.__doc__

    def integrate_simpson(self, axis, out=None):
        axis = self.axes_manager[axis]
        from dask.delayed import do as del_do
        from dask.array.core import slices_from_chunks
        from itertools import product
        from scipy import integrate
        axis = self.axes_manager[axis]
        data = self._lazy_data(axis=axis)
        chunks = data.chunks
        d_chunks = [data[_slice] for _slice in slices_from_chunks(chunks)]
        integs = [del_do(integrate.simps, pure=True)(dc, x=axis.axis,
                                                     axis=axis.index_in_array)
                  for dc in d_chunks]
        shapes = product(*chunks)
        result_list = [
            da.from_delayed(
                integ, shape, dtype=data.dtype) for integ, shape
            in zip(integs, shapes)]
        new_data = da.concatenate(result_list, axis=axis.index_in_array)
        s = out or self._deepcopy_with_new_data(new_data)
        if out:
            if out.data.shape == new_data.shape:
                out.data = new_data
                out.events.data_changed.trigger(obj=out)
            else:
                raise ValueError(
                    "The output shape %s does not match  the shape of "
                    "`out` %s" % (new_data.shape, out.data.shape))
        else:
            s._remove_axis(axis.index_in_axes_manager)
            return s
    integrate_simpson.__doc__ = BaseSignal.integrate_simpson.__doc__

    def valuemax(self, axis, out=None):
        idx = self.indexmax(axis)
        old_data = idx._lazy_data(axis=axis)
        data = old_data.map_blocks(
            lambda x: self.axes_manager[axis].index2value(x))
        if out is None:
            idx.data = data
            return idx
        else:
            out.data = data
            out.events.data_changed.trigger(obj=out)
    valuemax.__doc__ = BaseSignal.valuemax.__doc__

    def get_histogram(self, bins='freedman', out=None,
                      **kwargs):
        if 'range_bins' in kwargs:
            _logger.warning("'range_bins' argument not supported for lazy "
                            "signals")
            del kwargs['range_bins']
        from hyperspy.signals import Signal1D
        data = self._lazy_data().flatten()
        hist, bin_edges = dasky_histogram(data, bins=bins, **kwargs)
        if out is None:
            hist_spec = Signal1D(hist)
            hist_spec.metadata.Signal.lazy = True
            hist_spec._assign_subclass()
        else:
            hist_spec = out
            # we always overwrite the data because the computation is lazy ->
            # the result signal is lazy. Assume that the `out` is already lazy
            hist_spec.data = hist
        hist_spec.axes_manager[0].scale = bin_edges[1] - bin_edges[0]
        hist_spec.axes_manager[0].offset = bin_edges[0]
        hist_spec.axes_manager[0].size = hist.shape[-1]
        hist_spec.axes_manager[0].name = 'value'
        hist_spec.metadata.General.title = (self.metadata.General.title +
                                            " histogram")
        hist_spec.metadata.Signal.binned = True
        if out is None:
            return hist_spec
        else:
            out.events.data_changed.trigger(obj=out)
    get_histogram.__doc__ = BaseSignal.get_histogram.__doc__

    @staticmethod
    def _estimate_poissonian_noise_variance(dc, gain_factor, gain_offset,
                                            correlation_factor):
        variance = (dc * gain_factor + gain_offset) * correlation_factor
        # The lower bound of the variance is the gaussian noise.
        variance = da.clip(variance, gain_offset * correlation_factor, np.inf)
        return variance

    def _get_navigation_signal(self, data=None, dtype=None):
        res = super()._get_navigation_signal(data=data, dtype=dtype)
        if isinstance(res.data, da.Array):
            res = res.as_lazy()
        return res
    _get_navigation_signal.__doc__ = BaseSignal._get_navigation_signal.__doc__

    def _get_signal_signal(self, data=None, dtype=None):
        res = super()._get_signal_signal(data=data, dtype=dtype).as_lazy()
        if isinstance(res.data, da.Array):
            res = res.as_lazy()
        return res
    _get_signal_signal.__doc__ = BaseSignal._get_signal_signal.__doc__

    def _calculate_summary_statistics(self):
        data = self._lazy_data()
        _raveled = data.ravel()
        _mean, _std, _min, _q1, _q2, _q3, _max = da.compute(
            da.nanmean(data),
            da.nanstd(data),
            da.nanmin(data),
            da.percentile(_raveled, [25, ]),
            da.percentile(_raveled, [50, ]),
            da.percentile(_raveled, [75, ]),
            da.nanmax(data),)
        return _mean, _std, _min, _q1, _q2, _q3, _max

    def _map_all(self, function, **kwargs):
        calc_result = dd(function)(self.data, **kwargs)
        self.data = da.from_delayed(calc_result, shape=self.data.shape)

    def _map_iterate(self, function, iterating_kwargs=(),
                     show_progressbar=None, **kwargs):
        _logger.debug("Entering '_map_iterate'")
        self._make_sure_data_is_contiguous()
        orig_shape = self.data.shape
        iterators = tuple(signal[1]._iterate_signal()
                          if isinstance(signal[1], BaseSignal) else signal[1]
                          for signal in iterating_kwargs)
        iterators = (self._iterate_signal(),) + iterators
        all_delayed = []
        pixel_shape = self.axes_manager.signal_shape[::-1]
        _logger.debug("Entering delayed-creating loop")
        for data in zip(*iterators):
            for (key, value), datum in zip(iterating_kwargs, data[1:]):
                if isinstance(value, BaseSignal) and len(datum) == 1:
                    kwargs[key] = datum[0]
                else:
                    kwargs[key] = datum
            all_delayed.append(dd(function)(data[0], **kwargs))
        _logger.debug("Entering dask.array-creating loop")
        pixels = [da.from_delayed(res, shape=pixel_shape) for res in
                  all_delayed]
        _logger.debug("stacking pixels")
        data_stacked = da.stack(pixels, axis=0)
        _logger.debug("reshaping mapped data")
        self.data = data_stacked.reshape(orig_shape)
        _logger.debug("Exit '_map_iterate'")

    def _iterate_signal(self):
        if self.axes_manager.navigation_size < 2:
            yield self()
            return
        self._make_sure_data_is_contiguous()
        nav_dim = self.axes_manager.navigation_dimension
        sig_dim = self.axes_manager.signal_dimension
        from itertools import product
        nav_indices = self.axes_manager.navigation_indices_in_array
        nav_lengths = np.atleast_1d(np.array(self.data.shape)[nav_indices])
        getitem = [slice(None)] * (nav_dim + sig_dim)
        for indices in product(*[range(l) for l in nav_lengths]):
            for res, ind in zip(indices, nav_indices):
                getitem[ind] = res
            yield self.data[tuple(getitem)]
