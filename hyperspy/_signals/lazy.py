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
from functools import partial

import numpy as np
import dask.array as da
import dask.delayed as dd
from dask import threaded
from dask.diagnostics import ProgressBar
from itertools import product

from ..signal import BaseSignal
from ..misc.utils import underline, multiply, dummy_context_manager
from ..external.progressbar import progressbar
from ..external.astroML.histtools import dasky_histogram
from ..defaults_parser import preferences
from ..docstrings.signal import (ONE_AXIS_PARAMETER, OUT_ARG)

_logger = logging.getLogger(__name__)

lazyerror = NotImplementedError('This method is not available in lazy signals')


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
    if isinstance(thing, BaseSignal):
        thing = thing.data
    if chunks is None:
        if isinstance(thing, da.Array):
            thing = thing.compute()
        if isinstance(thing, np.ndarray):
            return thing
        else:
            raise ValueError
    else:
        if isinstance(thing, np.ndarray):
            thing = da.from_array(thing, chunks=chunks)
        if isinstance(thing, da.Array):
            if thing.chunks != chunks:
                thing = thing.rechunk(chunks)
            return thing
        else:
            raise ValueError


class LazySignal(BaseSignal):
    """A Lazy Signal instance that delays computation until explicitly saved
    (assuming storing the full result of computation in memory is not feasible)
    """
    _lazy = True

    def compute(self, progressbar=True):
        """Attempt to store the full signal in memory.."""
        if progressbar:
            cm = ProgressBar
        else:
            cm = dummy_context_manager
        with cm():
            self.data = self.data.compute()
        self._lazy = False
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
        dcshape = dc.shape
        for _axis in self.axes_manager._axes:
            if _axis.index_in_array < len(dcshape):
                _axis.size = int(dcshape[_axis.index_in_array])

        if axis is not None:
            need_axes = self.axes_manager[axis]
            if not np.iterable(need_axes):
                need_axes = [need_axes, ]
        else:
            need_axes = self.axes_manager.signal_axes

        typesize = dc.dtype.itemsize
        want_to_keep = multiply([ax.size for ax in need_axes]) * typesize

        # @mrocklin reccomends to have around 100MB chunks, so we do that:
        num_that_fit = int(100. * 2.**20 / want_to_keep)

        # want to have at least one "signal" per chunk
        if num_that_fit < 2:
            chunks = [tuple(1 for _ in range(i)) for i in dc.shape]
            for ax in need_axes:
                chunks[ax.index_in_array] = dc.shape[ax.index_in_array],
            return tuple(chunks)

        sizes = [
            ax.size for ax in self.axes_manager._axes if ax not in need_axes
        ]
        indices = [
            ax.index_in_array for ax in self.axes_manager._axes
            if ax not in need_axes
        ]

        while True:
            if multiply(sizes) <= num_that_fit:
                break

            i = np.argmax(sizes)
            sizes[i] = np.floor(sizes[i] / 2)
        chunks = []
        ndim = len(dc.shape)
        for i in range(ndim):
            if i in indices:
                size = float(dc.shape[i])
                split_array = np.array_split(
                    np.arange(size), np.ceil(size / sizes[indices.index(i)]))
                chunks.append(tuple(len(sp) for sp in split_array))
            else:
                chunks.append((dc.shape[i], ))
        return tuple(chunks)

    def _make_lazy(self, axis=None, rechunk=False):
        self.data = self._lazy_data(axis=axis, rechunk=rechunk)

    def _lazy_data(self, axis=None, rechunk=True):
        new_chunks = self._get_dask_chunks(axis=axis)
        if isinstance(self.data, da.Array):
            res = self.data
            if self.data.chunks != new_chunks and rechunk:
                res = self.data.rechunk(new_chunks)
        else:
            res = da.from_array(self.data, chunks=new_chunks)
        assert isinstance(res, da.Array)
        return res

    def _apply_function_on_data_and_remove_axis(self, function, axes,
                                                out=None):
        def get_dask_function(numpy_name):
            # Translate from the default numpy to dask functions
            translations = {'amax': 'max', 'amin': 'min'}
            if numpy_name in translations:
                numpy_name = translations[numpy_name]
            return getattr(da, numpy_name)

        function = get_dask_function(function.__name__)
        axes = self.axes_manager[axes]
        if not np.iterable(axes):
            axes = (axes, )
        ar_axes = tuple(ax.index_in_array for ax in axes)
        if len(ar_axes) == 1:
            ar_axes = ar_axes[0]
        current_data = self._lazy_data(axis=axes)
        new_data = function(current_data, axis=ar_axes)
        if not new_data.ndim:
            new_data = new_data.reshape((1, ))
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
            new_shape_in_array.append(new_shape[axis.index_in_axes_manager])
        factors = (np.array(self.data.shape) / np.array(new_shape_in_array))
        axis = {ax.index_in_array: ax
                for ax in self.axes_manager._axes}[factors.argmax()]
        self._make_lazy(axis=axis)
        return super().rebin(new_shape, out=out)

    rebin.__doc__ = BaseSignal.rebin.__doc__

    def __array__(self, dtype=None):
        return self.data.__array__(dtype=dtype)

    def _unfold(self, *args):
        raise lazyerror

    def _make_sure_data_is_contiguous(self, log=None):
        self._make_lazy(rechunk=True)

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
                return dask_diff(arr[slice1] - arr[slice2], n - 1, axis=axis)
            else:
                return arr[slice1] - arr[slice2]

        current_data = self._lazy_data(axis=axis)
        new_data = dask_diff(current_data, order, arr_axis)
        if not new_data.ndim:
            new_data = new_data.reshape((1, ))

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
        from scipy import integrate
        axis = self.axes_manager[axis]
        data = self._lazy_data(axis=axis)
        new_data = data.map_blocks(
            integrate.simps,
            x=axis.axis,
            axis=axis.index_in_array,
            drop_axis=axis.index_in_array,
            dtype=data.dtype)
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
        old_data = idx.data
        data = old_data.map_blocks(
            lambda x: self.axes_manager[axis].index2value(x))
        if out is None:
            idx.data = data
            return idx
        else:
            out.data = data
            out.events.data_changed.trigger(obj=out)

    valuemax.__doc__ = BaseSignal.valuemax.__doc__

    def get_histogram(self, bins='freedman', out=None, **kwargs):
        if 'range_bins' in kwargs:
            _logger.warning("'range_bins' argument not supported for lazy "
                            "signals")
            del kwargs['range_bins']
        from hyperspy.signals import Signal1D
        data = self._lazy_data().flatten()
        hist, bin_edges = dasky_histogram(data, bins=bins, **kwargs)
        if out is None:
            hist_spec = Signal1D(hist)
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
        hist_spec.axes_manager[0].name = 'value'
        hist_spec.metadata.General.title = (
            self.metadata.General.title + " histogram")
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

    # def _get_navigation_signal(self, data=None, dtype=None):
    #     return super()._get_navigation_signal(data=data, dtype=dtype).as_lazy()

    # _get_navigation_signal.__doc__ = BaseSignal._get_navigation_signal.__doc__

    # def _get_signal_signal(self, data=None, dtype=None):
    #     return super()._get_signal_signal(data=data, dtype=dtype).as_lazy()

    # _get_signal_signal.__doc__ = BaseSignal._get_signal_signal.__doc__

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
            da.nanmax(data), )
        return _mean, _std, _min, _q1, _q2, _q3, _max

    def _map_all(self, function, inplace=True, **kwargs):
        calc_result = dd(function)(self.data, **kwargs)
        if inplace:
            self.data = da.from_delayed(calc_result, shape=self.data.shape,
                                        dtype=self.data.dtype)
            return None
        return self._deepcopy_with_new_data(calc_result)

    def _map_iterate(self,
                     function,
                     iterating_kwargs=(),
                     show_progressbar=None,
                     parallel=None,
                     ragged=None,
                     inplace=True,
                     **kwargs):
        if ragged not in (True, False):
            raise ValueError('"ragged" kwarg has to be bool for lazy signals')
        _logger.debug("Entering '_map_iterate'")

        size = max(1, self.axes_manager.navigation_size)
        from hyperspy.misc.utils import (create_map_objects,
                                         map_result_construction)
        func, iterators = create_map_objects(function, size, iterating_kwargs,
                                             **kwargs)
        iterators = (self._iterate_signal(), ) + iterators
        res_shape = self.axes_manager._navigation_shape_in_array
        # no navigation
        if not len(res_shape) and ragged:
            res_shape = (1,)

        all_delayed = [dd(func)(data) for data in zip(*iterators)]

        if ragged:
            sig_shape = ()
            sig_dtype = np.dtype('O')
        else:
            one_compute = all_delayed[0].compute()
            sig_shape = one_compute.shape
            sig_dtype = one_compute.dtype
        pixels = [
            da.from_delayed(
                res, shape=sig_shape, dtype=sig_dtype) for res in all_delayed
        ]

        for step in reversed(res_shape):
            _len = len(pixels)
            starts = range(0, _len, step)
            ends = range(step, _len + step, step)
            pixels = [
                da.stack(
                    pixels[s:e], axis=0) for s, e in zip(starts, ends)
            ]
        result = pixels[0]
        res = map_result_construction(
            self, inplace, result, ragged, sig_shape, lazy=True)
        return res

    def _iterate_signal(self):
        if self.axes_manager.navigation_size < 2:
            yield self()
            return
        nav_dim = self.axes_manager.navigation_dimension
        sig_dim = self.axes_manager.signal_dimension
        nav_indices = self.axes_manager.navigation_indices_in_array[::-1]
        nav_lengths = np.atleast_1d(
            np.array(self.data.shape)[list(nav_indices)])
        getitem = [slice(None)] * (nav_dim + sig_dim)
        data = self._lazy_data()
        for indices in product(*[range(l) for l in nav_lengths]):
            for res, ind in zip(indices, nav_indices):
                getitem[ind] = res
            yield data[tuple(getitem)]

    def _block_iterator(self,
                        flat_signal=True,
                        get=threaded.get,
                        navigation_mask=None,
                        signal_mask=None):
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
        get : dask scheduler
            the dask scheduler to use for computations;
            default `dask.threaded.get`
        navigation_mask : {BaseSignal, numpy array, dask array}
            The navigation locations marked as True are not returned (flat) or
            set to NaN or 0.
        signal_mask : {BaseSignal, numpy array, dask array} 
            The signal locations marked as True are not returned (flat) or set
            to NaN or 0.

        """
        self._make_lazy()
        data = self._data_aligned_with_axes
        nav_chunks = data.chunks[:self.axes_manager.navigation_dimension]
        indices = product(*[range(len(c)) for c in nav_chunks])
        signalsize = self.axes_manager.signal_size
        sig_reshape = (signalsize,) if signalsize else ()
        data = data.reshape((self.axes_manager.navigation_shape[::-1] +
                             sig_reshape))

        if signal_mask is None:
            signal_mask = slice(None) if flat_signal else \
                    np.zeros(self.axes_manager.signal_size, dtype='bool')
        else:
            try:
                signal_mask = to_array(signal_mask).ravel()
            except ValueError:
                # re-raise with a message
                raise ValueError("signal_mask has to be a signal, numpy or"
                                 " dask array, but "
                                 "{} was given".format(type(signal_mask)))
            if flat_signal:
                signal_mask = ~signal_mask

        if navigation_mask is None:
            nav_mask = da.zeros(
                self.axes_manager.navigation_shape[::-1],
                chunks=nav_chunks,
                dtype='bool')
        else:
            try:
                nav_mask = to_array(navigation_mask, chunks=nav_chunks)
            except ValueError:
                # re-raise with a message
                raise ValueError("navigation_mask has to be a signal, numpy or"
                                 " dask array, but "
                                 "{} was given".format(type(navigation_mask)))
        if flat_signal:
            nav_mask = ~nav_mask
        for ind in indices:
            chunk = get(data.dask,
                        (data.name, ) + ind + (0,)*bool(signalsize))
            n_mask = get(nav_mask.dask, (nav_mask.name, ) + ind)
            if flat_signal:
                yield chunk[n_mask, ...][..., signal_mask]
            else:
                chunk = chunk.copy()
                value = np.nan if np.can_cast('float', chunk.dtype) else 0
                chunk[n_mask, ...] = value
                chunk[..., signal_mask] = value
                yield chunk.reshape(chunk.shape[:-1] +
                                    self.axes_manager.signal_shape[::-1])

    def decomposition(self,
                      output_dimension,
                      normalize_poissonian_noise=False,
                      algorithm='PCA',
                      signal_mask=None,
                      navigation_mask=None,
                      get=threaded.get,
                      num_chunks=None,
                      reproject=True,
                      bounds=True,
                      **kwargs):
        """Perform Incremental (Batch) decomposition on the data, keeping n
        significant components.

        Parameters
        ----------
        output_dimension : int
            the number of significant components to keep
        normalize_poissonian_noise : bool
            If True, scale the SI to normalize Poissonian noise
        algorithm : str
            One of ('PCA', 'ORPCA', 'ONMF'). By default ('PCA') IncrementalPCA
            from scikit-learn is run.
        get : dask scheduler
            the dask scheduler to use for computations;
            default `dask.threaded.get`
        num_chunks : int
            the number of dask chunks to pass to the decomposition model.
            More chunks require more memory, but should run faster. Will be
            increased to contain atleast output_dimension signals.
        navigation_mask : {BaseSignal, numpy array, dask array}
            The navigation locations marked as True are not used in the
            decompostion.
        signal_mask : {BaseSignal, numpy array, dask array}
            The signal locations marked as True are not used in the
            decomposition.
        reproject : bool
            Reproject data on the learnt components (factors) after learning.
        bounds : {tuple, bool}
            The (min, max) values of the data to normalize before learning.
            If tuple (min, max), those values will be used for normalization.
            If True, extremes will be looked up (expensive), default.
            If False, no normalization is done (learning may be very slow).
            If normalize_poissonian_noise is True, this cannot be True.
        **kwargs
            passed to the partial_fit/fit functions.

        Notes
        -----
        Various algorithm parameters and their default values:
            ONMF:
                lambda1=1,
                kappa=1,
                robust=False,
                store_r=False
                batch_size=None
            ORPCA:
                fast=True,
                lambda1=None,
                lambda2=None,
                method=None,
                learning_rate=None,
                init=None,
                training_samples=None,
                momentum=None
            PCA:
                batch_size=None,
                copy=True,
                white=False


        """
        explained_variance = None
        explained_variance_ratio = None
        _al_data = self._data_aligned_with_axes
        nav_chunks = _al_data.chunks[:self.axes_manager.navigation_dimension]
        sig_chunks = _al_data.chunks[self.axes_manager.navigation_dimension:]

        num_chunks = 1 if num_chunks is None else num_chunks
        blocksize = np.min([multiply(ar) for ar in product(*nav_chunks)])
        nblocks = multiply([len(c) for c in nav_chunks])
        if blocksize / output_dimension < num_chunks:
            num_chunks = np.ceil(blocksize / output_dimension)
        blocksize *= num_chunks

        ## LEARN
        if algorithm == 'PCA':
            from sklearn.decomposition import IncrementalPCA
            obj = IncrementalPCA(n_components=output_dimension)
            method = partial(obj.partial_fit, **kwargs)
            reproject = True

        elif algorithm == 'ORPCA':
            from hyperspy.learn.rpca import ORPCA
            kwg = {'fast': True}
            kwg.update(kwargs)
            obj = ORPCA(output_dimension, **kwg)
            method = partial(obj.fit, iterating=True)

        elif algorithm == 'ONMF':
            from hyperspy.learn.onmf import ONMF
            batch_size = kwargs.pop('batch_size', None)
            obj = ONMF(output_dimension, **kwargs)
            method = partial(obj.fit, batch_size=batch_size)

        else:
            raise ValueError('algorithm not known')

        original_data = self.data
        try:
            if normalize_poissonian_noise:
                if bounds is True:
                    bounds = False
                    # warnings.warn?
                data = self._data_aligned_with_axes
                ndim = self.axes_manager.navigation_dimension
                sdim = self.axes_manager.signal_dimension
                nm = da.logical_not(
                    da.zeros(
                        self.axes_manager.navigation_shape[::-1],
                        chunks=nav_chunks)
                    if navigation_mask is None else to_array(
                        navigation_mask, chunks=nav_chunks))
                sm = da.logical_not(
                    da.zeros(
                        self.axes_manager.signal_shape[::-1],
                        chunks=sig_chunks)
                    if signal_mask is None else to_array(
                        signal_mask, chunks=sig_chunks))
                ndim = self.axes_manager.navigation_dimension
                sdim = self.axes_manager.signal_dimension
                bH, aG = da.compute(
                    data.sum(axis=range(ndim)),
                    data.sum(axis=range(ndim, ndim + sdim)))
                bH = da.where(sm, bH, 1)
                aG = da.where(nm, aG, 1)

                raG = da.sqrt(aG)
                rbH = da.sqrt(bH)

                coeff = raG[(..., ) + (None, )*rbH.ndim] *\
                        rbH[(None, )*raG.ndim + (...,)]
                coeff.map_blocks(np.nan_to_num)
                coeff = da.where(coeff == 0, 1, coeff)
                data = data / coeff
                self.data = data

            # normalize the data for learning algs:
            if bounds:
                if bounds is True:
                    _min, _max = da.compute(self.data.min(), self.data.max())
                else:
                    _min, _max = bounds
                self.data = (self.data - _min) / (_max - _min)

            # LEARN
            this_data = []
            try:
                for chunk in progressbar(
                        self._block_iterator(
                            flat_signal=True,
                            get=get,
                            signal_mask=signal_mask,
                            navigation_mask=navigation_mask),
                        total=nblocks,
                        leave=True,
                        desc='Learn'):
                    this_data.append(chunk)
                    if len(this_data) == num_chunks:
                        thedata = np.concatenate(this_data, axis=0)
                        method(thedata)
                        this_data = []
                if len(this_data):
                    thedata = np.concatenate(this_data, axis=0)
                    method(thedata)
            except KeyboardInterrupt:
                pass

            # GET ALREADY CALCULATED RESULTS
            if algorithm == 'PCA':
                explained_variance = obj.explained_variance_
                explained_variance_ratio = obj.explained_variance_ratio_
                factors = obj.components_.T

            elif algorithm == 'ORPCA':
                _, _, U, S, V = obj.finish()
                factors = U * S
                loadings = V
                explained_variance = S**2 / len(factors)

            elif algorithm == 'ONMF':
                factors, loadings = obj.finish()
                loadings = loadings.T

            # REPROJECT
            if reproject:
                if algorithm == 'PCA':
                    method = obj.transform
                    post = lambda a: np.concatenate(a, axis=0)
                elif algorithm == 'ORPCA':
                    method = obj.project
                    obj.R = []
                    post = lambda a: obj.finish()[4]
                elif algorithm == 'ONMF':
                    method = obj.project
                    post = lambda a: np.concatenate(a, axis=1).T

                _map = map(lambda thing: method(thing),
                           self._block_iterator(
                               flat_signal=True,
                               get=get,
                               signal_mask=signal_mask,
                               navigation_mask=navigation_mask))
                H = []
                try:
                    for thing in progressbar(
                            _map, total=nblocks, desc='Project'):
                        H.append(thing)
                except KeyboardInterrupt:
                    pass
                loadings = post(H)

            if explained_variance is not None and \
                    explained_variance_ratio is None:
                explained_variance_ratio = \
                    explained_variance / explained_variance.sum()

            # RESHUFFLE "blocked" LOADINGS
            ndim = self.axes_manager.navigation_dimension
            try:
                loadings = _reshuffle_mixed_blocks(
                    loadings,
                    ndim,
                    (output_dimension,),
                    nav_chunks).reshape((-1, output_dimension))
            except ValueError:
                # In case the projection step was not finished, it's left
                # as scrambled
                pass
        finally:
            self.data = original_data

        target = self.learning_results
        target.decomposition_algorithm = algorithm
        target.output_dimension = output_dimension
        target._object = obj
        target.factors = factors
        target.loadings = loadings
        target.explained_variance = explained_variance
        target.explained_variance_ratio = explained_variance_ratio


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
    splits = np.cumsum([multiply(ar)
                        for ar in product(*nav_chunks)][:-1]).tolist()
    if splits:
        all_chunks = [
            ar.reshape(shape + sshape)
            for shape, ar in zip(
                product(*nav_chunks), np.split(array, splits))
        ]

        def split_stack_list(what, step, axis):
            total = len(what)
            if total != step:
                return [
                    np.concatenate(
                        what[i:i + step], axis=axis)
                    for i in range(0, total, step)
                ]
            else:
                return np.concatenate(what, axis=axis)

        for chunks, axis in zip(nav_chunks[::-1], range(ndim-1, -1, -1)):
            step = len(chunks)
            all_chunks = split_stack_list(all_chunks, step, axis)
        return all_chunks
    else:
        return array
