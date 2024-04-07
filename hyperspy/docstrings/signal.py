# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

"""Common docstring snippets for signal."""

ONE_AXIS_PARAMETER = """: :class:`int`, :class:`str`, or :class:`~hyperspy.axes.DataAxis`
            The axis can be passed directly, or specified using the index of
            the axis in the Signal's `axes_manager` or the axis name."""

MANY_AXIS_PARAMETER = """: :class:`int`, :class:`str`, :class:`~hyperspy.axes.DataAxis` or tuple
            Either one on its own, or many axes in a tuple can be passed. In
            both cases the axes can be passed directly, or specified using the
            index in `axes_manager` or the name of the axis. Any duplicates are
            removed. If ``None``, the operation is performed over all navigation
            axes (default)."""

OUT_ARG = """out : :class:`~hyperspy.api.signals.BaseSignal` (or subclass) or None
            If ``None``, a new Signal is created with the result of the
            operation and returned (default). If a Signal is passed,
            it is used to receive the output of the operation, and nothing is
            returned."""

NAN_FUNC = """Identical to :meth:`~hyperspy.api.signals.BaseSignal.{0}`, except ignores
    missing (NaN) values. See that method's documentation for details.
    """

OPTIMIZE_ARG = """optimize : bool
            If ``True``, the location of the data in memory is optimised for the
            fastest iteration over the navigation axes. This operation can
            cause a peak of memory usage and requires considerable processing
            times for large datasets and/or low specification hardware.
            See the :ref:`signal.transpose` section of the HyperSpy user guide
            for more information. When operating on lazy signals, if ``True``,
            the chunks are optimised for the new axes configuration."""

RECHUNK_ARG = """rechunk : bool
           Only has effect when operating on lazy signal. Default ``False``,
           which means the chunking structure will be retained. If ``True``,
           the data may be automatically rechunked before performing this
           operation."""

SHOW_PROGRESSBAR_ARG = """show_progressbar : None or bool
           If ``True``, display a progress bar. If ``None``, the default from
           the preferences settings is used."""

LAZY_OUTPUT_ARG = """lazy_output : None or bool
           If ``True``, the output will be returned as a lazy signal. This means
           the calculation itself will be delayed until either compute() is used,
           or the signal is stored as a file.
           If ``False``, the output will be returned as a non-lazy signal, this
           means the outputs will be calculated directly, and loaded into memory.
           If ``None`` the output will be lazy if the input signal is lazy, and
           non-lazy if the input signal is non-lazy."""

NUM_WORKERS_ARG = """num_workers : None or int
           Number of worker used by dask. If None, default
           to dask default value."""

CLUSTER_SIGNALS_ARG = """signal : {"mean", "sum", "centroid"}, optional
           If "mean" or "sum" return the mean signal or sum respectively
           over each cluster. If "centroid", returns the signals closest
           to the centroid."""

HISTOGRAM_BIN_ARGS = """bins : int or sequence of float or str, default "fd"
           If ``bins`` is an int, it defines the number of equal-width
           bins in the given range. If ``bins`` is a
           sequence, it defines the bin edges, including the rightmost
           edge, allowing for non-uniform bin widths.

           If ``bins`` is a string from the list below, will use
           the method chosen to calculate the optimal bin width and
           consequently the number of bins (see Notes for more detail on
           the estimators) from the data that falls within the requested
           range. While the bin width will be optimal for the actual data
           in the range, the number of bins will be computed to fill the
           entire range, including the empty portions. For visualisation,
           using the ``'auto'`` option is suggested. Weighted data is not
           supported for automated bin size selection.

           'auto'
               Maximum of the 'sturges' and 'fd' estimators. Provides good
               all around performance.

           'fd' (Freedman Diaconis Estimator)
               Robust (resilient to outliers) estimator that takes into
               account data variability and data size.

           'doane'
               An improved version of Sturges' estimator that works better
               with non-normal datasets.

           'scott'
               Less robust estimator that that takes into account data
               variability and data size.

           'stone'
               Estimator based on leave-one-out cross-validation estimate of
               the integrated squared error. Can be regarded as a generalization
               of Scott's rule.

           'rice'
               Estimator does not take variability into account, only data
               size. Commonly overestimates number of bins required.

           'sturges'
               R's default method, only accounts for data size. Only
               optimal for gaussian data and underestimates number of bins
               for large non-gaussian datasets.

           'sqrt'
               Square root (of data size) estimator, used by Excel and
               other programs for its speed and simplicity.

           'knuth'
               Knuth's rule is a fixed-width, Bayesian approach to determining
               the optimal bin width of a histogram.

           'blocks'
               Determination of optimal adaptive-width histogram bins using
               the Bayesian Blocks algorithm.
    """

HISTOGRAM_MAX_BIN_ARGS = """max_num_bins : int, default 250
           When estimating the bins using one of the str methods, the
           number of bins is capped by this number to avoid a MemoryError
           being raised by :func:`numpy.histogram`."""

SIGNAL_MASK_ARG = """signal_mask : numpy.ndarray of bool
            Restricts the operation to the signal locations not marked
            as True (masked)."""

NAVIGATION_MASK_ARG = """navigation_mask : numpy.ndarray of bool
            Restricts the operation to the navigation locations not marked
            as True (masked)."""

LAZYSIGNAL_DOC = """
    The computation is delayed until explicitly requested.

    This class is not expected to be instantiated directly, instead use:

    >>> data = da.ones((10, 10))
    >>> s = hs.signals.__BASECLASS__(data).as_lazy()
    """
