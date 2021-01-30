# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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
import warnings

import dask.array as da
import numpy as np

from hyperspy.docstrings.signal import HISTOGRAM_BIN_ARGS, HISTOGRAM_MAX_BIN_ARGS
from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.external.astropy.bayesian_blocks import bayesian_blocks
from hyperspy.external.astropy.histogram import knuth_bin_width

_logger = logging.getLogger(__name__)


def histogram(a, bins="fd", range=None, max_num_bins=250, weights=None, **kwargs):
    """Enhanced histogram.

    This is a histogram function that enables the use of more sophisticated
    algorithms for determining bins.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    %s
    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored. The first element of the range must be less than or
        equal to the second. `range` affects the automatic bin
        computation as well. While bin width is computed to be optimal
        based on the actual data within `range`, the bin count will fill
        the entire range including portions containing no data.
    %s
    weights : array_like, optional
        An array of weights, of the same shape as `a`.  Each value in
        `a` only contributes its associated weight towards the bin count
        (instead of 1). This is currently not used by any of the bin estimators,
        but may be in the future.
    **kwargs :
        Passed to :py:func:`numpy.histogram`

    Returns
    -------
    hist : array
        The values of the histogram. See `normed` and `weights` for a
        description of the possible semantics.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.

    See Also
    --------
    * :py:func:`numpy.histogram`

    """
    if isinstance(a, da.Array):
        return histogram_dask(a, bins=bins, max_num_bins=max_num_bins, **kwargs)

    if isinstance(bins, str):
        _deprecated_bins = {"scotts": "scott", "freedman": "fd"}
        new_bins = _deprecated_bins.get(bins, None)
        if new_bins:
            warnings.warn(
                f"`bins='{bins}'` has been deprecated and will be removed "
                f"in HyperSpy 2.0. Please use `bins='{new_bins}'` instead.",
                VisibleDeprecationWarning,
                )
            bins = new_bins

    _old_bins = bins

    if isinstance(bins, str) and bins in ["knuth", "blocks"]:
        # if range is specified, we need to truncate
        # the data for these bin-finding routines
        if range is not None:
            a = a[(a >= range[0]) & (a <= range[1])]

        if bins == "knuth":
            _, bins = knuth_bin_width(a, return_bins=True, max_num_bins=max_num_bins)
        elif bins == "blocks":
            bins = bayesian_blocks(a)
    else:
        bins = np.histogram_bin_edges(a, bins=bins, range=range, weights=weights)

    _bins_len = bins if not np.iterable(bins) else len(bins)

    if _bins_len > max_num_bins:
        # To avoid memory errors such as that detailed in
        # https://github.com/hyperspy/hyperspy/issues/784,
        # we log a warning and cap the number of bins at
        # a sensible value.
        _logger.warning(
            f"Estimated number of bins using `bins='{_old_bins}'` "
            f"is too large ({_bins_len}). Capping the number of bins "
            f"at `max_num_bins={max_num_bins}`. Consider using an "
            "alternative method for calculating the bins such as "
            "`bins='scott'`, or increasing the value of the "
            "`max_num_bins` keyword argument."
        )
        bins = max_num_bins

    return np.histogram(a, bins=bins, **kwargs)


histogram.__doc__ %= (HISTOGRAM_BIN_ARGS, HISTOGRAM_MAX_BIN_ARGS)


def histogram_dask(a, bins="fd", max_num_bins=250, **kwargs):
    """Enhanced histogram for dask arrays.

    The range keyword is ignored. Reads the data at most two times - once to
    determine best bins (if required), and second time to actually calculate
    the histogram.

    Parameters
    ----------
    a : array_like
        array of data to be histogrammed
    bins : int or list or str, default 10
        If bins is a string, then it must be one of:

        'fd' (Freedman Diaconis Estimator)
            Robust (resilient to outliers) estimator that takes into
            account data variability and data size.

        'scott'
            Less robust estimator that that takes into account data
            variability and data size.

    %s
    **kwargs :
        Passed to :py:func:`dask.histogram`

    Returns
    -------
    hist : array
        The values of the histogram. See `normed` and `weights` for a
        description of the possible semantics.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.

    See Also
    --------
    * :py:func:`dask.histogram`
    * :py:func:`numpy.histogram`

    """
    if not isinstance(a, da.Array):
        raise TypeError("Expected a dask array")

    if a.ndim != 1:
        a = a.flatten()

    if isinstance(bins, str):
        _deprecated_bins = {"scotts": "scott", "freedman": "fd"}
        new_bins = _deprecated_bins.get(bins, None)
        if new_bins:
            warnings.warn(
                f"`bins='{bins}'` has been deprecated and will be removed "
                f"in HyperSpy 2.0. Please use `bins='{new_bins}'` instead.",
                VisibleDeprecationWarning,
                )
            bins = new_bins

    _old_bins = bins

    if isinstance(bins, str):
        if bins == "scott":
            _, bins = _scott_bw_dask(a, True)
        elif bins == "fd":
            _, bins = _freedman_bw_dask(a, True)
        else:
            raise ValueError(f"Unrecognized 'bins' argument: got {bins}")
    elif not np.iterable(bins):
        kwargs["range"] = da.compute(a.min(), a.max())

    _bins_len = bins if not np.iterable(bins) else len(bins)

    if _bins_len > max_num_bins:
        # To avoid memory errors such as that detailed in
        # https://github.com/hyperspy/hyperspy/issues/784,
        # we log a warning and cap the number of bins at
        # a sensible value.
        _logger.warning(
            f"Estimated number of bins using `bins='{_old_bins}'` "
            f"is too large ({_bins_len}). Capping the number of bins "
            f"at `max_num_bins={max_num_bins}`. Consider using an "
            "alternative method for calculating the bins such as "
            "`bins='scott'`, or increasing the value of the "
            "`max_num_bins` keyword argument."
        )
        bins = max_num_bins
        kwargs["range"] = da.compute(a.min(), a.max())

    h, bins = da.histogram(a, bins=bins, **kwargs)

    return h.compute(), bins


histogram_dask.__doc__ %= HISTOGRAM_MAX_BIN_ARGS


def _scott_bw_dask(data, return_bins=True):
    r"""Dask version of scotts_bin_width

    Parameters
    ----------
    data : dask array
        the data
    return_bins : bool (optional)
        if True, then return the bin edges

    Returns
    -------
    width : float
        optimal bin width using Scott's rule
    bins : ndarray
        bin edges: returned if `return_bins` is True

    Notes
    -----
    The optimal bin width is:

    .. math::

        \Delta_b = \frac{3.5\sigma}{n^{1/3}}

    where :math:`\sigma` is the standard deviation of the data, and
    :math:`n` is the number of data points.

    """
    if not isinstance(data, da.Array):
        raise TypeError("Expected a dask array")

    if data.ndim != 1:
        data = data.flatten()

    n = data.size
    sigma = da.nanstd(data)
    dx = 3.5 * sigma * n ** (-1.0 / 3.0)
    c_dx, mx, mn = da.compute(dx, data.max(), data.min())

    if return_bins:
        Nbins = max(1, np.ceil((mx - mn) / c_dx))
        bins = mn + c_dx * np.arange(Nbins + 1)
        return c_dx, bins
    else:
        return c_dx


def _freedman_bw_dask(data, return_bins=True):
    r"""Dask version of freedman_bin_width

    Parameters
    ----------
    data : dask array
        the data
    return_bins : bool (optional)
        if True, then return the bin edges

    Returns
    -------
    width : float
        optimal bin width using Scott's rule
    bins : ndarray
        bin edges: returned if `return_bins` is True

    Notes
    -----
    The optimal bin width is

    .. math::

        \Delta_b = \frac{2(q_{75} - q_{25})}{n^{1/3}}

    where :math:`q_{N}` is the :math:`N` percent quartile of the data, and
    :math:`n` is the number of data points.

    """
    if not isinstance(data, da.Array):
        raise TypeError("Expected a dask array")

    if data.ndim != 1:
        data = data.flatten()

    n = data.size

    v25, v75 = da.percentile(data, [25, 75])
    dx = 2 * (v75 - v25) * n ** (-1.0 / 3.0)
    c_dx, mx, mn = da.compute(dx, data.max(), data.min())

    if return_bins:
        Nbins = max(1, np.ceil((mx - mn) / c_dx))
        bins = mn + c_dx * np.arange(Nbins + 1)
        return c_dx, bins
    else:
        return c_dx
