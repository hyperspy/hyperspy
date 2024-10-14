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

import logging
import math
import warnings

import dask.array as da
import numpy as np
import numpy.ma as ma
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt, savgol_filter

from hyperspy._signals.common_signal1d import CommonSignal1D
from hyperspy._signals.lazy import LazySignal
from hyperspy.decorators import interactive_range_selector
from hyperspy.defaults_parser import preferences
from hyperspy.docstrings.plot import (
    BASE_PLOT_DOCSTRING,
    BASE_PLOT_DOCSTRING_PARAMETERS,
    PLOT1D_DOCSTRING,
)
from hyperspy.docstrings.signal import (
    IN_PLACE,
    LAZYSIGNAL_DOC,
    NAVIGATION_MASK_ARG,
    NUM_WORKERS_ARG,
    SHOW_PROGRESSBAR_ARG,
    SIGNAL_MASK_ARG,
)
from hyperspy.docstrings.signal1d import (
    CROP_PARAMETER_DOC,
    SPIKES_REMOVAL_TOOL_DOCSTRING,
)
from hyperspy.misc.lowess_smooth import lowess
from hyperspy.misc.tv_denoise import _tv_denoise_1d
from hyperspy.models.model1d import Model1D
from hyperspy.signal import BaseSignal
from hyperspy.signal_tools import (
    BackgroundRemoval,
    ButterworthFilter,
    Signal1DCalibration,
    SimpleMessage,
    SmoothingLowess,
    SmoothingSavitzkyGolay,
    SmoothingTV,
    SpikesRemoval,
    SpikesRemovalInteractive,
    _get_background_estimator,
)
from hyperspy.ui_registry import DISPLAY_DT, TOOLKIT_DT

_logger = logging.getLogger(__name__)


def find_peaks_ohaver(
    y,
    x=None,
    slope_thresh=0.0,
    amp_thresh=None,
    medfilt_radius=5,
    maxpeakn=30000,
    peakgroup=10,
    subchannel=True,
):
    """Find peaks along a 1D line.

    Function to locate the positive peaks in a noisy x-y data set.
    Detects peaks by looking for downward zero-crossings in the first
    derivative that exceed 'slope_thresh'.
    Returns an array containing position, height, and width of each peak.
    Sorted by position.
    'slope_thresh' and 'amp_thresh', control sensitivity: higher values
    will neglect wider peaks (slope) and smaller features (amp),
    respectively.

    Parameters
    ----------

    y : array
        1D input array, e.g. a spectrum
    x : array (optional)
        1D array describing the calibration of y (must have same shape as y)
    slope_thresh : float (optional)
                   1st derivative threshold to count the peak;
                   higher values will neglect broader features;
                   default is set to 0.
    amp_thresh : float (optional)
                 intensity threshold below which peaks are ignored;
                 higher values will neglect smaller features;
                 default is set to 10% of max(y).
    medfilt_radius : int (optional)
                     median filter window to apply to smooth the data
                     (see scipy.signal.medfilt);
                     if 0, no filter will be applied;
                     default is set to 5.
    peakgroup : int (optional)
                number of points around the "top part" of the peak that
                are taken to estimate the peak height; for spikes or
                very narrow peaks, keep PeakGroup=1 or 2; for broad or
                noisy peaks, make PeakGroup larger to reduce the effect
                of noise;
                default is set to 10.
    maxpeakn : int (optional)
              number of maximum detectable peaks;
              default is set to 30000.
    subchannel : bool (optional)
             default is set to True.

    Returns
    -------
    P : structured array of shape (npeaks)
        contains fields: 'position', 'width', and 'height' for each peak.

    Examples
    --------
    >>> x = np.arange(0,50,0.01)
    >>> y = np.cos(x)
    >>> peaks = find_peaks_ohaver(y, x, 0, 0)

    Notes
    -----
    Original code from T. C. O'Haver, 1995.
    Version 2  Last revised Oct 27, 2006 Converted to Python by
    Michael Sarahan, Feb 2011.
    Revised to handle edges better.  MCS, Mar 2011
    """

    if x is None:
        x = np.arange(len(y), dtype=np.int64)
    if not amp_thresh:
        amp_thresh = 0.1 * y.max()
    peakgroup = np.round(peakgroup)
    if medfilt_radius:
        d = np.gradient(medfilt(y, medfilt_radius))
    else:
        d = np.gradient(y)
    n = np.round(peakgroup / 2 + 1)
    peak_dt = np.dtype([("position", float), ("height", float), ("width", float)])
    P = np.array([], dtype=peak_dt)
    peak = 0
    for j in range(len(y) - 4):
        if np.sign(d[j]) > np.sign(d[j + 1]):  # Detects zero-crossing
            if np.sign(d[j + 1]) == 0:
                continue
            # if slope of derivative is larger than slope_thresh
            if d[j] - d[j + 1] > slope_thresh:
                # if height of peak is larger than amp_thresh
                if y[j] > amp_thresh:
                    # the next section is very slow, and actually messes
                    # things up for images (discrete pixels),
                    # so by default, don't do subchannel precision in the
                    # 1D peakfind step.
                    if subchannel:
                        xx = np.zeros(peakgroup)
                        yy = np.zeros(peakgroup)
                        s = 0
                        for k in range(peakgroup):
                            groupindex = int(j + k - n + 1)
                            if groupindex < 1:
                                xx = xx[1:]
                                yy = yy[1:]
                                s += 1
                                continue
                            elif groupindex > y.shape[0] - 1:
                                xx = xx[: groupindex - 1]
                                yy = yy[: groupindex - 1]
                                break
                            xx[k - s] = x[groupindex]
                            yy[k - s] = y[groupindex]
                        avg = np.average(xx)
                        stdev = np.std(xx)
                        xxf = (xx - avg) / stdev
                        # Fit parabola to log10 of sub-group with
                        # centering and scaling
                        yynz = yy != 0
                        coef = np.polyfit(xxf[yynz], np.log10(abs(yy[yynz])), 2)
                        c1 = coef[2]
                        c2 = coef[1]
                        c3 = coef[0]
                        with np.errstate(invalid="ignore"):
                            width = np.linalg.norm(
                                stdev * 2.35703 / (np.sqrt(2) * np.sqrt(-1 * c3))
                            )
                        # if the peak is too narrow for least-squares
                        # technique to work  well, just use the max value
                        # of y in the sub-group of points near peak.
                        if peakgroup < 7:
                            height = np.max(yy)
                            position = xx[np.argmin(abs(yy - height))]
                        else:
                            position = -((stdev * c2 / (2 * c3)) - avg)
                            height = np.exp(c1 - c3 * (c2 / (2 * c3)) ** 2)
                    # Fill results array P. One row for each peak
                    # detected, containing the
                    # peak position (x-value) and peak height (y-value).
                    else:
                        position = x[j]
                        height = y[j]
                        # no way to know peak width without
                        # the above measurements.
                        width = 0
                    if not np.isnan(position) and 0 < position < x[-1]:
                        P = np.hstack(
                            (P, np.array([(position, height, width)], dtype=peak_dt))
                        )
                        peak += 1
    # return only the part of the array that contains peaks
    # (not the whole maxpeakn x 3 array)
    if len(P) > maxpeakn:
        minh = np.sort(P["height"])[-maxpeakn]
        P = P[P["height"] >= minh]

    # Sorts the values as a function of position
    P.sort(0)

    return P


def interpolate1D(number_of_interpolation_points, data):
    ip = number_of_interpolation_points
    ch = len(data)
    old_ax = np.linspace(0, 100, ch)
    new_ax = np.linspace(0, 100, ch * ip - (ip - 1))

    data = ma.masked_invalid(data)
    interpolator = interpolate.make_interp_spline(
        old_ax,
        data,
        k=1,
        check_finite=False,
    )
    return interpolator(new_ax)


def _estimate_shift1D(
    data, data_slice=slice(None), ref=None, ip=5, interpolate=True, mask=None, **kwargs
):
    if bool(mask):
        # asarray is required for consistency as argmax
        # returns a numpy scalar array
        return np.asarray(np.nan)
    data = data[data_slice]
    if interpolate is True:
        data = interpolate1D(ip, data)
    # Normalise the data before the cross correlation
    ref = ref - ref.mean()
    data = data - data.mean()
    return (np.argmax(np.correlate(ref, data, "full")) - len(ref) + 1).astype(float)


def _shift1D(data, **kwargs):
    """Used to shift a data array by a specified amount in axes units. Axis must
    be passed as a kwarg."""
    shift = kwargs.get("shift", 0.0)
    original_axis = kwargs.get("original_axis", None)

    # shift.size for cases where shift is empty
    if shift.size == 0 or np.isnan(shift) or shift == 0:
        return data

    data = ma.masked_invalid(data)
    # #This is the interpolant function
    si = interpolate.make_interp_spline(original_axis, data, k=1, check_finite=False)

    # Evaluate interpolated data at shifted positions
    return si(original_axis - shift)


class Signal1D(BaseSignal, CommonSignal1D):
    """General 1D signal class."""

    _signal_dimension = 1

    def __init__(self, *args, **kwargs):
        if kwargs.get("ragged", False):
            raise ValueError("Signal1D can't be ragged.")
        super().__init__(*args, **kwargs)

    def _spikes_diagnosis(
        self,
        signal_mask=None,
        navigation_mask=None,
        show_plot=False,
        use_gui=False,
        **kwargs,
    ):
        self._check_signal_dimension_equals_one()
        dc = self.data
        axis = self.axes_manager.signal_axes[0].axis
        if signal_mask is not None:
            dc = dc[..., ~signal_mask]
            axis = axis[~signal_mask]
        if navigation_mask is not None:
            dc = dc[~navigation_mask, :]
        if dc.size == 0:
            raise ValueError("The data size must be higher than 0.")
        der = abs(np.gradient(dc, axis, axis=-1))
        n = (
            (~navigation_mask).sum()
            if navigation_mask
            else self.axes_manager.navigation_size
        )

        # arbitrary cutoff for number of spectra necessary before histogram
        # data is compressed by finding maxima of each spectrum
        tmp = BaseSignal(der) if n < 2000 else BaseSignal(np.ravel(der.max(-1)))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=Warning,
                message="Estimated number of bins using",
                module="hyperspy",
            )
            s_ = tmp.get_histogram(**kwargs)
        s_.axes_manager[0].name = "Derivative magnitude"
        s_.metadata.Signal.quantity = "Counts"
        s_.metadata.General.title = "Spikes Analysis"

        if s_.data.size == 1:
            message = "The derivative of the data is constant."
            if use_gui:
                m = SimpleMessage(text=message)
                try:
                    m.gui()
                except (NotImplementedError, ImportError):
                    # This is only available for traitsui, in case of ipywidgets
                    # we show a warning
                    warnings.warn(message)
            else:
                warnings.warn(message)
        elif show_plot:
            s_.plot(norm="log")

        return s_

    def spikes_diagnosis(self, signal_mask=None, navigation_mask=None, **kwargs):
        """
        Plots a histogram to help in choosing the threshold for
        spikes removal.

        Parameters
        ----------
        %s
        %s
        **kwargs : dict
            Keyword arguments pass to
            :meth:`~hyperspy.api.signals.BaseSignal.get_histogram`

        See Also
        --------
        spikes_removal_tool

        """
        self._spikes_diagnosis(
            signal_mask=signal_mask,
            navigation_mask=navigation_mask,
            show_plot=True,
            use_gui=False,
            **kwargs,
        )

    spikes_diagnosis.__doc__ %= (SIGNAL_MASK_ARG, NAVIGATION_MASK_ARG)

    def spikes_removal_tool(
        self,
        signal_mask=None,
        navigation_mask=None,
        threshold="auto",
        interactive=True,
        display=True,
        toolkit=None,
        **kwargs,
    ):
        self._check_signal_dimension_equals_one()
        if interactive:
            sr = SpikesRemovalInteractive(
                self,
                signal_mask=signal_mask,
                navigation_mask=navigation_mask,
                threshold=threshold,
            )
            return sr.gui(display=display, toolkit=toolkit)
        else:
            sr = SpikesRemoval(
                self,
                signal_mask=signal_mask,
                navigation_mask=navigation_mask,
                threshold=threshold,
                **kwargs,
            )
            sr.remove_all_spikes()
            return sr

    spikes_removal_tool.__doc__ = SPIKES_REMOVAL_TOOL_DOCSTRING % (
        SIGNAL_MASK_ARG,
        NAVIGATION_MASK_ARG,
        "",
        DISPLAY_DT,
        TOOLKIT_DT,
    )

    def create_model(self, dictionary=None):
        """Create a model for the current data.

        Returns
        -------
        model : `Model1D` instance.

        """

        model = Model1D(self, dictionary=dictionary)
        return model

    def shift1D(
        self,
        shift_array,
        interpolation_method="linear",
        crop=True,
        expand=False,
        fill_value=np.nan,
        show_progressbar=None,
        num_workers=None,
    ):
        """Shift the data in place over the signal axis by the amount specified
        by an array.

        Parameters
        ----------
        shift_array : :class:`~hyperspy.api.signals.BaseSignal` or numpy.ndarray
            An array containing the shifting amount. It must have the same
            ``axes_manager.navigation_shape``
            ``axes_manager._navigation_shape_in_array`` shape.
        interpolation_method : str or int
            Specifies the kind of interpolation as a string (``'linear'``,
            ``'nearest'``, ``'zero'``, ``'slinear'``, ``'quadratic'``,
            ``'cubic'``) or as an integer specifying the order of the spline
            interpolator to use.
        %s
        expand : bool
            If True, the data will be expanded to fit all data after alignment.
            Overrides `crop`.
        fill_value : float
            If crop is False fill the data outside of the original
            interval with the given value where needed.
        %s
        %s

        Raises
        ------
        SignalDimensionError
            If the signal dimension is not 1.
        NotImplementedError
            If the signal axis is a non-uniform axis.
        """
        if not np.any(shift_array):
            # Nothing to do, the shift array if filled with zeros
            return
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_one()
        axis = self.axes_manager.signal_axes[0]

        if not axis.is_uniform:
            raise NotImplementedError(
                "This operation is not implemented for non-uniform axes."
            )

        # Figure out min/max shifts, and translate to shifts in index as well
        minimum, maximum = np.nanmin(shift_array), np.nanmax(shift_array)
        if minimum < 0:
            ihigh = 1 + axis.value2index(axis.high_value + minimum, rounding=math.floor)
        else:
            ihigh = axis.high_index + 1
        if maximum > 0:
            ilow = axis.value2index(axis.offset + maximum, rounding=math.ceil)
        else:
            ilow = axis.low_index
        if expand:
            if self._lazy:
                ind = axis.index_in_array
                pre_shape = list(self.data.shape)
                post_shape = list(self.data.shape)
                pre_chunks = list(self.data.chunks)
                post_chunks = list(self.data.chunks)

                pre_shape[ind] = axis.high_index - ihigh + 1
                post_shape[ind] = ilow - axis.low_index
                for chunks, shape in zip(
                    (pre_chunks, post_chunks), (pre_shape, post_shape)
                ):
                    maxsize = min(np.max(chunks[ind]), shape[ind])
                    num = np.ceil(shape[ind] / maxsize)
                    chunks[ind] = tuple(
                        len(ar) for ar in np.array_split(np.arange(shape[ind]), num)
                    )
                pre_array = da.full(
                    tuple(pre_shape), fill_value, chunks=tuple(pre_chunks)
                )

                post_array = da.full(
                    tuple(post_shape), fill_value, chunks=tuple(post_chunks)
                )
                self.data = da.concatenate(
                    (pre_array, self.data, post_array), axis=ind
                ).rechunk({ind: -1})
            else:
                padding = []
                for i in range(self.data.ndim):
                    if i == axis.index_in_array:
                        padding.append(
                            (axis.high_index - ihigh + 1, ilow - axis.low_index)
                        )
                    else:
                        padding.append((0, 0))
                self.data = np.pad(
                    self.data, padding, mode="constant", constant_values=(fill_value,)
                )
            axis.offset += minimum
            axis.size += axis.high_index - ihigh + 1 + ilow - axis.low_index
        if isinstance(shift_array, np.ndarray):
            shift_array = BaseSignal(shift_array.squeeze()).T

        self.map(
            _shift1D,
            shift=shift_array,
            original_axis=axis.axis,
            fill_value=fill_value,
            kind=interpolation_method,
            show_progressbar=show_progressbar,
            num_workers=num_workers,
            ragged=False,
        )

        if crop and not expand:
            _logger.debug("Cropping %s from index %i to %i" % (self, ilow, ihigh))
            self.crop(axis.index_in_axes_manager, ilow, ihigh)

        self.events.data_changed.trigger(obj=self)

    shift1D.__doc__ %= (CROP_PARAMETER_DOC, SHOW_PROGRESSBAR_ARG, NUM_WORKERS_ARG)

    def interpolate_in_between(
        self,
        start,
        end,
        delta=3,
        show_progressbar=None,
        num_workers=None,
        **kwargs,
    ):
        """Replace the data in a given range by interpolation.
        The operation is performed in place.

        Parameters
        ----------
        start, end : int or float
            The limits of the interval. If int, they are taken as the
            axis index. If float, they are taken as the axis value.
        delta : int or float
            The windows around the (start, end) to use for interpolation. If
            int, they are taken as index steps. If float, they are taken in
            units of the axis value.
        %s
        %s
        **kwargs : dict
            All extra keyword arguments are passed to
            :class:`scipy.interpolate.interp1d`. See the function documentation
            for details.

        Raises
        ------
        SignalDimensionError
            If the signal dimension is not 1.
        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_one()
        axis = self.axes_manager.signal_axes[0]
        i1 = axis._get_index(start)
        i2 = axis._get_index(end)
        if isinstance(delta, float):
            if isinstance(start, int):
                start = axis.axis[start]
            if isinstance(end, int):
                end = axis.axis[end]
            i0 = axis._get_index(start - delta) if start - delta < axis.low_value else 0
            i3 = (
                axis._get_index(end + delta)
                if end + delta > axis.high_value
                else axis.size
            )
        else:
            i0 = int(np.clip(i1 - delta, 0, np.inf))
            i3 = int(np.clip(i2 + delta, 0, axis.size))

        def interpolating_function(dat):
            dat_int = interpolate.interp1d(
                list(range(i0, i1)) + list(range(i2, i3)),
                dat[i0:i1].tolist() + dat[i2:i3].tolist(),
                **kwargs,
            )
            dat[i1:i2] = dat_int(list(range(i1, i2)))
            return dat

        self.map(
            interpolating_function,
            ragged=False,
            show_progressbar=show_progressbar,
            num_workers=num_workers,
        )
        self.events.data_changed.trigger(obj=self)

    interpolate_in_between.__doc__ %= (SHOW_PROGRESSBAR_ARG, NUM_WORKERS_ARG)

    def estimate_shift1D(
        self,
        start=None,
        end=None,
        reference_indices=None,
        max_shift=None,
        interpolate=True,
        number_of_interpolation_points=5,
        mask=None,
        show_progressbar=None,
        num_workers=None,
    ):
        """Estimate the shifts in the current signal axis using
        cross-correlation.
        This method can only estimate the shift by comparing
        unidimensional features that should not change the position in
        the signal axis. To decrease the memory usage, the time of
        computation and the accuracy of the results it is convenient to
        select the feature of interest providing sensible values for
        ``start`` and ``end``. By default interpolation is used to obtain
        subpixel precision.

        Parameters
        ----------
        start, end : int, float or None, default None
            The limits of the interval. If int they are taken as the
            axis index. If float they are taken as the axis value.
        reference_indices : tuple of int or None, default None
            Defines the coordinates of the spectrum that will be used
            as reference. If None the spectrum at the current
            coordinates is used for this purpose.
        max_shift : int
            "Saturation limit" for the shift.
        interpolate : bool, default True
            If True, interpolation is used to provide sub-pixel
            accuracy.
        number_of_interpolation_points : int
            Number of interpolation points. Warning: making this number
            too big can saturate the memory
        mask : :class:`~.api.signals.BaseSignal` of bool.
            It must have signal_dimension = 0 and navigation_shape equal to the
            current signal. Where mask is True the shift is not computed
            and set to nan.
        %s
        %s

        Returns
        -------
        numpy.ndarray
            An array with the result of the estimation in the axis units.
            Although the computation is performed in batches if the signal is
            lazy, the result is computed in memory because it depends on the
            current state of the axes that could change later on in the workflow.

        Raises
        ------
        SignalDimensionError
            If the signal dimension is not 1.
        NotImplementedError
            If the signal axis is a non-uniform axis.
        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_one()
        ip = number_of_interpolation_points + 1
        axis = self.axes_manager.signal_axes[0]
        if not axis.is_uniform:
            raise NotImplementedError(
                "The function is not implemented for non-uniform signal axes."
            )
        self._check_navigation_mask(mask)
        # we compute for now
        if isinstance(start, da.Array):
            start = start.compute()
        if isinstance(end, da.Array):
            end = end.compute()
        i1, i2 = axis._get_index(start), axis._get_index(end)
        if reference_indices is None:
            reference_indices = self.axes_manager.indices
        ref = self.inav[reference_indices].data[i1:i2]

        if interpolate is True:
            ref = interpolate1D(ip, ref)
        shift_signal = self.map(
            _estimate_shift1D,
            mask=mask,
            data_slice=slice(i1, i2),
            ref=ref,
            ip=ip,
            interpolate=interpolate,
            ragged=False,
            inplace=False,
            show_progressbar=show_progressbar,
            num_workers=num_workers,
        )
        shift_array = shift_signal.data
        if max_shift is not None:
            if interpolate is True:
                max_shift *= ip
            shift_array.clip(-max_shift, max_shift)
        if interpolate is True:
            shift_array = shift_array / ip
        shift_array = shift_array * axis.scale
        if shift_signal._lazy:
            # We must compute right now because otherwise any changes to the
            # axes_manager of the signal later in the workflow may result in
            # a wrong shift_array
            shift_array = shift_array.compute()
        return shift_array

    estimate_shift1D.__doc__ %= (SHOW_PROGRESSBAR_ARG, NUM_WORKERS_ARG)

    def align1D(
        self,
        start=None,
        end=None,
        reference_indices=None,
        max_shift=None,
        interpolate=True,
        number_of_interpolation_points=5,
        interpolation_method="linear",
        crop=True,
        expand=False,
        fill_value=np.nan,
        also_align=None,
        mask=None,
        show_progressbar=None,
        iterpath="serpentine",
    ):
        """Estimate the shifts in the signal axis using
        cross-correlation and use the estimation to align the data in place.
        This method can only estimate the shift by comparing
        unidimensional
        features that should not change the position.

        To decrease memory usage, time of computation and improve
        accuracy it is convenient to select the feature of interest
        setting the ``start`` and ``end`` keywords. By default interpolation is
        used to obtain subpixel precision.

        Parameters
        ----------
        start, end : int, float or None
            The limits of the interval. If int they are taken as the
            axis index. If float they are taken as the axis value.
        reference_indices : tuple of int or None
            Defines the coordinates of the spectrum that will be used
            as reference. If None the spectrum at the current
            coordinates is used for this purpose.
        max_shift : int
            "Saturation limit" for the shift.
        interpolate : bool
            If True, interpolation is used to provide sub-pixel
            accuracy.
        number_of_interpolation_points : int
            Number of interpolation points. Warning: making this number
            too big can saturate the memory
        interpolation_method : str or int
            Specifies the kind of interpolation as a string ('linear',
            'nearest', 'zero', 'slinear', 'quadratic, 'cubic') or as an
            integer specifying the order of the spline interpolator to
            use.
        %s
        expand : bool
            If True, the data will be expanded to fit all data after alignment.
            Overrides ``crop`` argument.
        fill_value : float
            If crop is False fill the data outside of the original
            interval with the given value where needed.
        also_align : list of :class:`~.api.signals.BaseSignal`, None
            A list of BaseSignal instances that has exactly the same
            dimensions as this one and that will be aligned using the shift map
            estimated using the this signal.
        mask : :class:`~.api.signals.BaseSignal` or bool
            It must have signal_dimension = 0 and navigation_shape equal to the
            current signal. Where mask is True the shift is not computed
            and set to nan.
        %s

        Returns
        -------
        numpy.ndarray
            The result of the estimation.

        Raises
        ------
        SignalDimensionError
            If the signal dimension is not 1.

        See Also
        --------
        estimate_shift1D
        """
        if also_align is None:
            also_align = []
        self._check_signal_dimension_equals_one()
        if self._lazy:
            _logger.warning(
                "In order to properly expand, the lazy "
                "reference signal will be read twice (once to "
                "estimate shifts, and second time to shift "
                "appropriately), which might take a long time. "
                "Use expand=False to only pass through the data "
                "once."
            )
        with self.axes_manager.switch_iterpath(iterpath):
            shift_array = self.estimate_shift1D(
                start=start,
                end=end,
                reference_indices=reference_indices,
                max_shift=max_shift,
                interpolate=interpolate,
                number_of_interpolation_points=number_of_interpolation_points,
                mask=mask,
                show_progressbar=show_progressbar,
            )
            signals_to_shift = [self] + also_align
            for signal in signals_to_shift:
                signal.shift1D(
                    shift_array=shift_array,
                    interpolation_method=interpolation_method,
                    crop=crop,
                    fill_value=fill_value,
                    expand=expand,
                    show_progressbar=show_progressbar,
                )

    align1D.__doc__ %= (CROP_PARAMETER_DOC, SHOW_PROGRESSBAR_ARG)

    def calibrate(self, display=True, toolkit=None):
        """
        Calibrate the spectral dimension using a gui.
        It displays a window where the new calibration can be set by:

        * setting the values of offset, units and scale directly
        * or selecting a range by dragging the mouse on the spectrum figure
          and setting the new values for the given range limits

        Parameters
        ----------
        %s
        %s

        Notes
        -----
        For this method to work the output_dimension must be 1.

        Raises
        ------
        SignalDimensionError
            If the signal dimension is not 1.
        NotImplementedError
            If called with a non-uniform axes.
        """
        self._check_signal_dimension_equals_one()
        calibration = Signal1DCalibration(self)
        return calibration.gui(display=display, toolkit=toolkit)

    calibrate.__doc__ %= (DISPLAY_DT, TOOLKIT_DT)

    def smooth_savitzky_golay(
        self,
        polynomial_order=None,
        window_length=None,
        differential_order=0,
        num_workers=None,
        display=True,
        toolkit=None,
    ):
        """
        Apply a Savitzky-Golay filter to the data in place.
        If `polynomial_order` or `window_length` or `differential_order` are
        None the method is run in interactive mode.

        Parameters
        ----------
        polynomial_order : int, optional
            The order of the polynomial used to fit the samples.
            `polyorder` must be less than `window_length`.
        window_length : int, optional
            The length of the filter window (i.e. the number of coefficients).
            `window_length` must be a positive odd integer.
        differential_order: int, optional
            The order of the derivative to compute.  This must be a
            nonnegative integer.  The default is 0, which means to filter
            the data without differentiating.
        %s
        %s
        %s

        Raises
        ------
        NotImplementedError
            If the signal axis is a non-uniform axis.

        Notes
        -----
        More information about the filter in `scipy.signal.savgol_filter`.
        """
        self._check_signal_dimension_equals_one()
        if not self.axes_manager.signal_axes[0].is_uniform:
            raise NotImplementedError(
                "This functionality is not implemented for signals with non-uniform axes. "
                "Consider using `smooth_lowess` instead."
            )
        if polynomial_order is not None and window_length is not None:
            axis = self.axes_manager.signal_axes[0]
            self.map(
                savgol_filter,
                window_length=window_length,
                polyorder=polynomial_order,
                deriv=differential_order,
                delta=axis.scale,
                ragged=False,
                num_workers=num_workers,
            )
        else:
            # Interactive mode
            smoother = SmoothingSavitzkyGolay(self)
            smoother.differential_order = differential_order
            if polynomial_order is not None:
                smoother.polynomial_order = polynomial_order
            if window_length is not None:
                smoother.window_length = window_length
            return smoother.gui(display=display, toolkit=toolkit)

    smooth_savitzky_golay.__doc__ %= (NUM_WORKERS_ARG, DISPLAY_DT, TOOLKIT_DT)

    def smooth_lowess(
        self,
        smoothing_parameter=None,
        number_of_iterations=None,
        show_progressbar=None,
        num_workers=None,
        display=True,
        toolkit=None,
    ):
        """
        Lowess data smoothing in place.
        If `smoothing_parameter` or `number_of_iterations` are None the method
        is run in interactive mode.

        Parameters
        ----------
        smoothing_parameter: float or None
            Between 0 and 1. The fraction of the data used
            when estimating each y-value.
        number_of_iterations: int or None
            The number of residual-based reweightings
            to perform.
        %s
        %s
        %s
        %s

        Raises
        ------
        SignalDimensionError
            If the signal dimension is not 1.

        """
        self._check_signal_dimension_equals_one()
        if smoothing_parameter is None or number_of_iterations is None:
            smoother = SmoothingLowess(self)
            if smoothing_parameter is not None:
                smoother.smoothing_parameter = smoothing_parameter
            if number_of_iterations is not None:
                smoother.number_of_iterations = number_of_iterations
            return smoother.gui(display=display, toolkit=toolkit)
        else:
            self.map(
                lowess,
                x=self.axes_manager[-1].axis,
                f=smoothing_parameter,
                n_iter=number_of_iterations,
                show_progressbar=show_progressbar,
                ragged=False,
                num_workers=num_workers,
                silence_warnings="non-uniform",
            )

    smooth_lowess.__doc__ %= (
        SHOW_PROGRESSBAR_ARG,
        NUM_WORKERS_ARG,
        DISPLAY_DT,
        TOOLKIT_DT,
    )

    def smooth_tv(
        self,
        smoothing_parameter=None,
        show_progressbar=None,
        num_workers=None,
        display=True,
        toolkit=None,
    ):
        """
        Total variation data smoothing in place.

        Parameters
        ----------
        smoothing_parameter: float or None
           Denoising weight relative to L2 minimization. If None the method
           is run in interactive mode.
        %s
        %s
        %s
        %s

        Raises
        ------
        SignalDimensionError
            If the signal dimension is not 1.
        NotImplementedError
            If the signal axis is a non-uniform axis.
        """
        self._check_signal_dimension_equals_one()
        if not self.axes_manager.signal_axes[0].is_uniform:
            raise NotImplementedError(
                "This functionality is not implemented for signals with non-uniform axes. "
                "Consider using `smooth_lowess` instead."
            )
        if smoothing_parameter is None:
            smoother = SmoothingTV(self)
            return smoother.gui(display=display, toolkit=toolkit)
        else:
            self.map(
                _tv_denoise_1d,
                weight=smoothing_parameter,
                ragged=False,
                show_progressbar=show_progressbar,
                num_workers=num_workers,
            )

    smooth_tv.__doc__ %= (SHOW_PROGRESSBAR_ARG, NUM_WORKERS_ARG, DISPLAY_DT, TOOLKIT_DT)

    def filter_butterworth(
        self,
        cutoff_frequency_ratio=None,
        type="low",
        order=2,
        display=True,
        toolkit=None,
    ):
        """
        Butterworth filter in place.

        Parameters
        ----------
        %s
        %s

        Raises
        ------
        SignalDimensionError
            If the signal dimension is not 1.
        NotImplementedError
            If the signal axis is a non-uniform axis.
        """
        if not self.axes_manager.signal_axes[0].is_uniform:
            raise NotImplementedError(
                "This functionality is not implemented for signals with non-uniform axes. "
                "Consider using `smooth_lowess` instead."
            )
        self._check_signal_dimension_equals_one()
        smoother = ButterworthFilter(self)
        if cutoff_frequency_ratio is not None:
            smoother.cutoff_frequency_ratio = cutoff_frequency_ratio
            smoother.type = type
            smoother.order = order
            smoother.apply()
        else:
            return smoother.gui(display=display, toolkit=toolkit)

    filter_butterworth.__doc__ %= (DISPLAY_DT, TOOLKIT_DT)

    def _remove_background_cli(
        self,
        signal_range,
        background_estimator,
        fast=True,
        zero_fill=False,
        show_progressbar=None,
        model=None,
        return_model=False,
    ):
        """See :meth:`~hyperspy._signal1d.signal1D.remove_background`."""
        if model is None:
            from hyperspy.models.model1d import Model1D

            model = Model1D(self)
        if background_estimator not in model:
            model.append(background_estimator)
        background_estimator.estimate_parameters(
            self, signal_range[0], signal_range[1], only_current=False
        )

        if not fast:
            model.set_signal_range(signal_range[0], signal_range[1])
            model.multifit(show_progressbar=show_progressbar, iterpath="serpentine")
            model.reset_signal_range()

        if self._lazy:
            result = self - model.as_signal(show_progressbar=show_progressbar)
        else:
            try:
                axis = self.axes_manager.signal_axes[0]
                if axis.is_binned:
                    if axis.is_uniform:
                        scale_factor = axis.scale
                    else:
                        scale_factor = np.gradient(axis.axis)
                else:
                    scale_factor = 1
                bkg = background_estimator.function_nd(axis.axis) * scale_factor
                result = self - bkg
            except MemoryError:
                result = self - model.as_signal(show_progressbar=show_progressbar)

        if zero_fill:
            if self._lazy:
                low_idx = result.axes_manager[-1].value2index(signal_range[0])
                z = da.zeros(low_idx, chunks=(low_idx,))
                cropped_da = result.data[low_idx:]
                result.data = da.concatenate([z, cropped_da])
            else:
                result.isig[: signal_range[0]] = 0
        if return_model:
            if fast:
                # Calculate the variance for each navigation position only when
                # using fast, otherwise the chisq is already calculated when
                # doing the multifit
                d = result.data[..., np.where(model._channel_switches)[0]]
                variance = model._get_variance(only_current=False)
                d *= d / (1.0 * variance)  # d = difference^2 / variance.
                model.chisq.data = d.sum(-1)
            result = (result, model)
        return result

    def remove_background(
        self,
        signal_range="interactive",
        background_type="Power law",
        polynomial_order=2,
        fast=True,
        zero_fill=False,
        plot_remainder=True,
        show_progressbar=None,
        return_model=False,
        display=True,
        toolkit=None,
    ):
        """
        Remove the background, either in place using a GUI or returned as a new
        spectrum using the command line. The fast option is not accurate for
        most background types - except Gaussian, Offset and
        Power law - but it is useful to estimate the initial fitting parameters
        before performing a full fit.

        Parameters
        ----------
        signal_range : "interactive", tuple of int or float, optional
            If this argument is not specified, the signal range has to be
            selected using a GUI. And the original spectrum will be replaced.
            If tuple is given, a spectrum will be returned.
        background_type : str
            The type of component which should be used to fit the background.
            Possible components: Doniach, Gaussian, Lorentzian, Offset,
            Polynomial, PowerLaw, Exponential, SkewNormal, SplitVoigt, Voigt.
            If Polynomial is used, the polynomial order can be specified
        polynomial_order : int, default 2
            Specify the polynomial order if a Polynomial background is used.
        fast : bool
            If True, perform an approximative estimation of the parameters.
            If False, the signal is fitted using non-linear least squares
            afterwards. This is slower compared to the estimation but
            often more accurate.
        zero_fill : bool
            If True, all spectral channels lower than the lower bound of the
            fitting range will be set to zero (this is the default behavior
            of Gatan's DigitalMicrograph). Setting this value to False
            allows for inspection of the quality of background fit throughout
            the pre-fitting region.
        plot_remainder : bool
            If True, add a (green) line previewing the remainder signal after
            background removal. This preview is obtained from a Fast calculation
            so the result may be different if a NLLS calculation is finally
            performed.
        return_model : bool
            If True, the background model is returned. The chi² can be obtained
            from this model using
            :meth:`~hyperspy.model.BaseModel.chisq`.
        %s
        %s
        %s

        Returns
        -------
        None or (:class:`hyperspy.api.signals.BaseSignal`, :class:`hyperspy.models.model1d.Model1D`)
            If ``signal_range`` is not ``'interactive'``, the signal with background
            subtracted is returned. If ``return_model=True``, returns the
            background model, otherwise, the GUI widget dictionary is returned
            if ``display=False`` - see the display parameter documentation.

        Examples
        --------
        Using GUI, replaces spectrum s

        >>> s = hs.signals.Signal1D(range(1000))
        >>> s.remove_background() #doctest: +SKIP

        Using command line, returns a Signal1D:

        >>> s.remove_background(
        ...    signal_range=(400,450), background_type='PowerLaw'
        ... )
        <Signal1D, title: , dimensions: (|1000)>

        Using a full model to fit the background:

        >>> s.remove_background(signal_range=(400,450), fast=False)
        <Signal1D, title: , dimensions: (|1000)>

        Returns background subtracted and the model:

        >>> s.remove_background(
        ...    signal_range=(400,450), fast=False, return_model=True
        ... )
        (<Signal1D, title: , dimensions: (|1000)>, <Model1D>)

        Raises
        ------
        SignalDimensionError
            If the signal dimension is not 1.
        """

        self._check_signal_dimension_equals_one()
        # Create model here, so that we can return it
        from hyperspy.models.model1d import Model1D

        model = Model1D(self)
        if signal_range == "interactive":
            br = BackgroundRemoval(
                self,
                background_type=background_type,
                polynomial_order=polynomial_order,
                fast=fast,
                plot_remainder=plot_remainder,
                show_progressbar=show_progressbar,
                zero_fill=zero_fill,
                model=model,
            )
            gui_dict = br.gui(display=display, toolkit=toolkit)
            if return_model:
                return model
            else:
                # for testing purposes
                return gui_dict
        else:
            background_estimator = _get_background_estimator(
                background_type, polynomial_order
            )[0]
            result = self._remove_background_cli(
                signal_range=signal_range,
                background_estimator=background_estimator,
                fast=fast,
                zero_fill=zero_fill,
                show_progressbar=show_progressbar,
                model=model,
                return_model=return_model,
            )
            return result

    remove_background.__doc__ %= (SHOW_PROGRESSBAR_ARG, DISPLAY_DT, TOOLKIT_DT)

    def remove_baseline(
        self,
        method=None,
        inplace=True,
        display=True,
        toolkit=None,
        **kwargs,
    ):
        """
        Remove baselines using algorithms implemented in pybaselines.

        Parameters
        ----------
        method : str or None
            If ``str``, any of the algorithm name in :class:`pybaselines.api.Baseline`.
            If ``None``, a widget is open to select an algorithm and adjust
            the parameters.
        %s
        %s
        %s
        **kwargs : dict
            Keyword arguments of baseline algorithm. These are passed
            to baseline function.

        Examples
        --------
        import hyperspy.api as hs
        s = hs.data.two_gaussians()

        s.remove_baselines(method="aspls", lam=1E7)

        """
        if method is None:
            from hyperspy.utils.baseline_removal_tool import BaselineRemoval

            br = BaselineRemoval(self, **kwargs)
            return br.gui(display=display, toolkit=toolkit)
        else:
            from pybaselines import Baseline

            baseline_fitter = getattr(
                Baseline(
                    self.axes_manager[-1].axis,
                    check_finite=False,
                ),
                method,
            )

            def baseline_fitting(data, **kwargs):
                return data - baseline_fitter(data, **kwargs)[0]

            kwargs["silence_warnings"] = "non-uniform"
            return self.map(baseline_fitting, inplace=inplace, **kwargs)

    remove_baseline.__doc__ %= (IN_PLACE, DISPLAY_DT, TOOLKIT_DT)

    @interactive_range_selector
    def crop_signal(
        self,
        left_value=None,
        right_value=None,
    ):
        """Crop in place in the signal space.

        Parameters
        ----------
        left_value, right_value : int, float or None
            If int the values are taken as indices. If float they are
            converted to indices using the spectral axis calibration.
            If left_value is None crops from the beginning of the axis.
            If right_value is None crops up to the end of the axis. If
            both are None the interactive cropping interface is activated
            enabling cropping the spectrum using a span selector in the
            signal plot.

        Raises
        ------
        SignalDimensionError
            If the signal dimension is not 1.
        """
        self._check_signal_dimension_equals_one()
        try:
            left_value, right_value = left_value
        except TypeError:
            # It was not a ROI, we carry on
            pass
        self.crop(
            axis=self.axes_manager.signal_axes[0].index_in_axes_manager,
            start=left_value,
            end=right_value,
        )

    def gaussian_filter(self, FWHM):
        """Applies a Gaussian filter in the spectral dimension in place.

        Parameters
        ----------
        FWHM : float
            The Full Width at Half Maximum of the gaussian in the
            spectral axis units

        Raises
        ------
        ValueError
            If FWHM is equal or less than zero.
        SignalDimensionError
            If the signal dimension is not 1.
        NotImplementedError
            If the signal axis is a non-uniform axis.
        """
        self._check_signal_dimension_equals_one()
        for _axis in self.axes_manager.signal_axes:
            if not _axis.is_uniform:
                raise NotImplementedError(
                    "The function is not implemented for non-uniform axes."
                )
        if FWHM <= 0:
            raise ValueError("FWHM must be greater than zero")
        axis = self.axes_manager.signal_axes[0]
        FWHM *= 1 / axis.scale
        self.map(gaussian_filter1d, sigma=FWHM / 2.35482, ragged=False)

    def hanning_taper(self, side="both", channels=None, offset=0):
        """Apply a hanning taper to the data in place.

        Parameters
        ----------
        side : { ``'left'`` | ``'right'`` | ``'both'`` }
            Specify which side to use.
        channels : None or int
            The number of channels to taper. If None 5% of the total
            number of channels are tapered.
        offset : int

        Returns
        -------
        int

        Raises
        ------
        SignalDimensionError
            If the signal dimension is not 1.
        """
        if not np.issubdtype(self.data.dtype, np.floating):
            raise TypeError(
                "The data dtype should be `float`. It can be "
                "changed by using the `change_dtype('float')` "
                "method of the signal."
            )

        # TODO: generalize it
        self._check_signal_dimension_equals_one()
        if channels is None:
            channels = int(round(len(self._get_current_data()) * 0.02))
            if channels < 20:
                channels = 20
        dc = self._data_aligned_with_axes
        if self._lazy and offset != 0:
            shp = dc.shape
            if len(shp) == 1:
                nav_shape = ()
                nav_chunks = ()
            else:
                nav_shape = shp[:-1]
                nav_chunks = dc.chunks[:-1]
            zeros = da.zeros(nav_shape + (offset,), chunks=nav_chunks + ((offset,),))

        if side == "left" or side == "both":
            if self._lazy:
                tapered = dc[..., offset : channels + offset]
                tapered *= np.hanning(2 * channels)[:channels]
                therest = dc[..., channels + offset :]
                thelist = [] if offset == 0 else [zeros]
                thelist.extend([tapered, therest])
                dc = da.concatenate(thelist, axis=-1)
            else:
                dc[..., offset : channels + offset] *= np.hanning(2 * channels)[
                    :channels
                ]
                dc[..., :offset] *= 0.0
        if side == "right" or side == "both":
            rl = None if offset == 0 else -offset
            if self._lazy:
                therest = dc[..., : -channels - offset]
                tapered = dc[..., -channels - offset : rl]
                tapered *= np.hanning(2 * channels)[-channels:]
                thelist = [therest, tapered]
                if offset != 0:
                    thelist.append(zeros)
                dc = da.concatenate(thelist, axis=-1)
            else:
                dc[..., -channels - offset : rl] *= np.hanning(2 * channels)[-channels:]
                if offset != 0:
                    dc[..., -offset:] *= 0.0

        if self._lazy:
            self.data = dc
        self.events.data_changed.trigger(obj=self)
        return channels

    def find_peaks1D_ohaver(
        self,
        xdim=None,
        slope_thresh=0,
        amp_thresh=None,
        subchannel=True,
        medfilt_radius=5,
        maxpeakn=30000,
        peakgroup=10,
        num_workers=None,
    ):
        """Find positive peaks along a 1D Signal. It detects peaks by looking
        for downward zero-crossings in the first derivative that exceed
        ``slope_thresh``.

        ``slope_thresh`` and ``amp_thresh``, control sensitivity: higher
        values will neglect broad peaks (slope) and smaller features (amp),
        respectively.

        Parameters
        ----------
        slope_thresh : float, default 0
            1st derivative threshold to count the peak;
            higher values will neglect broader features.
        amp_thresh : float, optional
            Intensity threshold below which peaks are ignored;
            higher values will neglect smaller features;
            default is set to 10%% of max(y).
        medfilt_radius : int, default 5
            Median filter window to apply to smooth the data
            (see :func:`scipy.signal.medfilt`);
            if 0, no filter will be applied.
        peakgroup : int, default 10
            Number of points around the "top part" of the peak
            that is taken to estimate the peak height.
            For spikes or very narrow peaks, set `peakgroup` to 1 or 2;
            for broad or noisy peaks, make ``peakgroup`` larger to
            reduce the effect of noise.
        maxpeakn : int, default 5000
            Number of maximum detectable peaks.
        subchannel : bool, default True
            Whether to use subchannel precision or not.
        %s

        Returns
        -------
        numpy.ndarray
            Structured array of shape (npeaks) containing fields:
            'position', 'width', and 'height' for each peak.

        Raises
        ------
        SignalDimensionError
            If the signal dimension is not 1.
        """
        # TODO: add scipy.signal.find_peaks_cwt
        self._check_signal_dimension_equals_one()
        axis = self.axes_manager.signal_axes[0].axis
        peaks = self.map(
            find_peaks_ohaver,
            x=axis,
            slope_thresh=slope_thresh,
            amp_thresh=amp_thresh,
            medfilt_radius=medfilt_radius,
            maxpeakn=maxpeakn,
            peakgroup=peakgroup,
            subchannel=subchannel,
            ragged=True,
            output_dtype=object,
            output_signal_size=(),
            num_workers=num_workers,
            inplace=False,
            lazy_output=False,
            silence_warnings="non-uniform",
        )
        return peaks.data

    find_peaks1D_ohaver.__doc__ %= NUM_WORKERS_ARG

    def estimate_peak_width(
        self,
        factor=0.5,
        window=None,
        return_interval=False,
        show_progressbar=None,
        num_workers=None,
    ):
        """Estimate the width of the highest intensity of peak
        of the spectra at a given fraction of its maximum.

        It can be used with asymmetric peaks. For accurate results any
        background must be previously subtracted.
        The estimation is performed by interpolation using cubic splines.

        Parameters
        ----------
        factor : float, default 0.5
            Normalized height (in interval [0, 1]) at which to estimate the
            width. The default (0.5) estimates the FWHM.
        window : None or float
            The size of the window centred at the peak maximum
            used to perform the estimation.
            The window size must be chosen with care: if it is narrower
            than the width of the peak at some positions or if it is
            so wide that it includes other more intense peaks this
            method cannot compute the width and a NaN is stored instead.
        return_interval : bool
            If True, returns 2 extra signals with the positions of the
            desired height fraction at the left and right of the
            peak.
        %s
        %s

        Returns
        -------
        float or list of float
            width or [width, left, right], depending on the value of
            `return_interval`.

        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_one()
        if not 0 < factor < 1:
            raise ValueError("factor must be between 0 and 1.")

        # axis is a keyword already used by self.map so calling this axis_arg
        # to avoid "parameter collision
        axis_arg = self.axes_manager.signal_axes[0]
        maxval = self.axes_manager.navigation_size
        show_progressbar = show_progressbar and maxval > 0

        def estimating_function(spectrum, window=None, factor=0.5, axis_arg=None):
            x = axis_arg.axis
            if window is not None:
                vmax = axis_arg.index2value(spectrum.argmax())
                slices = axis_arg._get_array_slices(
                    slice(vmax - window * 0.5, vmax + window * 0.5)
                )
                spectrum = spectrum[slices]
                x = x[slices]
            spline = interpolate.UnivariateSpline(
                x, spectrum - factor * spectrum.max(), s=0
            )
            roots = spline.roots()
            if len(roots) == 2:
                return np.array(roots)
            else:
                return np.full((2,), np.nan)

        both = self.map(
            estimating_function,
            window=window,
            factor=factor,
            axis_arg=axis_arg,
            ragged=False,
            inplace=False,
            show_progressbar=show_progressbar,
            num_workers=num_workers,
        )
        left, right = both.T.split()
        width = right - left
        if factor == 0.5:
            width.metadata.General.title = self.metadata.General.title + " FWHM"
            left.metadata.General.title = (
                self.metadata.General.title + " FWHM left position"
            )

            right.metadata.General.title = (
                self.metadata.General.title + " FWHM right position"
            )
        else:
            width.metadata.General.title = (
                self.metadata.General.title + " full-width at %.1f maximum" % factor
            )

            left.metadata.General.title = (
                self.metadata.General.title
                + " full-width at %.1f maximum left position" % factor
            )
            right.metadata.General.title = (
                self.metadata.General.title
                + " full-width at %.1f maximum right position" % factor
            )
        for signal in (left, width, right):
            signal = signal.transpose(signal_axes=[])
            signal.set_signal_type("")
        if return_interval is True:
            return [width, left, right]
        else:
            return width

    estimate_peak_width.__doc__ %= (SHOW_PROGRESSBAR_ARG, NUM_WORKERS_ARG)

    def plot(
        self,
        navigator="auto",
        plot_markers=True,
        autoscale="v",
        norm="auto",
        axes_manager=None,
        navigator_kwds={},
        **kwargs,
    ):
        """%s
        %s
        %s
        """
        for c in autoscale:
            if c not in ["x", "v"]:
                raise ValueError(
                    "`autoscale` only accepts 'x', 'v' as " "valid characters."
                )
        super().plot(
            navigator=navigator,
            plot_markers=plot_markers,
            autoscale=autoscale,
            norm=norm,
            axes_manager=axes_manager,
            navigator_kwds=navigator_kwds,
            **kwargs,
        )

    plot.__doc__ %= (
        BASE_PLOT_DOCSTRING,
        BASE_PLOT_DOCSTRING_PARAMETERS,
        PLOT1D_DOCSTRING,
    )


class LazySignal1D(LazySignal, Signal1D):
    """Lazy general 1D signal class."""

    __doc__ += LAZYSIGNAL_DOC.replace("__BASECLASS__", "Signal1D")
