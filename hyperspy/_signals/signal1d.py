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
import math

import matplotlib.pyplot as plt
import numpy as np
import dask.array as da
import scipy.interpolate
import scipy as sp
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter1d
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    statsmodels_installed = True
except BaseException:
    statsmodels_installed = False

from hyperspy.signal import BaseSignal
from hyperspy._signals.common_signal1d import CommonSignal1D
from hyperspy.signal_tools import SpikesRemoval
from hyperspy.models.model1d import Model1D


from hyperspy.misc.utils import signal_range_from_roi
from hyperspy.defaults_parser import preferences
from hyperspy.signal_tools import (
    Signal1DCalibration,
    SmoothingSavitzkyGolay,
    SmoothingLowess,
    SmoothingTV,
    ButterworthFilter)
from hyperspy.ui_registry import DISPLAY_DT, TOOLKIT_DT
from hyperspy.misc.tv_denoise import _tv_denoise_1d
from hyperspy.signal_tools import BackgroundRemoval
from hyperspy.decorators import interactive_range_selector
from hyperspy.signal_tools import IntegrateArea
from hyperspy import components1d
from hyperspy._signals.lazy import LazySignal
from hyperspy.docstrings.signal1d import CROP_PARAMETER_DOC

_logger = logging.getLogger(__name__)


def find_peaks_ohaver(y, x=None, slope_thresh=0., amp_thresh=None,
                      medfilt_radius=5, maxpeakn=30000, peakgroup=10,
                      subchannel=True,):
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
        d = np.gradient(scipy.signal.medfilt(y, medfilt_radius))
    else:
        d = np.gradient(y)
    n = np.round(peakgroup / 2 + 1)
    peak_dt = np.dtype([('position', np.float),
                        ('height', np.float),
                        ('width', np.float)])
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
                                xx = xx[:groupindex - 1]
                                yy = yy[:groupindex - 1]
                                break
                            xx[k - s] = x[groupindex]
                            yy[k - s] = y[groupindex]
                        avg = np.average(xx)
                        stdev = np.std(xx)
                        xxf = (xx - avg) / stdev
                        # Fit parabola to log10 of sub-group with
                        # centering and scaling
                        yynz = yy != 0
                        coef = np.polyfit(
                            xxf[yynz], np.log10(np.abs(yy[yynz])), 2)
                        c1 = coef[2]
                        c2 = coef[1]
                        c3 = coef[0]
                        with np.errstate(invalid='ignore'):
                            width = np.linalg.norm(stdev * 2.35703 /
                                                   (np.sqrt(2) * np.sqrt(-1 *
                                                                         c3)))
                        # if the peak is too narrow for least-squares
                        # technique to work  well, just use the max value
                        # of y in the sub-group of points near peak.
                        if peakgroup < 7:
                            height = np.max(yy)
                            position = xx[np.argmin(np.abs(yy - height))]
                        else:
                            position = - ((stdev * c2 / (2 * c3)) - avg)
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
                    if (not np.isnan(position) and 0 < position < x[-1]):
                        P = np.hstack((P,
                                       np.array([(position, height, width)],
                                                dtype=peak_dt)))
                        peak += 1
    # return only the part of the array that contains peaks
    # (not the whole maxpeakn x 3 array)
    if len(P) > maxpeakn:
        minh = np.sort(P['height'])[-maxpeakn]
        P = P[P['height'] >= minh]

    # Sorts the values as a function of position
    P.sort(0)

    return P


def interpolate1D(number_of_interpolation_points, data):
    ip = number_of_interpolation_points
    ch = len(data)
    old_ax = np.linspace(0, 100, ch)
    new_ax = np.linspace(0, 100, ch * ip - (ip - 1))
    interpolator = scipy.interpolate.interp1d(old_ax, data)
    return interpolator(new_ax)


def _estimate_shift1D(data, **kwargs):
    mask = kwargs.get('mask', None)
    ref = kwargs.get('ref', None)
    interpolate = kwargs.get('interpolate', True)
    ip = kwargs.get('ip', 5)
    data_slice = kwargs.get('data_slice', slice(None))
    if bool(mask):
        # asarray is required for consistensy as argmax
        # returns a numpy scalar array
        return np.asarray(np.nan)
    data = data[data_slice]
    if interpolate is True:
        data = interpolate1D(ip, data)
    return np.argmax(np.correlate(ref, data, 'full')) - len(ref) + 1


def _shift1D(data, **kwargs):
    shift = kwargs.get('shift', 0.)
    original_axis = kwargs.get('original_axis', None)
    fill_value = kwargs.get('fill_value', np.nan)
    kind = kwargs.get('kind', 'linear')
    offset = kwargs.get('offset', 0.)
    scale = kwargs.get('scale', 1.)
    size = kwargs.get('size', 2)
    if np.isnan(shift) or shift == 0:
        return data
    axis = np.linspace(offset, offset + scale * (size - 1), size)

    si = sp.interpolate.interp1d(original_axis,
                                 data,
                                 bounds_error=False,
                                 fill_value=fill_value,
                                 kind=kind)
    offset = float(offset - shift)
    axis = np.linspace(offset, offset + scale * (size - 1), size)
    return si(axis)


class Signal1D(BaseSignal, CommonSignal1D):

    """
    """
    _signal_dimension = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.axes_manager.signal_dimension != 1:
            self.axes_manager.set_signal_dimension(1)

    def _spikes_diagnosis(self, signal_mask=None,
                          navigation_mask=None):
        """Plots a histogram to help in choosing the threshold for
        spikes removal.

        Parameters
        ----------
        signal_mask: boolean array
            Restricts the operation to the signal locations not marked
            as True (masked)
        navigation_mask: boolean array
            Restricts the operation to the navigation locations not
            marked as True (masked).

        See also
        --------
        spikes_removal_tool

        """
        self._check_signal_dimension_equals_one()
        dc = self.data
        if signal_mask is not None:
            dc = dc[..., ~signal_mask]
        if navigation_mask is not None:
            dc = dc[~navigation_mask, :]
        der = np.abs(np.diff(dc, 1, -1))
        n = ((~navigation_mask).sum() if navigation_mask else
             self.axes_manager.navigation_size)

        # arbitrary cutoff for number of spectra necessary before histogram
        # data is compressed by finding maxima of each spectrum
        tmp = BaseSignal(der) if n < 2000 else BaseSignal(
            np.ravel(der.max(-1)))

        # get histogram signal using smart binning and plot
        tmph = tmp.get_histogram()
        tmph.plot()

        # Customize plot appearance
        plt.gca().set_title('')
        plt.gca().fill_between(tmph.axes_manager[0].axis,
                               tmph.data,
                               facecolor='#fddbc7',
                               interpolate=True,
                               color='none')
        ax = tmph._plot.signal_plot.ax
        axl = tmph._plot.signal_plot.ax_lines[0]
        axl.set_line_properties(color='#b2182b')
        plt.xlabel('Derivative magnitude')
        plt.ylabel('Log(Counts)')
        ax.set_yscale('log')
        ax.set_ylim(10 ** -1, plt.ylim()[1])
        ax.set_xlim(plt.xlim()[0], 1.1 * plt.xlim()[1])
        plt.draw()

    def spikes_removal_tool(self, signal_mask=None,
                            navigation_mask=None, display=True, toolkit=None):
        """Graphical interface to remove spikes from EELS spectra.

        Parameters
        ----------
        signal_mask: boolean array
            Restricts the operation to the signal locations not marked
            as True (masked)
        navigation_mask: boolean array
            Restricts the operation to the navigation locations not
            marked as True (masked)

        See also
        --------
        _spikes_diagnosis,

        """
        self._check_signal_dimension_equals_one()
        sr = SpikesRemoval(self,
                           navigation_mask=navigation_mask,
                           signal_mask=signal_mask)
        return sr.gui(display=display, toolkit=toolkit)
    spikes_removal_tool.__doc__ =\
        """Graphical interface to remove spikes from EELS spectra.

Parameters
----------
signal_mask: boolean array
    Restricts the operation to the signal locations not marked
    as True (masked)
navigation_mask: boolean array
    Restricts the operation to the navigation locations not
    marked as True (masked)
%s
%s
See also
--------
_spikes_diagnosis,

""" % (DISPLAY_DT, TOOLKIT_DT)

    def create_model(self, dictionary=None):
        """Create a model for the current data.

        Returns
        -------
        model : `Model1D` instance.

        """

        model = Model1D(self, dictionary=dictionary)
        return model

    def shift1D(self,
                shift_array,
                interpolation_method='linear',
                crop=True,
                expand=False,
                fill_value=np.nan,
                parallel=None,
                show_progressbar=None):
        """Shift the data in place over the signal axis by the amount specified
        by an array.

        Parameters
        ----------
        shift_array : numpy array
            An array containing the shifting amount. It must have
            `axes_manager._navigation_shape_in_array` shape.
        interpolation_method : str or int
            Specifies the kind of interpolation as a string ('linear',
            'nearest', 'zero', 'slinear', 'quadratic, 'cubic') or as an
            integer specifying the order of the spline interpolator to
            use.
        %s
        expand : bool
            If True, the data will be expanded to fit all data after alignment.
            Overrides `crop`.
        fill_value : float
            If crop is False fill the data outside of the original
            interval with the given value where needed.
        parallel : {None, bool}
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.
        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """
        if not np.any(shift_array):
            # Nothing to do, the shift array if filled with zeros
            return
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_one()
        axis = self.axes_manager.signal_axes[0]

        # Figure out min/max shifts, and translate to shifts in index as well
        minimum, maximum = np.nanmin(shift_array), np.nanmax(shift_array)
        if minimum < 0:
            ihigh = 1 + axis.value2index(
                axis.high_value + minimum,
                rounding=math.floor)
        else:
            ihigh = axis.high_index + 1
        if maximum > 0:
            ilow = axis.value2index(axis.offset + maximum,
                                    rounding=math.ceil)
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
                for chunks, shape in zip((pre_chunks, post_chunks),
                                         (pre_shape, post_shape)):
                    maxsize = min(np.max(chunks[ind]), shape[ind])
                    num = np.ceil(shape[ind] / maxsize)
                    chunks[ind] = tuple(len(ar) for ar in
                                        np.array_split(np.arange(shape[ind]),
                                                       num))
                pre_array = da.full(tuple(pre_shape),
                                    fill_value,
                                    chunks=tuple(pre_chunks))

                post_array = da.full(tuple(post_shape),
                                     fill_value,
                                     chunks=tuple(post_chunks))

                self.data = da.concatenate((pre_array, self.data, post_array),
                                           axis=ind)
            else:
                padding = []
                for i in range(self.data.ndim):
                    if i == axis.index_in_array:
                        padding.append((axis.high_index - ihigh + 1,
                                        ilow - axis.low_index))
                    else:
                        padding.append((0, 0))
                self.data = np.pad(self.data, padding, mode='constant',
                                   constant_values=(fill_value,))
            axis.offset += minimum
            axis.size += axis.high_index - ihigh + 1 + ilow - axis.low_index

        self._map_iterate(_shift1D, (('shift', shift_array.ravel()),),
                          original_axis=axis.axis,
                          fill_value=fill_value,
                          kind=interpolation_method,
                          offset=axis.offset,
                          scale=axis.scale,
                          size=axis.size,
                          show_progressbar=show_progressbar,
                          parallel=parallel,
                          ragged=False)

        if crop and not expand:
            _logger.debug("Cropping %s from index %i to %i"
                          % (self, ilow, ihigh))
            self.crop(axis.index_in_axes_manager,
                      ilow,
                      ihigh)

        self.events.data_changed.trigger(obj=self)
    shift1D.__doc__ %= CROP_PARAMETER_DOC

    def interpolate_in_between(self, start, end, delta=3, parallel=None,
                               show_progressbar=None, **kwargs):
        """Replace the data in a given range by interpolation.
        The operation is performed in place.
        Parameters
        ----------
        start, end : {int | float}
            The limits of the interval. If int they are taken as the
            axis index. If float they are taken as the axis value.
        delta : {int | float}
            The windows around the (start, end) to use for interpolation
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.
        parallel: {None, bool}
        All extra keyword arguments are passed to
        scipy.interpolate.interp1d. See the function documentation
        for details.
        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.
        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_one()
        axis = self.axes_manager.signal_axes[0]
        i1 = axis._get_index(start)
        i2 = axis._get_index(end)
        if isinstance(delta, float):
            delta = int(delta / axis.scale)
        i0 = int(np.clip(i1 - delta, 0, np.inf))
        i3 = int(np.clip(i2 + delta, 0, axis.size))

        def interpolating_function(dat):
            dat_int = sp.interpolate.interp1d(
                list(range(i0, i1)) + list(range(i2, i3)),
                dat[i0:i1].tolist() + dat[i2:i3].tolist(),
                **kwargs)
            dat[i1:i2] = dat_int(list(range(i1, i2)))
            return dat
        self._map_iterate(interpolating_function, ragged=False,
                          parallel=parallel, show_progressbar=show_progressbar)
        self.events.data_changed.trigger(obj=self)

    def _check_navigation_mask(self, mask):
        if mask is not None:
            if not isinstance(mask, BaseSignal):
                raise ValueError("mask must be a BaseSignal instance.")
            elif mask.axes_manager.signal_dimension not in (0, 1):
                raise ValueError("mask must be a BaseSignal "
                                 "with signal_dimension equal to 1")
            elif (mask.axes_manager.navigation_dimension !=
                  self.axes_manager.navigation_dimension):
                raise ValueError("mask must be a BaseSignal with the same "
                                 "navigation_dimension as the current signal.")

    def estimate_shift1D(self,
                         start=None,
                         end=None,
                         reference_indices=None,
                         max_shift=None,
                         interpolate=True,
                         number_of_interpolation_points=5,
                         mask=None,
                         parallel=None,
                         show_progressbar=None):
        """Estimate the shifts in the current signal axis using
         cross-correlation.
        This method can only estimate the shift by comparing
        unidimensional features that should not change the position in
        the signal axis. To decrease the memory usage, the time of
        computation and the accuracy of the results it is convenient to
        select the feature of interest providing sensible values for
        `start` and `end`. By default interpolation is used to obtain
        subpixel precision.
        Parameters
        ----------
        start, end : {int | float | None}
            The limits of the interval. If int they are taken as the
            axis index. If float they are taken as the axis value.
        reference_indices : tuple of ints or None
            Defines the coordinates of the spectrum that will be used
            as eference. If None the spectrum at the current
            coordinates is used for this purpose.
        max_shift : int
            "Saturation limit" for the shift.
        interpolate : bool
            If True, interpolation is used to provide sub-pixel
            accuracy.
        number_of_interpolation_points : int
            Number of interpolation points. Warning: making this number
            too big can saturate the memory
        mask : BaseSignal of bool data type.
            It must have signal_dimension = 0 and navigation_shape equal to the
            current signal. Where mask is True the shift is not computed
            and set to nan.
        parallel : {None, bool}
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.
        Returns
        -------
        An array with the result of the estimation in the axis units. although
        the computation is performed in batches if the signal is lazy, the
        result is computed in memory because it depends on the current state
        of the axes that could change later on in the workflow.

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_one()
        ip = number_of_interpolation_points + 1
        axis = self.axes_manager.signal_axes[0]
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
        iterating_kwargs = ()
        if mask is not None:
            iterating_kwargs += (('mask', mask),)
        shift_signal = self._map_iterate(
            _estimate_shift1D,
            iterating_kwargs=iterating_kwargs,
            data_slice=slice(i1, i2),
            ref=ref,
            ip=ip,
            interpolate=interpolate,
            ragged=False,
            parallel=parallel,
            inplace=False,
            show_progressbar=show_progressbar,)
        shift_array = shift_signal.data
        if max_shift is not None:
            if interpolate is True:
                max_shift *= ip
            shift_array.clip(-max_shift, max_shift)
        if interpolate is True:
            shift_array = shift_array / ip
        shift_array *= axis.scale
        if self._lazy:
            # We must compute right now because otherwise any changes to the
            # axes_manager of the signal later in the workflow may result in
            # a wrong shift_array
            shift_array = shift_array.compute()
        return shift_array

    def align1D(self,
                start=None,
                end=None,
                reference_indices=None,
                max_shift=None,
                interpolate=True,
                number_of_interpolation_points=5,
                interpolation_method='linear',
                crop=True,
                expand=False,
                fill_value=np.nan,
                also_align=None,
                mask=None,
                show_progressbar=None):
        """Estimate the shifts in the signal axis using
        cross-correlation and use the estimation to align the data in place.
        This method can only estimate the shift by comparing
        unidimensional
        features that should not change the position.

        To decrease memory usage, time of computation and improve
        accuracy it is convenient to select the feature of interest
        setting the `start` and `end` keywords. By default interpolation is
        used to obtain subpixel precision.

        Parameters
        ----------
        start, end : {int | float | None}
            The limits of the interval. If int they are taken as the
            axis index. If float they are taken as the axis value.
        reference_indices : tuple of ints or None
            Defines the coordinates of the spectrum that will be used
            as eference. If None the spectrum at the current
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
            Overrides `crop`.
        fill_value : float
            If crop is False fill the data outside of the original
            interval with the given value where needed.
        also_align : list of signals, None
            A list of BaseSignal instances that has exactly the same
            dimensions as this one and that will be aligned using the shift map
            estimated using the this signal.
        mask : BaseSignal of bool data type.
            It must have signal_dimension = 0 and navigation_shape equal to the
            current signal. Where mask is True the shift is not computed
            and set to nan.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.
        Returns
        -------
        An array with the result of the estimation. The shift will be

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.
        See also
        --------
        estimate_shift1D
        """
        if also_align is None:
            also_align = []
        self._check_signal_dimension_equals_one()
        if self._lazy:
            _logger.warning('In order to properly expand, the lazy '
                            'reference signal will be read twice (once to '
                            'estimate shifts, and second time to shift '
                            'appropriatelly), which might take a long time. '
                            'Use expand=False to only pass through the data '
                            'once.')
        shift_array = self.estimate_shift1D(
            start=start,
            end=end,
            reference_indices=reference_indices,
            max_shift=max_shift,
            interpolate=interpolate,
            number_of_interpolation_points=number_of_interpolation_points,
            mask=mask,
            show_progressbar=show_progressbar)
        signals_to_shift = [self] + also_align
        for signal in signals_to_shift:
            signal.shift1D(shift_array=shift_array,
                           interpolation_method=interpolation_method,
                           crop=crop,
                           fill_value=fill_value,
                           expand=expand,
                           show_progressbar=show_progressbar)
    align1D.__doc__ %= CROP_PARAMETER_DOC

    def integrate_in_range(self, signal_range='interactive',
                           display=True, toolkit=None):
        """ Sums the spectrum over an energy range, giving the integrated
        area.
        The energy range can either be selected through a GUI or the command
        line.
        Parameters
        ----------
        signal_range : {a tuple of this form (l, r), "interactive"}
            l and r are the left and right limits of the range. They can be
            numbers or None, where None indicates the extremes of the interval.
            If l and r are floats the `signal_range` will be in axis units (for
            example eV). If l and r are integers the `signal_range` will be in
            index units. When `signal_range` is "interactive" (default) the
            range is selected using a GUI.
        Returns
        -------
        integrated_spectrum : BaseSignal subclass
        See Also
        --------
        integrate_simpson
        Examples
        --------
        Using the GUI
        >>> s = hs.signals.Signal1D(range(1000))
        >>> s.integrate_in_range() #doctest: +SKIP

        Using the CLI
        >>> s_int = s.integrate_in_range(signal_range=(560,None))

        Selecting a range in the axis units, by specifying the
        signal range with floats.
        >>> s_int = s.integrate_in_range(signal_range=(560.,590.))

        Selecting a range using the index, by specifying the
        signal range with integers.
        >>> s_int = s.integrate_in_range(signal_range=(100,120))
        """
        from hyperspy.misc.utils import deprecation_warning
        msg = (
            "The `Signal1D.integrate_in_range` method is deprecated and will "
            "be removed in v2.0. Use a `roi.SpanRoi` followed by `integrate1D` "
            "instead.")
        deprecation_warning(msg)
        signal_range = signal_range_from_roi(signal_range)

        if signal_range == 'interactive':
            self_copy = self.deepcopy()
            ia = IntegrateArea(self_copy, signal_range)
            ia.gui(display=display, toolkit=toolkit)
            integrated_signal1D = self_copy
        else:
            integrated_signal1D = self._integrate_in_range_commandline(
                signal_range)
        return integrated_signal1D

    def _integrate_in_range_commandline(self, signal_range):
        signal_range = signal_range_from_roi(signal_range)
        e1 = signal_range[0]
        e2 = signal_range[1]
        integrated_signal1D = self.isig[e1:e2].integrate1D(-1)
        return integrated_signal1D

    def calibrate(self, display=True, toolkit=None):
        self._check_signal_dimension_equals_one()
        calibration = Signal1DCalibration(self)
        return calibration.gui(display=display, toolkit=toolkit)
    calibrate.__doc__ = \
        """
        Calibrate the spectral dimension using a gui.
        It displays a window where the new calibration can be set by:
        * Setting the offset, units and scale directly
        * Selection a range by dragging the mouse on the spectrum figure
         and
        setting the new values for the given range limits

        Parameters
        ----------
        %s
        %s

        Notes
        -----
        For this method to work the output_dimension must be 1. Set the
        view
        accordingly
        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.
        """ % (DISPLAY_DT, TOOLKIT_DT)

    def smooth_savitzky_golay(self,
                              polynomial_order=None,
                              window_length=None,
                              differential_order=0,
                              parallel=None, display=True, toolkit=None):
        self._check_signal_dimension_equals_one()
        if (polynomial_order is not None and
                window_length is not None):
            axis = self.axes_manager.signal_axes[0]
            self.map(savgol_filter, window_length=window_length,
                     polyorder=polynomial_order, deriv=differential_order,
                     delta=axis.scale, ragged=False, parallel=parallel)
        else:
            # Interactive mode
            smoother = SmoothingSavitzkyGolay(self)
            smoother.differential_order = differential_order
            if polynomial_order is not None:
                smoother.polynomial_order = polynomial_order
            if window_length is not None:
                smoother.window_length = window_length
            return smoother.gui(display=display, toolkit=toolkit)
    smooth_savitzky_golay.__doc__ = \
        """
        Apply a Savitzky-Golay filter to the data in place.
        If `polynomial_order` or `window_length` or `differential_order` are
        None the method is run in interactive mode.
        Parameters
        ----------
        window_length : int
            The length of the filter window (i.e. the number of coefficients).
            `window_length` must be a positive odd integer.
        polynomial_order : int
            The order of the polynomial used to fit the samples.
            `polyorder` must be less than `window_length`.
        differential_order: int, optional
            The order of the derivative to compute.  This must be a
            nonnegative integer.  The default is 0, which means to filter
            the data without differentiating.
        parallel : {bool, None}
            Perform the operation in a threaded manner (parallely).
        %s
        %s
        Notes
        -----
        More information about the filter in `scipy.signal.savgol_filter`.
        """ % (DISPLAY_DT, TOOLKIT_DT)

    def smooth_lowess(self,
                      smoothing_parameter=None,
                      number_of_iterations=None,
                      show_progressbar=None,
                      parallel=None, display=True, toolkit=None):
        if not statsmodels_installed:
            raise ImportError("statsmodels is not installed. This package is "
                              "required for this feature.")
        self._check_signal_dimension_equals_one()
        if smoothing_parameter is None or number_of_iterations is None:
            smoother = SmoothingLowess(self)
            if smoothing_parameter is not None:
                smoother.smoothing_parameter = smoothing_parameter
            if number_of_iterations is not None:
                smoother.number_of_iterations = number_of_iterations
            return smoother.gui(display=display, toolkit=toolkit)
        else:
            self.map(lowess,
                     exog=self.axes_manager[-1].axis,
                     frac=smoothing_parameter,
                     it=number_of_iterations,
                     is_sorted=True,
                     return_sorted=False,
                     show_progressbar=show_progressbar,
                     ragged=False,
                     parallel=parallel)
    smooth_lowess.__doc__ = \
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
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.
        parallel : {Bool, None, int}
            Perform the operation parallel
        %s
        %s

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.
        ImportError if statsmodels is not installed.
        Notes
        -----
        This method uses the lowess algorithm from statsmodels. statsmodels
        is required for this method.
        """ % (DISPLAY_DT, TOOLKIT_DT)

    def smooth_tv(self, smoothing_parameter=None, show_progressbar=None,
                  parallel=None, display=True, toolkit=None):
        self._check_signal_dimension_equals_one()
        if smoothing_parameter is None:
            smoother = SmoothingTV(self)
            return smoother.gui(display=display, toolkit=toolkit)
        else:
            self.map(_tv_denoise_1d, weight=smoothing_parameter,
                     ragged=False,
                     show_progressbar=show_progressbar,
                     parallel=parallel)
    smooth_tv.__doc__ = \
        """
        Total variation data smoothing in place.
        Parameters
        ----------
        smoothing_parameter: float or None
           Denoising weight relative to L2 minimization. If None the method
           is run in interactive mode.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.
        parallel : {Bool, None, int}
            Perform the operation parallely
        %s
        %s
        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """ % (DISPLAY_DT, TOOLKIT_DT)

    def filter_butterworth(self,
                           cutoff_frequency_ratio=None,
                           type='low',
                           order=2, display=True, toolkit=None):
        self._check_signal_dimension_equals_one()
        smoother = ButterworthFilter(self)
        if cutoff_frequency_ratio is not None:
            smoother.cutoff_frequency_ratio = cutoff_frequency_ratio
            smoother.type = type
            smoother.order = order
            smoother.apply()
        else:
            return smoother.gui(display=display, toolkit=toolkit)
    filter_butterworth.__doc__ = \
        """
        Butterworth filter in place.

        Parameters
        ----------
        %s
        %s
        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """ % (DISPLAY_DT, TOOLKIT_DT)

    def _remove_background_cli(
            self, signal_range, background_estimator, fast=True,
            zero_fill=False, show_progressbar=None):
        signal_range = signal_range_from_roi(signal_range)
        from hyperspy.models.model1d import Model1D
        model = Model1D(self)
        model.append(background_estimator)
        background_estimator.estimate_parameters(
            self,
            signal_range[0],
            signal_range[1],
            only_current=False)
        if fast and not self._lazy:
            try:
                axis = self.axes_manager.signal_axes[0].axis
                result = Signal1D(self.data -
                                  background_estimator.function_nd(axis))
            except MemoryError:
                result = self - model.as_signal(
                    show_progressbar=show_progressbar)
        else:
            model.set_signal_range(signal_range[0], signal_range[1])
            model.multifit(show_progressbar=show_progressbar)
            model.reset_signal_range()
            result = self - model.as_signal(show_progressbar=show_progressbar)

        if zero_fill:
            if self._lazy:
                low_idx = result.axes_manager[-1].value2index(signal_range[0])
                z = da.zeros(low_idx, chunks=(low_idx,))
                cropped_da = result.data[low_idx:]
                result.data = da.concatenate([z, cropped_da])
            else:
                result.isig[:signal_range[0]] = 0
        return result

    def remove_background(
            self,
            signal_range='interactive',
            background_type='Power Law',
            polynomial_order=2,
            fast=True,
            zero_fill=False,
            plot_remainder=True,
            show_progressbar=None, display=True, toolkit=None):

        self._check_signal_dimension_equals_one()
        if signal_range == 'interactive':
            br = BackgroundRemoval(self, background_type=background_type,
                                   polynomial_order=polynomial_order,
                                   fast=fast,
                                   plot_remainder=plot_remainder,
                                   show_progressbar=show_progressbar,
                                   zero_fill=zero_fill)
            return br.gui(display=display, toolkit=toolkit)
        else:
            if background_type in ('PowerLaw', 'Power Law'):
                background_estimator = components1d.PowerLaw()
            elif background_type == 'Gaussian':
                background_estimator = components1d.Gaussian()
            elif background_type == 'Offset':
                background_estimator = components1d.Offset()
            elif background_type == 'Polynomial':
                background_estimator = components1d.Polynomial(
                    polynomial_order)
            else:
                raise ValueError(
                    "Background type: " +
                    background_type +
                    " not recognized")
            spectra = self._remove_background_cli(
                signal_range=signal_range,
                background_estimator=background_estimator,
                fast=fast,
                zero_fill=zero_fill,
                show_progressbar=show_progressbar)
            return spectra
    remove_background.__doc__ = \
        """
        Remove the background, either in place using a gui or returned as a new
        spectrum using the command line.
        Parameters
        ----------
        signal_range : tuple, optional
            If this argument is not specified, the signal range has to be
            selected using a GUI. And the original spectrum will be replaced.
            If tuple is given, the a spectrum will be returned.
        background_type : string
            The type of component which should be used to fit the background.
            Possible components: PowerLaw, Gaussian, Offset, Polynomial
            If Polynomial is used, the polynomial order can be specified
        polynomial_order : int, default 2
            Specify the polynomial order if a Polynomial background is used.
        fast : bool
            If True, perform an approximative estimation of the parameters.
            If False, the signal is fitted using non-linear least squares
            afterwards.This is slower compared to the estimation but
            possibly more accurate.
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
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.
        %s
        %s
        Examples
        --------
        Using gui, replaces spectrum s
        >>> s = hs.signals.Signal1D(range(1000))
        >>> s.remove_background() #doctest: +SKIP

        Using command line, returns a spectrum
        >>> s1 = s.remove_background(signal_range=(400,450), background_type='PowerLaw')

        Using a full model to fit the background
        >>> s1 = s.remove_background(signal_range=(400,450), fast=False)

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.
        """ % (DISPLAY_DT, TOOLKIT_DT)

    @interactive_range_selector
    def crop_signal1D(self, left_value=None, right_value=None,):
        """Crop in place the spectral dimension.

        Parameters
        ----------
        left_value, righ_value: {int | float | None}
            If int the values are taken as indices. If float they are
            converted to indices using the spectral axis calibration.
            If left_value is None crops from the beginning of the axis.
            If right_value is None crops up to the end of the axis. If
            both are
            None the interactive cropping interface is activated
            enabling
            cropping the spectrum using a span selector in the signal
            plot.

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """
        self._check_signal_dimension_equals_one()
        try:
            left_value, right_value = signal_range_from_roi(left_value)
        except TypeError:
            # It was not a ROI, we carry on
            pass
        self.crop(axis=self.axes_manager.signal_axes[0].index_in_axes_manager,
                  start=left_value, end=right_value)

    def gaussian_filter(self, FWHM):
        """Applies a Gaussian filter in the spectral dimension in place.

        Parameters
        ----------
        FWHM : float
            The Full Width at Half Maximum of the gaussian in the
            spectral axis units

        Raises
        ------
        ValueError if FWHM is equal or less than zero.

        SignalDimensionError if the signal dimension is not 1.

        """
        self._check_signal_dimension_equals_one()
        if FWHM <= 0:
            raise ValueError(
                "FWHM must be greater than zero")
        axis = self.axes_manager.signal_axes[0]
        FWHM *= 1 / axis.scale
        self.map(gaussian_filter1d, sigma=FWHM / 2.35482, ragged=False)

    def hanning_taper(self, side='both', channels=None, offset=0):
        """Apply a hanning taper to the data in place.

        Parameters
        ----------
        side : {'left', 'right', 'both'}
        channels : {None, int}
            The number of channels to taper. If None 5% of the total
            number of channels are tapered.
        offset : int

        Returns
        -------
        channels

        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """
        if not np.issubdtype(self.data.dtype, np.floating):
            raise TypeError("The data dtype should be `float`. It can be "
                            "changed by using the `change_dtype('float')` "
                            "method of the signal.")

        # TODO: generalize it
        self._check_signal_dimension_equals_one()
        if channels is None:
            channels = int(round(len(self()) * 0.02))
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
            zeros = da.zeros(nav_shape + (offset,),
                             chunks=nav_chunks + ((offset,),))

        if side == 'left' or side == 'both':
            if self._lazy:
                tapered = dc[..., offset:channels + offset]
                tapered *= np.hanning(2 * channels)[:channels]
                therest = dc[..., channels + offset:]
                thelist = [] if offset == 0 else [zeros]
                thelist.extend([tapered, therest])
                dc = da.concatenate(thelist, axis=-1)
            else:
                dc[..., offset:channels + offset] *= (
                    np.hanning(2 * channels)[:channels])
                dc[..., :offset] *= 0.
        if side == 'right' or side == 'both':
            rl = None if offset == 0 else -offset
            if self._lazy:
                therest = dc[..., :-channels - offset]
                tapered = dc[..., -channels - offset:rl]
                tapered *= np.hanning(2 * channels)[-channels:]
                thelist = [therest, tapered]
                if offset != 0:
                    thelist.append(zeros)
                dc = da.concatenate(thelist, axis=-1)
            else:
                dc[..., -channels - offset:rl] *= (
                    np.hanning(2 * channels)[-channels:])
                if offset != 0:
                    dc[..., -offset:] *= 0.

        if self._lazy:
            self.data = dc
        self.events.data_changed.trigger(obj=self)
        return channels

    def find_peaks1D_ohaver(self, xdim=None, slope_thresh=0, amp_thresh=None,
                            subchannel=True, medfilt_radius=5, maxpeakn=30000,
                            peakgroup=10, parallel=None):
        """Find peaks along a 1D line (peaks in spectrum/spectra).

        Function to locate the positive peaks in a noisy x-y data set.

        Detects peaks by looking for downward zero-crossings in the
        first derivative that exceed 'slope_thresh'.

        Returns an array containing position, height, and width of each
        peak.

        'slope_thresh' and 'amp_thresh', control sensitivity: higher
        values will neglect broad peaks (slope) and smaller features (amp),
        respectively.

        peakgroup is the number of points around the top of the peak
        that are taken to estimate the peak height. For spikes or very
        narrow peaks, keep PeakGroup=1 or 2; for broad or noisy peaks,
        make PeakGroup larger to reduce the effect of noise.

        Parameters
        ----------

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
                    number of points around the "top part" of the peak
                    that are taken to estimate the peak height;
                    default is set to 10

        maxpeakn : int (optional)
                   number of maximum detectable peaks;
                   default is set to 5000.

        subchannel : bool (optional)
                 default is set to True.

        parallel : {None, bool}
            Perform the operation in a threaded (parallel) manner.

        Returns
        -------
        peaks : structured array of shape (npeaks)
            contains fields: 'position', 'width', and 'height' for each peak.


        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        """
        # TODO: add scipy.signal.find_peaks_cwt
        self._check_signal_dimension_equals_one()
        axis = self.axes_manager.signal_axes[0].axis
        peaks = self.map(find_peaks_ohaver,
                         x=axis,
                         slope_thresh=slope_thresh,
                         amp_thresh=amp_thresh,
                         medfilt_radius=medfilt_radius,
                         maxpeakn=maxpeakn,
                         peakgroup=peakgroup,
                         subchannel=subchannel,
                         ragged=True,
                         parallel=parallel,
                         inplace=False)
        return peaks.data

    def estimate_peak_width(self,
                            factor=0.5,
                            window=None,
                            return_interval=False,
                            parallel=None,
                            show_progressbar=None):
        """Estimate the width of the highest intensity of peak
        of the spectra at a given fraction of its maximum.

        It can be used with asymmetric peaks. For accurate results any
        background must be previously substracted.
        The estimation is performed by interpolation using cubic splines.

        Parameters
        ----------
        factor : 0 < float < 1
            The default, 0.5, estimates the FWHM.
        window : None, float
            The size of the window centred at the peak maximum
            used to perform the estimation.
            The window size must be chosen with care: if it is narrower
            than the width of the peak at some positions or if it is
            so wide that it includes other more intense peaks this
            method cannot compute the width and a NaN is stored instead.
        return_interval: bool
            If True, returns 2 extra signals with the positions of the
            desired height fraction at the left and right of the
            peak.
        parallel : {None, bool}
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.

        Returns
        -------
        width or [width, left, right], depending on the value of
        `return_interval`.

        """

        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_one()
        if not 0 < factor < 1:
            raise ValueError("factor must be between 0 and 1.")

        axis = self.axes_manager.signal_axes[0]
        # x = axis.axis
        maxval = self.axes_manager.navigation_size
        show_progressbar = show_progressbar and maxval > 0

        def estimating_function(spectrum,
                                window=None,
                                factor=0.5,
                                axis=None):
            x = axis.axis
            if window is not None:
                vmax = axis.index2value(spectrum.argmax())
                slices = axis._get_array_slices(
                    slice(vmax - window * 0.5, vmax + window * 0.5))
                spectrum = spectrum[slices]
                x = x[slices]
            spline = scipy.interpolate.UnivariateSpline(
                x,
                spectrum - factor * spectrum.max(),
                s=0)
            roots = spline.roots()
            if len(roots) == 2:
                return np.array(roots)
            else:
                return np.full((2,), np.nan)

        both = self._map_iterate(estimating_function,
                                 window=window,
                                 factor=factor,
                                 axis=axis,
                                 ragged=False,
                                 inplace=False,
                                 parallel=parallel,
                                 show_progressbar=show_progressbar)
        left, right = both.T.split()
        width = right - left
        if factor == 0.5:
            width.metadata.General.title = (
                self.metadata.General.title + " FWHM")
            left.metadata.General.title = (
                self.metadata.General.title + " FWHM left position")

            right.metadata.General.title = (
                self.metadata.General.title + " FWHM right position")
        else:
            width.metadata.General.title = (
                self.metadata.General.title +
                " full-width at %.1f maximum" % factor)

            left.metadata.General.title = (
                self.metadata.General.title +
                " full-width at %.1f maximum left position" % factor)
            right.metadata.General.title = (
                self.metadata.General.title +
                " full-width at %.1f maximum right position" % factor)
        for signal in (left, width, right):
            signal.axes_manager.set_signal_dimension(0)
            signal.set_signal_type("")
        if return_interval is True:
            return [width, left, right]
        else:
            return width


class LazySignal1D(LazySignal, Signal1D):

    """
    """
    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axes_manager.set_signal_dimension(1)
