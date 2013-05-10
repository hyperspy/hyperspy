# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

import copy

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import traits.api as t
from scipy.ndimage.filters import gaussian_filter1d

from hyperspy.signal import Signal
from hyperspy.misc import progressbar
from hyperspy.misc import utils
from hyperspy.misc import utils_varia
from hyperspy.gui.tools import (
    SpectrumCalibration,
    SmoothingSavitzkyGolay,
    SmoothingLowess,
    SpectrumRangeSelector,
    SpanSelectorInSpectrum,
    SmoothingTV,
    ButterworthFilter)
from hyperspy.gui.egerton_quantification import BackgroundRemoval
from hyperspy.drawing import signal as sigdraw
from hyperspy.decorators import only_interactive
from hyperspy.defaults_parser import preferences
from hyperspy.decorators import interactive_range_selector
from hyperspy.decorators import auto_replot
from hyperspy.misc.utils import one_dim_findpeaks
from hyperspy.exceptions import SignalSizeError
            
class Spectrum(Signal):
    """
    """
    _default_record_by = 'spectrum'
    
    def __init__(self, *args, **kwargs):
        Signal.__init__(self, *args, **kwargs)
        self.axes_manager.set_signal_dimension(1)

    def shift_1D(self,
                 shift_array,
                 interpolation_method='linear',
                 crop=True,
                 fill_value=np.nan):
        """Shift the data over the signal axis by the amount specified
        by an array.

        Parameters
        ----------
        shift_array : numpy array
            An array containing the shifting amount. It must have
            `axes_manager.navigation_shape` shape.
        interpolation_method : str or int
            Specifies the kind of interpolation as a string ('linear',
            'nearest', 'zero', 'slinear', 'quadratic, 'cubic') or as an 
            integer specifying the order of the spline interpolator to 
            use.
        crop : bool
            If True automatically crop the signal axis at both ends if 
            needed.
        fill_value : float
            If crop is False fill the data outside of the original 
            interval with the given value where needed.
            
        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.
            
        """
        
        self._check_signal_dimension_equals_one()
        axis = self.axes_manager.signal_axes[0]
        offset = axis.offset
        original_axis = axis.axis.copy()
        pbar = progressbar.progressbar(
            maxval=self.axes_manager.navigation_size)
        for i, (dat, shift) in enumerate(zip(
                self._iterate_signal(),
                shift_array)):
            si = sp.interpolate.interp1d(original_axis,
                                         dat,
                                         bounds_error=False,
                                         fill_value=fill_value,
                                         kind=interpolation_method)
            axis.offset = float(offset - shift)
            dat[:] = si(axis.axis)
            pbar.update(i + 1)
            
        axis.offset = offset

        if crop is True:
            mini, maxi = shift_array.min(), shift_array.max()
            if mini < 0:
                self.crop(axis.index_in_axes_manager,
                          None,
                          axis.axis[-1] + mini + axis.scale)
            if maxi > 0:
                self.crop(axis.index_in_axes_manager,
                          float(offset + maxi))
            
    def interpolate_in_between(self, start, end, delta=3, **kwargs):
        """Replace the data in a given range by interpolation.
        
        The operation is performed in place.
        
        Parameters
        ----------
        start, end : {int | float}
            The limits of the interval. If int they are taken as the 
            axis index. If float they are taken as the axis value.
        
        All extra keyword arguments are passed to 
        scipy.interpolate.interp1d. See the function documentation 
        for details.
        
        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.
        
        """
        self._check_signal_dimension_equals_one()
        axis = self.axes_manager.signal_axes[0]
        i1 = axis._get_index(start)
        i2 = axis._get_index(end)
        i0 = int(np.clip(i1 - delta, 0, np.inf))
        i3 = int(np.clip(i2 + delta, 0, axis.size))
        pbar = progressbar.progressbar(
            maxval=self.axes_manager.navigation_size)
        for i, dat in enumerate(self._iterate_signal()):
            dat_int = sp.interpolate.interp1d(
                range(i0,i1) + range(i2,i3),
                dat[i0:i1].tolist() + dat[i2:i3].tolist(),
                **kwargs)
            dat[i1:i2] = dat_int(range(i1,i2))
            pbar.update(i + 1)
            
    def estimate_shift_1D(self,
                          start=None,
                          end=None,
                          reference_indices=None,
                          max_shift=None,
                          interpolate=True,
                          number_of_interpolation_points=5):
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

        Return
        ------
        An array with the result of the estimation in the axis units.
        
        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.
        
        """
        self._check_signal_dimension_equals_one()
        ip = number_of_interpolation_points + 1
        axis = self.axes_manager.signal_axes[0]
        if reference_indices is None:
            reference_indices = self.axes_manager.indices

        i1, i2 = axis._get_index(start), axis._get_index(end) 
        shift_array = np.zeros(self.axes_manager.navigation_shape)
        ref = self.navigation_indexer[reference_indices].data[i1:i2]
        if interpolate is True:
            ref = utils.interpolate_1D(ip, ref)
        pbar = progressbar.progressbar(
            maxval=self.axes_manager.navigation_size)
        for i, (dat, indices) in enumerate(zip(
                    self._iterate_signal(),
                    np.ndindex(shift_array.shape))):
            dat = dat[i1:i2]
            if interpolate is True:
                dat = utils.interpolate_1D(ip, dat)
            shift_array[indices] = np.argmax(
                np.correlate(ref, dat,'full')) - len(ref) + 1
            pbar.update(i + 1)
        pbar.finish()

        if max_shift is not None:
            if interpolate is True:
                max_shift *= ip
            shift_array.clip(a_min=-max_shift, a_max=max_shift)
        if interpolate is True:
            shift_array /= ip
        shift_array *= axis.scale
        return shift_array

    def align_1D(self,
                 start=None,
                 end=None,
                 reference_indices=None,
                 max_shift=None,
                 interpolate=True,
                 number_of_interpolation_points=5,
                 interpolation_method='linear',
                 crop=True,
                 fill_value=np.nan,
                 also_align=None):
        """Estimates the shifts in the signal axis using 
        cross-correlation and uses
         the estimation to align the data.

        This method can only estimate the shift by comparing 
        unidimensional
        features that should not change the position. 
        To decrease memory usage, time of computation and improve 
        accuracy it is convenient to select the feature of interest 
        setting the `start` and `end` keywords. By default interpolation is used to obtain subpixel precision.

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
        crop : bool
            If True automatically crop the signal axis at both ends if 
            needed.
        fill_value : float
            If crop is False fill the data outside of the original 
            interval with the given value where needed.
        also_align : list of signals
            A list of Signal instances that has exactly the same 
            dimensions
            as this one and that will be aligned using the shift map
            estimated using the this signal.

        Return
        ------
        An array with the result of the estimation. The shift will be
        
        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.

        See also
        --------
        estimate_shift_in_units_1D, estimate_shift_in_index_1D
        
        """
        self._check_signal_dimension_equals_one()
        shift_array = self.estimate_shift_1D(
            start=start,
            end=end,
            reference_indices=reference_indices,
            max_shift=max_shift,
            interpolate=interpolate,
            number_of_interpolation_points=
                number_of_interpolation_points)
        if also_align is None:
            also_align = list()
        also_align.append(self)
        for signal in also_align:
            signal.shift_1D(shift_array=shift_array,
                            interpolation_method=interpolation_method,
                            crop=crop,
                            fill_value=fill_value)

    def peakfind_1D(self, xdim=None,slope_thresh=0.5, amp_thresh=None, 
                    subchannel=True, medfilt_radius=5, maxpeakn=30000, 
                    peakgroup=10):
        """Find peaks along a 1D line (peaks in spectrum/spectra).

        Function to locate the positive peaks in a noisy x-y data set.

        Detects peaks by looking for downward zero-crossings in the 
        first
        derivative that exceed 'slope_thresh'.

        Returns an array containing position, height, and width of each 
        peak.

        'slope_thresh' and 'amp_thresh', control sensitivity: higher 
        values will
        neglect smaller features.

        peakgroup is the number of points around the top peak to search 
        around

        Parameters
        ---------

        slope_thresh : float (optional)
                       1st derivative threshold to count the peak
                       default is set to 0.5
                       higher values will neglect smaller features.

        amp_thresh : float (optional)
                     intensity threshold above which
                     default is set to 10% of max(y)
                     higher values will neglect smaller features.

        medfilt_radius : int (optional)
                     median filter window to apply to smooth the data
                     (see scipy.signal.medfilt)
                     if 0, no filter will be applied.
                     default is set to 5

        peakgroup : int (optional)
                    number of points around the "top part" of the peak
                    default is set to 10

        maxpeakn : int (optional)
                   number of maximum detectable peaks
                   default is set to 5000

        subpix : bool (optional)
                 default is set to True

        Returns
        -------
        P : array of shape (npeaks, 3)
            contains position, height, and width of each peak
            
        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.
            
        """
        # TODO: generalize the code
        self._check_signal_dimension_equals_one()
        if len(self.data.shape)==1:
            # preallocate a large array for the results
            self.peaks=one_dim_findpeaks(self.data,
                slope_thresh=slope_thresh,
                amp_thresh=amp_thresh,
                medfilt_radius=medfilt_radius,
                maxpeakn=maxpeakn,
                peakgroup=peakgroup,
                subchannel=subchannel)
                
        elif len(self.data.shape)==2:
            pbar=progressbar.ProgressBar(
                    maxval=self.data.shape[1]).start()
            # preallocate a large array for the results
            # the third dimension is for the number of rows in your 
            # data.
            # assumes spectra are rows of the 2D array, each col is a 
            # channel.
            self.peaks=np.zeros((maxpeakn,3,self.data.shape[0]))
            for i in xrange(self.data.shape[1]):
                tmp=one_dim_findpeaks(
                                    self.data[i],
                                    slope_thresh=slope_thresh,
                                    amp_thresh=amp_thresh,
                                    medfilt_radius=medfilt_radius,
                                    maxpeakn=maxpeakn,
                                    peakgroup=peakgroup,
                                    subchannel=subchannel)
                self.peaks[:tmp.shape[0],:,i]=tmp
                pbar.update(i + 1)
            # trim any extra blank space
            # works by summing along axes to compress to a 1D line, then
            # finds
            # the first 0 along that line.
            trim_id=np.min(np.nonzero(np.sum(np.sum(self.peaks,axis=2),
                                             axis=1)==0))
            pbar.finish()
            self.peaks=self.peaks[:trim_id,:,:]
        elif len(self.data.shape)==3:
            # preallocate a large array for the results
            self.peaks=np.zeros((maxpeakn,3,self.data.shape[0],
                                 self.data.shape[1]))
            pbar=progressbar.ProgressBar(
                maxval=self.data.shape[0]).start()
            for i in xrange(self.data.shape[0]):
                for j in xrange(self.data.shape[1]):
                    tmp=one_dim_findpeaks(self.data[i,j],
                            slope_thresh=slope_thresh,
                            amp_thresh=amp_thresh,
                            medfilt_radius=medfilt_radius,
                            maxpeakn=maxpeakn,
                            peakgroup=peakgroup,
                            subchannel=subchannel)
                    self.peaks[:tmp.shape[0],:,i,j]=tmp
                pbar.update(i + 1)
            # trim any extra blank space
            # works by summing along axes to compress to a 1D line, 
            # then finds
            # the first 0 along that line.
            trim_id=np.min(np.nonzero(np.sum(np.sum(np.sum(
                            self.peaks,axis=3),axis=2),axis=1)==0))
            pbar.finish()
            self.peaks=self.peaks[:trim_id,:,:,:]

    def to_image(self, signal_to_index=0):
        """Spectrum to image

        Parameters
        ----------
        signal_to_index : integer
            Position to move the signal axis.        
            
        Examples
        --------        
        >>> s = signals.Spectrum({'data' : np.ones((3,4,5,6))})
        >>> s
        <Spectrum, title: , dimensions: (3L, 4L, 5L, 6L)>

        >>> s.to_image()
        <Image, title: , dimensions: (6L, 3L, 4L, 5L)>

        >>> s.to_image(1)
        <Image, title: , dimensions: (3L, 6L, 4L, 5L)>
        
        """
        from hyperspy.signals.image import Image
        dic = self._get_signal_dict()
        dic['mapped_parameters']['record_by'] = 'image'
        dic['data'] = np.rollaxis(dic['data'], -1, signal_to_index)
        dic['axes'] = utils_varia.rollelem(dic['axes'],-1,signal_to_index)
        im = Image(dic)
        
        if hasattr(self, 'learning_results'):
            if (signal_to_index != 0 and 
                self.learning_results.loadings is not None):
                print("The learning results won't be transfered correctly")
            else:
                im.learning_results = copy.deepcopy(
                    self.learning_results)
                im.learning_results._transpose_results()
                im.learning_results.original_shape = self.data.shape

        im.tmp_parameters = self.tmp_parameters.deepcopy()
        return im

    def to_EELS(self):
        from hyperspy.signals.eels import EELSSpectrum
        dic = self._get_signal_dict()
        dic['mapped_parameters']['signal_type'] = 'EELS'
        eels = EELSSpectrum(dic)
        if hasattr(self, 'learning_results'):
            eels.learning_results = copy.deepcopy(self.learning_results)
        eels.tmp_parameters = self.tmp_parameters.deepcopy()
        return eels
        
    def to_simulation(self):
        from hyperspy.signals.spectrum_simulation import (
                SpectrumSimulation)
        dic = self._get_signal_dict()
        signal_type = self.mapped_parameters.signal_type
        dic['mapped_parameters']['signal_type'] = \
            signal_type + '_simulation'
        simu = SpectrumSimulation(dic)
        if hasattr(self, 'learning_results'):
            simu.learning_results = copy.deepcopy(self.learning_results)
        simu.tmp_parameters = self.tmp_parameters.deepcopy()
        return simu
    
    @only_interactive
    def calibrate(self, return_obj = False):
        """Calibrate the spectral dimension using a gui

        It displays a window where the new calibration can be set by:
        * Setting the offset, units and scale directly
        * Selection a range by dragging the mouse on the spectrum figure
         and
        setting the new values for the given range limits

        Notes
        -----
        For this method to work the output_dimension must be 1. Set the 
        view
        accordingly
        
        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.
        
        """
        self._check_signal_dimension_equals_one()
        calibration = SpectrumCalibration(self)
        calibration.edit_traits()
        if return_obj is True:
            return calibration

    def smooth_savitzky_golay(self, polynomial_order=None,
        number_of_points=None, differential_order=0):
        """Savitzky-Golay data smoothing.
        
        """
        self._check_signal_dimension_equals_one()
        if (polynomial_order is not None and 
            number_of_points) is not None:
            for spectrum in self:
                spectrum.data[:] = utils.sg(self(),
                                            number_of_points, 
                                            polynomial_order,
                                            differential_order)
        else:
            smoother = SmoothingSavitzkyGolay(self)
            smoother.differential_order = differential_order
            if polynomial_order is not None:
                smoother.polynomial_order = polynomial_order
            if number_of_points is not None:
                smoother.number_of_points = number_of_points

            smoother.edit_traits()
            
    def smooth_lowess(self, smoothing_parameter=None,
        number_of_iterations=None, differential_order=0):
        """Lowess data smoothing.
        
        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.
        
        """
        self._check_signal_dimension_equals_one()
        smoother = SmoothingLowess(self)
        smoother.differential_order = differential_order
        if smoothing_parameter is not None:
            smoother.smoothing_parameter = smoothing_parameter
        if number_of_iterations is not None:
            smoother.number_of_iterations = number_of_iterations
        if smoothing_parameter is None or smoothing_parameter is None:
            smoother.edit_traits()
        else:
            smoother.apply()

    def smooth_tv(self, smoothing_parameter=None, differential_order=0):
        """Total variation data smoothing.
        
        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.
        
        """
        self._check_signal_dimension_equals_one()
        smoother = SmoothingTV(self)
        smoother.differential_order = differential_order
        if smoothing_parameter is not None:
            smoother.smoothing_parameter = smoothing_parameter
        if smoothing_parameter is None:
            smoother.edit_traits()
        else:
            smoother.apply()
    
    def filter_butterworth(self,
                           cutoff_frequency_ratio=None,
                           type='low',
                           order=2):
        """Butterworth filter.
        
        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.
        
        """
        self._check_signal_dimension_equals_one()
        smoother = ButterworthFilter(self)
        if cutoff_frequency_ratio is not None:
            smoother.cutoff_frequency_ratio = cutoff_frequency_ratio
            smoother.apply()
        else:
            smoother.edit_traits()
        
    @only_interactive
    def remove_background(self):
        """Remove the background using a gui.
        
        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.
        
        """
        self._check_signal_dimension_equals_one()
        br = BackgroundRemoval(self)
        br.edit_traits()

    @interactive_range_selector    
    def crop_spectrum(self, left_value=None, right_value=None,):
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
        self.crop(
            axis=self.axes_manager.signal_axes[0].index_in_axes_manager,
            start=left_value, end=right_value)
        
    @auto_replot    
    def gaussian_filter(self, FWHM):
        """Applies a Gaussian filter in the spectral dimension.
        
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
        FWHM *= 1/axis.scale
        self.data = gaussian_filter1d(
            self.data,
            axis=axis.index_in_array, 
            sigma=FWHM/2.35482)
    
    @auto_replot
    def hanning_taper(self, side='both', channels=None, offset=0):
        """Hanning taper
        
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
        # TODO: generalize it
        self._check_signal_dimension_equals_one()
        if channels is None:
            channels = int(round(len(self()) * 0.02))
            if channels < 20:
                channels = 20
        dc = self.data
        if side == 'left' or side == 'both':
            dc[..., offset:channels+offset] *= (
                np.hanning(2*channels)[:channels])
            dc[...,:offset] *= 0. 
        if side== 'right' or side == 'both':
            if offset == 0:
                rl = None
            else:
                rl = -offset
            dc[..., -channels-offset:rl] *= (
                np.hanning(2*channels)[-channels:])
            if offset != 0:
                dc[..., -offset:] *= 0.
        return channels
