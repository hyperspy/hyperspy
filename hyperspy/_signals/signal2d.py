# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import numpy.ma as ma
import dask.array as da
import logging
import warnings

from scipy import ndimage
try:
    # For scikit-image >= 0.17.0
    from skimage.registration._phase_cross_correlation import _upsampled_dft
except ModuleNotFoundError:
    from skimage.feature.register_translation import _upsampled_dft

from hyperspy.defaults_parser import preferences
from hyperspy.external.progressbar import progressbar
from hyperspy.misc.math_tools import symmetrize, antisymmetrize, optimal_fft_size
from hyperspy.signal import BaseSignal
from hyperspy._signals.lazy import LazySignal
from hyperspy._signals.common_signal2d import CommonSignal2D
from hyperspy.signal_tools import PeaksFinder2D
from hyperspy.docstrings.plot import (
    BASE_PLOT_DOCSTRING, BASE_PLOT_DOCSTRING_PARAMETERS, PLOT2D_DOCSTRING,
    PLOT2D_KWARGS_DOCSTRING)
from hyperspy.docstrings.signal import SHOW_PROGRESSBAR_ARG, PARALLEL_ARG, MAX_WORKERS_ARG
from hyperspy.ui_registry import DISPLAY_DT, TOOLKIT_DT
from hyperspy.utils.peakfinders2D import (
        find_local_max, find_peaks_max, find_peaks_minmax, find_peaks_zaefferer,
        find_peaks_stat, find_peaks_log, find_peaks_dog, find_peaks_xc)


_logger = logging.getLogger(__name__)


def shift_image(im, shift=0, interpolation_order=1, fill_value=np.nan):
    if np.any(shift):
        fractional, integral = np.modf(shift)
        if fractional.any():
            order = interpolation_order
        else:
            # Disable interpolation
            order = 0
        return ndimage.shift(im, shift, cval=fill_value, order=order)
    else:
        return im


def triu_indices_minus_diag(n):
    """Returns the indices for the upper-triangle of an (n, n) array
    excluding its diagonal

    Parameters
    ----------
    n : int
        The length of the square array

    """
    ti = np.triu_indices(n)
    isnotdiag = ti[0] != ti[1]
    return ti[0][isnotdiag], ti[1][isnotdiag]


def hanning2d(M, N):
    """
    A 2D hanning window created by outer product.
    """
    return np.outer(np.hanning(M), np.hanning(N))


def sobel_filter(im):
    sx = ndimage.sobel(im, axis=0, mode='constant')
    sy = ndimage.sobel(im, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    return sob


def fft_correlation(in1, in2, normalize=False, real_only=False):
    """Correlation of two N-dimensional arrays using FFT.

    Adapted from scipy's fftconvolve.

    Parameters
    ----------
    in1, in2 : array
        Input arrays to convolve.
    normalize: bool, default False
        If True performs phase correlation.
    real_only : bool, default False
        If True, and in1 and in2 are real-valued inputs, uses
        rfft instead of fft for approx. 2x speed-up.

    """
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    size = s1 + s2 - 1

    # Calculate optimal FFT size
    complex_result = (in1.dtype.kind == 'c' or in2.dtype.kind == 'c')
    fsize = [optimal_fft_size(a, not complex_result) for a in size]

    # For real-valued inputs, rfftn is ~2x faster than fftn
    if not complex_result and real_only:
        fft_f, ifft_f = np.fft.rfftn, np.fft.irfftn
    else:
        fft_f, ifft_f = np.fft.fftn, np.fft.ifftn

    fprod = fft_f(in1, fsize)
    fprod *= fft_f(in2, fsize).conjugate()

    if normalize is True:
        fprod = np.nan_to_num(fprod / np.absolute(fprod))

    ret = ifft_f(fprod).real.copy()

    return ret, fprod


def estimate_image_shift(ref, image, roi=None, sobel=True,
                         medfilter=True, hanning=True, plot=False,
                         dtype='float', normalize_corr=False,
                         sub_pixel_factor=1,
                         return_maxval=True):
    """Estimate the shift in a image using phase correlation

    This method can only estimate the shift by comparing
    bidimensional features that should not change the position
    in the given axis. To decrease the memory usage, the time of
    computation and the accuracy of the results it is convenient
    to select a region of interest by setting the roi keyword.

    Parameters
    ----------
    ref : 2D numpy.ndarray
        Reference image
    image : 2D numpy.ndarray
        Image to register
    roi : tuple of ints (top, bottom, left, right)
         Define the region of interest
    sobel : bool
        apply a sobel filter for edge enhancement
    medfilter :  bool
        apply a median filter for noise reduction
    hanning : bool
        Apply a 2d hanning filter
    plot : bool or matplotlib.Figure
        If True, plots the images after applying the filters and the phase
        correlation. If a figure instance, the images will be plotted to the
        given figure.
    reference : 'current' or 'cascade'
        If 'current' (default) the image at the current
        coordinates is taken as reference. If 'cascade' each image
        is aligned with the previous one.
    dtype : str or dtype
        Typecode or data-type in which the calculations must be
        performed.
    normalize_corr : bool
        If True use phase correlation instead of standard correlation
    sub_pixel_factor : float
        Estimate shifts with a sub-pixel accuracy of 1/sub_pixel_factor parts
        of a pixel. Default is 1, i.e. no sub-pixel accuracy.

    Returns
    -------
    shifts: np.array
        containing the estimate shifts
    max_value : float
        The maximum value of the correlation

    Notes
    -----
    The statistical analysis approach to the translation estimation
    when using reference='stat' roughly follows [*]_ . If you use
    it please cite their article.

    References
    ----------
    .. [*] Bernhard Schaffer, Werner Grogger and Gerald Kothleitner.
       “Automated Spatial Drift Correction for EFTEM Image Series.”
       Ultramicroscopy 102, no. 1 (December 2004): 27–36.

    """

    ref, image = da.compute(ref, image)
    # Make a copy of the images to avoid modifying them
    ref = ref.copy().astype(dtype)
    image = image.copy().astype(dtype)
    if roi is not None:
        top, bottom, left, right = roi
    else:
        top, bottom, left, right = [None, ] * 4

    # Select region of interest
    ref = ref[top:bottom, left:right]
    image = image[top:bottom, left:right]

    # Apply filters
    for im in (ref, image):
        if hanning is True:
            im *= hanning2d(*im.shape)
        if medfilter is True:
            # This is faster than sp.signal.med_filt,
            # which was the previous implementation.
            # The size is fixed at 3 to be consistent
            # with the previous implementation.
            im[:] = sp.ndimage.median_filter(im, size=3)
        if sobel is True:
            im[:] = sobel_filter(im)

    # If sub-pixel alignment not being done, use faster real-valued fft
    real_only = (sub_pixel_factor == 1)

    phase_correlation, image_product = fft_correlation(
        ref, image, normalize=normalize_corr, real_only=real_only)

    # Estimate the shift by getting the coordinates of the maximum
    argmax = np.unravel_index(np.argmax(phase_correlation),
                              phase_correlation.shape)
    threshold = (phase_correlation.shape[0] / 2 - 1,
                 phase_correlation.shape[1] / 2 - 1)
    shift0 = argmax[0] if argmax[0] < threshold[0] else  \
        argmax[0] - phase_correlation.shape[0]
    shift1 = argmax[1] if argmax[1] < threshold[1] else \
        argmax[1] - phase_correlation.shape[1]
    max_val = phase_correlation.real.max()
    shifts = np.array((shift0, shift1))

    # The following code is more or less copied from
    # skimage.feature.register_feature, to gain access to the maximum value:
    if sub_pixel_factor != 1:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * sub_pixel_factor) / sub_pixel_factor
        upsampled_region_size = np.ceil(sub_pixel_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        sub_pixel_factor = np.array(sub_pixel_factor, dtype=np.float64)
        normalization = (image_product.size * sub_pixel_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * sub_pixel_factor
        correlation = _upsampled_dft(image_product.conj(),
                                     upsampled_region_size,
                                     sub_pixel_factor,
                                     sample_region_offset).conj()
        correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
            np.argmax(np.abs(correlation)),
            correlation.shape),
            dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + maxima / sub_pixel_factor
        max_val = correlation.real.max()

    # Plot on demand
    if plot is True or isinstance(plot, plt.Figure):
        if isinstance(plot, plt.Figure):
            fig = plot
            axarr = plot.axes
            if len(axarr) < 3:
                for i in range(3):
                    fig.add_subplot(1, 3, i + 1)
                axarr = fig.axes
        else:
            fig, axarr = plt.subplots(1, 3)
        full_plot = len(axarr[0].images) == 0
        if full_plot:
            axarr[0].set_title('Reference')
            axarr[1].set_title('Image')
            axarr[2].set_title('Phase correlation')
            axarr[0].imshow(ref)
            axarr[1].imshow(image)
            d = (np.array(phase_correlation.shape) - 1) // 2
            extent = [-d[1], d[1], -d[0], d[0]]
            axarr[2].imshow(np.fft.fftshift(phase_correlation),
                            extent=extent)
            plt.show()
        else:
            axarr[0].images[0].set_data(ref)
            axarr[1].images[0].set_data(image)
            axarr[2].images[0].set_data(np.fft.fftshift(phase_correlation))
            # TODO: Renormalize images
            fig.canvas.draw_idle()
    # Liberate the memory. It is specially necessary if it is a
    # memory map
    del ref
    del image
    if return_maxval:
        return -shifts, max_val
    else:
        return -shifts


class Signal2D(BaseSignal, CommonSignal2D):

    """
    """
    _signal_dimension = 2
    _lazy = False

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if self.axes_manager.signal_dimension != 2:
            self.axes_manager.set_signal_dimension(2)

    def plot(self,
             navigator="auto",
             plot_markers=True,
             autoscale='v',
             saturated_pixels=None,
             norm="auto",
             vmin=None,
             vmax=None,
             gamma=1.0,
             linthresh=0.01,
             linscale=0.1,
             scalebar=True,
             scalebar_color="white",
             axes_ticks=None,
             axes_off=False,
             axes_manager=None,
             no_nans=False,
             colorbar=True,
             centre_colormap="auto",
             min_aspect=0.1,
             navigator_kwds={},
             **kwargs
             ):
        """%s
        %s
        %s
        %s

        """
        for c in autoscale:
            if c not in ['x', 'y', 'v']:
                raise ValueError("`autoscale` only accepts 'x', 'y', 'v' as "
                                 "valid characters.")
        super().plot(
            navigator=navigator,
            plot_markers=plot_markers,
            autoscale=autoscale,
            saturated_pixels=saturated_pixels,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            gamma=gamma,
            linthresh=linthresh,
            linscale=linscale,
            scalebar=scalebar,
            scalebar_color=scalebar_color,
            axes_ticks=axes_ticks,
            axes_off=axes_off,
            axes_manager=axes_manager,
            no_nans=no_nans,
            colorbar=colorbar,
            centre_colormap=centre_colormap,
            min_aspect=min_aspect,
            navigator_kwds=navigator_kwds,
            **kwargs
        )
    plot.__doc__ %= (BASE_PLOT_DOCSTRING, BASE_PLOT_DOCSTRING_PARAMETERS,
                     PLOT2D_DOCSTRING, PLOT2D_KWARGS_DOCSTRING)

    def create_model(self, dictionary=None):
        """Create a model for the current signal

        Parameters
        ----------
        dictionary : {None, dict}, optional
            A dictionary to be used to recreate a model. Usually generated
            using :meth:`hyperspy.model.as_dictionary`

        Returns
        -------
        A Model class

        """
        from hyperspy.models.model2d import Model2D
        return Model2D(self, dictionary=dictionary)

    def estimate_shift2D(self,
                         reference='current',
                         correlation_threshold=None,
                         chunk_size=30,
                         roi=None,
                         normalize_corr=False,
                         sobel=True,
                         medfilter=True,
                         hanning=True,
                         plot=False,
                         dtype='float',
                         show_progressbar=None,
                         sub_pixel_factor=1):
        """Estimate the shifts in an image using phase correlation.

        This method can only estimate the shift by comparing
        bi-dimensional features that should not change position
        between frames. To decrease the memory usage, the time of
        computation and the accuracy of the results it is convenient
        to select a region of interest by setting the ``roi`` argument.

        Parameters
        ----------
        reference : {'current', 'cascade' ,'stat'}
            If 'current' (default) the image at the current
            coordinates is taken as reference. If 'cascade' each image
            is aligned with the previous one. If 'stat' the translation
            of every image with all the rest is estimated and by
            performing statistical analysis on the result the
            translation is estimated.
        correlation_threshold : {None, 'auto', float}
            This parameter is only relevant when reference='stat'.
            If float, the shift estimations with a maximum correlation
            value lower than the given value are not used to compute
            the estimated shifts. If 'auto' the threshold is calculated
            automatically as the minimum maximum correlation value
            of the automatically selected reference image.
        chunk_size : {None, int}
            If int and reference='stat' the number of images used
            as reference are limited to the given value.
        roi : tuple of ints or floats (left, right, top, bottom)
            Define the region of interest. If int(float) the position
            is given axis index(value). Note that ROIs can be used
            in place of a tuple.
        normalize_corr : bool, default False
            If True, use phase correlation to align the images, otherwise
            use cross correlation.
        sobel : bool, default True
            Apply a Sobel filter for edge enhancement
        medfilter : bool, default True
            Apply a median filter for noise reduction
        hanning : bool, default True
            Apply a 2D hanning filter
        plot : bool or 'reuse'
            If True plots the images after applying the filters and
            the phase correlation. If 'reuse', it will also plot the images,
            but it will only use one figure, and continuously update the images
            in that figure as it progresses through the stack.
        dtype : str or dtype
            Typecode or data-type in which the calculations must be
            performed.
        %s
        sub_pixel_factor : float
            Estimate shifts with a sub-pixel accuracy of 1/sub_pixel_factor
            parts of a pixel. Default is 1, i.e. no sub-pixel accuracy.

        Returns
        -------
        shifts : list of array
            List of estimated shifts

        Notes
        -----
        The statistical analysis approach to the translation estimation
        when using ``reference='stat'`` roughly follows [Schaffer2004]_.
        If you use it please cite their article.

        References
        ----------
        .. [Schaffer2004] Schaffer, Bernhard, Werner Grogger, and Gerald Kothleitner.
           “Automated Spatial Drift Correction for EFTEM Image Series.”
           Ultramicroscopy 102, no. 1 (December 2004): 27–36.

        See Also
        --------
        * :py:meth:`~._signals.signal2d.Signal2D.align2D`

        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_two()
        if roi is not None:
            # Get the indices of the roi
            yaxis = self.axes_manager.signal_axes[1]
            xaxis = self.axes_manager.signal_axes[0]
            roi = tuple([xaxis._get_index(i) for i in roi[2:]] +
                        [yaxis._get_index(i) for i in roi[:2]])

        ref = None if reference == 'cascade' else \
            self.__call__().copy()
        shifts = []
        nrows = None
        images_number = self.axes_manager._max_index + 1
        if plot == 'reuse':
            # Reuse figure for plots
            plot = plt.figure()
        if reference == 'stat':
            nrows = images_number if chunk_size is None else \
                min(images_number, chunk_size)
            pcarray = ma.zeros((nrows, self.axes_manager._max_index + 1,
                                ),
                               dtype=np.dtype([('max_value', np.float),
                                               ('shift', np.int32,
                                                (2,))]))
            nshift, max_value = estimate_image_shift(
                self(),
                self(),
                roi=roi,
                sobel=sobel,
                medfilter=medfilter,
                hanning=hanning,
                normalize_corr=normalize_corr,
                plot=plot,
                dtype=dtype,
                sub_pixel_factor=sub_pixel_factor)
            np.fill_diagonal(pcarray['max_value'], max_value)
            pbar_max = nrows * images_number
        else:
            pbar_max = images_number

        # Main iteration loop. Fills the rows of pcarray when reference
        # is stat
        with progressbar(total=pbar_max,
                         disable=not show_progressbar,
                         leave=True) as pbar:
            for i1, im in enumerate(self._iterate_signal()):
                if reference in ['current', 'cascade']:
                    if ref is None:
                        ref = im.copy()
                        shift = np.array([0., 0.])
                    nshift, max_val = estimate_image_shift(
                        ref, im, roi=roi, sobel=sobel, medfilter=medfilter,
                        hanning=hanning, plot=plot,
                        normalize_corr=normalize_corr, dtype=dtype,
                        sub_pixel_factor=sub_pixel_factor)
                    if reference == 'cascade':
                        shift += nshift
                        ref = im.copy()
                    else:
                        shift = nshift
                    shifts.append(shift.copy())
                    pbar.update(1)
                elif reference == 'stat':
                    if i1 == nrows:
                        break
                    # Iterate to fill the columns of pcarray
                    for i2, im2 in enumerate(
                            self._iterate_signal()):
                        if i2 > i1:
                            nshift, max_value = estimate_image_shift(
                                im,
                                im2,
                                roi=roi,
                                sobel=sobel,
                                medfilter=medfilter,
                                hanning=hanning,
                                normalize_corr=normalize_corr,
                                plot=plot,
                                dtype=dtype,
                                sub_pixel_factor=sub_pixel_factor)
                            pcarray[i1, i2] = max_value, nshift
                        del im2
                        pbar.update(1)
                    del im
        if reference == 'stat':
            # Select the reference image as the one that has the
            # higher max_value in the row
            sqpcarr = pcarray[:, :nrows]
            sqpcarr['max_value'][:] = symmetrize(sqpcarr['max_value'])
            sqpcarr['shift'][:] = antisymmetrize(sqpcarr['shift'])
            ref_index = np.argmax(pcarray['max_value'].min(1))
            self.ref_index = ref_index
            shifts = (pcarray['shift'] +
                      pcarray['shift'][ref_index, :nrows][:, np.newaxis])
            if correlation_threshold is not None:
                if correlation_threshold == 'auto':
                    correlation_threshold = \
                        (pcarray['max_value'].min(0)).max()
                    _logger.info("Correlation threshold = %1.2f",
                                 correlation_threshold)
                shifts[pcarray['max_value'] <
                       correlation_threshold] = ma.masked
                shifts.mask[ref_index, :] = False

            shifts = shifts.mean(0)
        else:
            shifts = np.array(shifts)
            del ref
        return shifts

    estimate_shift2D.__doc__ %= SHOW_PROGRESSBAR_ARG

    def align2D(
        self,
        crop=True,
        fill_value=np.nan,
        shifts=None,
        expand=False,
        interpolation_order=1,
        show_progressbar=None,
        parallel=None,
        max_workers=None,
        **kwargs,
    ):
        """Align the images in-place using :py:func:`scipy.ndimage.shift`.

        The images can be aligned using either user-provided shifts or
        by first estimating the shifts.

        See :py:meth:`~._signals.signal2d.Signal2D.estimate_shift2D`
        for more details on estimating image shifts.

        Parameters
        ----------
        crop : bool
            If True, the data will be cropped not to include regions
            with missing data
        fill_value : int, float, nan
            The areas with missing data are filled with the given value.
            Default is nan.
        shifts : None or list of tuples
            If None the shifts are estimated using
            :py:meth:`~._signals.signal2D.estimate_shift2D`.
        expand : bool
            If True, the data will be expanded to fit all data after alignment.
            Overrides `crop`.
        interpolation_order: int, default 1.
            The order of the spline interpolation. Default is 1, linear
            interpolation.
        %s
        %s
        %s
        **kwargs :
            Keyword arguments passed to :py:meth:`~._signals.signal2d.Signal2D.estimate_shift2D`

        Returns
        -------
        shifts : np.array
            The estimated shifts are returned only if ``shifts`` is None

        See Also
        --------
        * :py:meth:`~._signals.signal2d.Signal2D.estimate_shift2D`

        """
        self._check_signal_dimension_equals_two()

        return_shifts = False

        if shifts is None:
            shifts = self.estimate_shift2D(**kwargs)
            return_shifts = True

            if not np.any(shifts):
                warnings.warn(
                    "The estimated shifts are all zero, suggesting "
                    "the images are already aligned",
                    UserWarning,
                )
                return shifts

        elif not np.any(shifts):
            warnings.warn(
                "The provided shifts are all zero, no alignment done",
                UserWarning,
            )
            return None

        if expand:
            # Expand to fit all valid data
            left, right = (
                int(np.floor(shifts[:, 1].min())) if shifts[:, 1].min() < 0 else 0,
                int(np.ceil(shifts[:, 1].max())) if shifts[:, 1].max() > 0 else 0,
            )
            top, bottom = (
                int(np.floor(shifts[:, 0].min())) if shifts[:, 0].min() < 0 else 0,
                int(np.ceil(shifts[:, 0].max())) if shifts[:, 0].max() > 0 else 0,
            )
            xaxis = self.axes_manager.signal_axes[0]
            yaxis = self.axes_manager.signal_axes[1]
            padding = []

            for i in range(self.data.ndim):
                if i == xaxis.index_in_array:
                    padding.append((right, -left))
                elif i == yaxis.index_in_array:
                    padding.append((bottom, -top))
                else:
                    padding.append((0, 0))

            self.data = np.pad(
                self.data, padding, mode="constant", constant_values=(fill_value,)
            )

            if left < 0:
                xaxis.offset += left * xaxis.scale
            if np.any((left < 0, right > 0)):
                xaxis.size += right - left
            if top < 0:
                yaxis.offset += top * yaxis.scale
            if np.any((top < 0, bottom > 0)):
                yaxis.size += bottom - top

        # Translate, with sub-pixel precision if necesary,
        # note that we operate in-place here
        self._map_iterate(
            shift_image,
            iterating_kwargs=(("shift", -shifts),),
            show_progressbar=show_progressbar,
            parallel=parallel,
            max_workers=max_workers,
            ragged=False,
            inplace=True,
            fill_value=fill_value,
            interpolation_order=interpolation_order,
        )

        if crop and not expand:
            max_shift = np.max(shifts, axis=0) - np.min(shifts, axis=0)

            if np.any(max_shift >= np.array(self.axes_manager.signal_shape)):
                raise ValueError("Shift outside range of signal axes. Cannot crop signal.")

            # Crop the image to the valid size
            shifts = -shifts
            bottom, top = (
                int(np.floor(shifts[:, 0].min())) if shifts[:, 0].min() < 0 else None,
                int(np.ceil(shifts[:, 0].max())) if shifts[:, 0].max() > 0 else 0,
            )
            right, left = (
                int(np.floor(shifts[:, 1].min())) if shifts[:, 1].min() < 0 else None,
                int(np.ceil(shifts[:, 1].max())) if shifts[:, 1].max() > 0 else 0,
            )
            self.crop_image(top, bottom, left, right)
            shifts = -shifts

        self.events.data_changed.trigger(obj=self)

        if return_shifts:
            return shifts

    align2D.__doc__ %= (SHOW_PROGRESSBAR_ARG, PARALLEL_ARG, MAX_WORKERS_ARG)

    def crop_image(self, top=None, bottom=None,
                   left=None, right=None, convert_units=False):
        """Crops an image in place.

        Parameters
        ----------
        top, bottom, left, right : {int | float}
            If int the values are taken as indices. If float the values are
            converted to indices.
        convert_units : bool
            Default is False
            If True, convert the signal units using the 'convert_to_units'
            method of the `axes_manager`. If False, does nothing.

        See also
        --------
        crop

        """
        self._check_signal_dimension_equals_two()
        self.crop(self.axes_manager.signal_axes[1].index_in_axes_manager,
                  top,
                  bottom)
        self.crop(self.axes_manager.signal_axes[0].index_in_axes_manager,
                  left,
                  right)
        if convert_units:
            self.axes_manager.convert_units('signal')

    def add_ramp(self, ramp_x, ramp_y, offset=0):
        """Add a linear ramp to the signal.

        Parameters
        ----------
        ramp_x: float
            Slope of the ramp in x-direction.
        ramp_y: float
            Slope of the ramp in y-direction.
        offset: float, optional
            Offset of the ramp at the signal fulcrum.

        Notes
        -----
            The fulcrum of the linear ramp is at the origin and the slopes are
            given in units of the axis with the according scale taken into
            account. Both are available via the `axes_manager` of the signal.

        """
        yy, xx = np.indices(self.axes_manager._signal_shape_in_array)
        if self._lazy:
            import dask.array as da
            ramp = offset * da.ones(self.data.shape, dtype=self.data.dtype,
                                    chunks=self.data.chunks)
        else:
            ramp = offset * np.ones(self.data.shape, dtype=self.data.dtype)
        ramp += ramp_x * xx
        ramp += ramp_y * yy
        self.data += ramp

    def find_peaks(self, method='local_max', interactive=True,
                   current_index=False, show_progressbar=None,
                   parallel=None, max_workers=None, display=True, toolkit=None,
                   **kwargs):
        """Find peaks in a 2D signal.

        Function to locate the positive peaks in an image using various, user
        specified, methods. Returns a structured array containing the peak
        positions.

        Parameters
        ----------
        method : str
             Select peak finding algorithm to implement. Available methods
             are:

             * 'local_max' - simple local maximum search using the
               :py:func:`skimage.feature.peak_local_max` function
             * 'max' - simple local maximum search using the
               :py:func:`~hyperspy.utils.peakfinders2D.find_peaks_max`.
             * 'minmax' - finds peaks by comparing maximum filter results
               with minimum filter, calculates centers of mass. See the
               :py:func:`~hyperspy.utils.peakfinders2D.find_peaks_minmax`
               function.
             * 'zaefferer' - based on gradient thresholding and refinement
               by local region of interest optimisation. See the
               :py:func:`~hyperspy.utils.peakfinders2D.find_peaks_zaefferer`
               function.
             * 'stat' - based on statistical refinement and difference with
               respect to mean intensity. See the
               :py:func:`~hyperspy.utils.peakfinders2D.find_peaks_stat`
               function.
             * 'laplacian_of_gaussian' - a blob finder using the laplacian of
               Gaussian matrices approach. See the
               :py:func:`~hyperspy.utils.peakfinders2D.find_peaks_log`
               function.
             * 'difference_of_gaussian' - a blob finder using the difference
               of Gaussian matrices approach. See the
               :py:func:`~hyperspy.utils.peakfinders2D.find_peaks_log`
               function.
             * 'template_matching' - A cross correlation peakfinder. This
               method requires providing a template with the ``template``
               parameter, which is used as reference pattern to perform the
               template matching to the signal. It uses the
               :py:func:`skimage.feature.match_template` function and the peaks
               position are obtained by using `minmax` method on the
               template matching result.

        interactive : bool
            If True, the method parameter can be adjusted interactively.
            If False, the results will be returned.
        current_index : bool
            if True, the computation will be performed for the current index.
        %s
        %s
        %s
        %s
        %s

        **kwargs : dict
            Keywords parameters associated with above methods, see the
            documentation of each method for more details.

        Notes
        -----
        As a convenience, the 'local_max' method accepts the 'distance' and
        'threshold' argument, which will be map to the 'min_distance' and
        'threshold_abs' of the :py:func:`skimage.feature.peak_local_max`
        function.

        Returns
        -------
        peaks : :py:class:`~hyperspy.signal.BaseSignal` or numpy.ndarray if current_index=True
            Array of shape `_navigation_shape_in_array` in which each cell
            contains an array with dimensions (npeaks, 2) that contains
            the `x, y` pixel coordinates of peaks found in each image sorted
            first along `y` and then along `x`.
        """
        method_dict = {
            'local_max': find_local_max,
            'max': find_peaks_max,
            'minmax': find_peaks_minmax,
            'zaefferer': find_peaks_zaefferer,
            'stat': find_peaks_stat,
            'laplacian_of_gaussian':  find_peaks_log,
            'difference_of_gaussian': find_peaks_dog,
            'template_matching' : find_peaks_xc,
        }
        # As a convenience, we map 'distance' to 'min_distance' and
        # 'threshold' to 'threshold_abs' when using the 'local_max' method to
        # match with the arguments of skimage.feature.peak_local_max.
        if method == 'local_max':
            if 'distance' in kwargs.keys():
                kwargs['min_distance'] = kwargs.pop('distance')
            if 'threshold' in kwargs.keys():
                kwargs['threshold_abs'] = kwargs.pop('threshold')
        if method in method_dict.keys():
            method_func = method_dict[method]
        else:
            raise NotImplementedError(f"The method `{method}` is not "
                                      "implemented. See documentation for "
                                      "available implementations.")
        if interactive:
            # Create a peaks signal with the same navigation shape as a
            # placeholder for the output
            axes_dict = self.axes_manager._get_axes_dicts(
                self.axes_manager.navigation_axes)
            peaks = BaseSignal(np.empty(self.axes_manager.navigation_shape),
                               axes=axes_dict)
            pf2D = PeaksFinder2D(self, method=method, peaks=peaks, **kwargs)
            pf2D.gui(display=display, toolkit=toolkit)
        elif current_index:
            peaks = method_func(self.__call__(), **kwargs)
        else:
            peaks = self.map(method_func, show_progressbar=show_progressbar,
                             parallel=parallel, inplace=False, ragged=True,
                             max_workers=max_workers, **kwargs)

        return peaks

    find_peaks.__doc__ %= (SHOW_PROGRESSBAR_ARG, PARALLEL_ARG, MAX_WORKERS_ARG,
                           DISPLAY_DT, TOOLKIT_DT)


class LazySignal2D(LazySignal, Signal2D):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
