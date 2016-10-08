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

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy as sp
import logging
from scipy.fftpack import fftn, ifftn

from hyperspy.defaults_parser import preferences
from hyperspy.external.progressbar import progressbar
from hyperspy.misc.math_tools import symmetrize, antisymmetrize
from hyperspy.signal import BaseSignal
from hyperspy._signals.common_signal2d import CommonSignal2D
from hyperspy.docstrings.plot import (
    BASE_PLOT_DOCSTRING, PLOT2D_DOCSTRING, KWARGS_DOCSTRING)


_logger = logging.getLogger(__name__)


def shift_image(im, shift, interpolation_order=1, fill_value=np.nan):
    fractional, integral = np.modf(shift)
    if fractional.any():
        order = interpolation_order
    else:
        # Disable interpolation
        order = 0
    im[:] = sp.ndimage.shift(im, shift, cval=fill_value, order=order)


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
    sx = sp.ndimage.sobel(im, axis=0, mode='constant')
    sy = sp.ndimage.sobel(im, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    return sob


def fft_correlation(in1, in2, normalize=False):
    """Correlation of two N-dimensional arrays using FFT.

    Adapted from scipy's fftconvolve.

    Parameters
    ----------
    in1, in2 : array
    normalize: bool
        If True performs phase correlation

    """
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    size = s1 + s2 - 1
    # Use 2**n-sized FFT
    fsize = 2 ** np.ceil(np.log2(size))
    IN1 = fftn(in1, fsize)
    IN1 *= fftn(in2, fsize).conjugate()
    if normalize is True:
        ret = ifftn(np.nan_to_num(IN1 / np.absolute(IN1))).real.copy()
    else:
        ret = ifftn(IN1).real.copy()
    del IN1
    return ret


def estimate_image_shift(ref, image, roi=None, sobel=True,
                         medfilter=True, hanning=True, plot=False,
                         dtype='float', normalize_corr=False,):
    """Estimate the shift in a image using phase correlation

    This method can only estimate the shift by comparing
    bidimensional features that should not change the position
    in the given axis. To decrease the memory usage, the time of
    computation and the accuracy of the results it is convenient
    to select a region of interest by setting the roi keyword.

    Parameters
    ----------

    roi : tuple of ints (top, bottom, left, right)
         Define the region of interest
    sobel : bool
        apply a sobel filter for edge enhancement
    medfilter :  bool
        apply a median filter for noise reduction
    hanning : bool
        Apply a 2d hanning filter
    plot : bool
        If True plots the images after applying the filters and
        the phase correlation
    reference : \'current\' | \'cascade\'
        If \'current\' (default) the image at the current
        coordinates is taken as reference. If \'cascade\' each image
        is aligned with the previous one.
    dtype : str or dtype
        Typecode or data-type in which the calculations must be
        performed.

    normalize_corr : bool
        If True use phase correlation instead of standard correlation

    Returns
    -------

    shifts: np.array
        containing the estimate shifts
    max_value : float
        The maximum value of the correlation

    """
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
            im[:] = sp.signal.medfilt(im)
        if sobel is True:
            im[:] = sobel_filter(im)

    phase_correlation = fft_correlation(ref, image,
                                        normalize=normalize_corr)

    # Estimate the shift by getting the coordinates of the maximum
    argmax = np.unravel_index(np.argmax(phase_correlation),
                              phase_correlation.shape)
    threshold = (phase_correlation.shape[0] / 2 - 1,
                 phase_correlation.shape[1] / 2 - 1)
    shift0 = argmax[0] if argmax[0] < threshold[0] else  \
        argmax[0] - phase_correlation.shape[0]
    shift1 = argmax[1] if argmax[1] < threshold[1] else \
        argmax[1] - phase_correlation.shape[1]
    max_val = phase_correlation.max()

    # Plot on demand
    if plot is True:
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(ref)
        axarr[1].imshow(image)
        axarr[2].imshow(phase_correlation)
        axarr[0].set_title('Reference')
        axarr[1].set_title('Signal2D')
        axarr[2].set_title('Phase correlation')
        plt.show()
    # Liberate the memory. It is specially necessary if it is a
    # memory map
    del ref
    del image

    return -np.array((shift0, shift1)), max_val


def get_largest_rectangle_from_rotation(width, height, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    degrees), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    from: http://stackoverflow.com/a/16778797/1018861
    In hyperspy, it is centered around centre coordinate of the signal.
    """
    import math
    angle = math.radians(angle)
    if width <= 0 or height <= 0:
        return 0, 0

    width_is_longer = width >= height
    side_long, side_short = (width, height) if width_is_longer else (height, width)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (width * cos_a - height * sin_a) / cos_2a, (height * cos_a - width * sin_a) / cos_2a
    return wr, hr


def get_signal_width_height(s):
    "Return pixel width and height of a signal"
    w = s.axes_manager[s.axes_manager.signal_indices_in_array[1]].size
    h = s.axes_manager[s.axes_manager.signal_indices_in_array[0]].size
    return (w, h)

class Signal2D(BaseSignal, CommonSignal2D):

    """
    """
    _signal_dimension = 2

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if self.axes_manager.signal_dimension != 2:
            self.axes_manager.set_signal_dimension(2)

    def plot(self,
             colorbar=True,
             scalebar=True,
             scalebar_color="white",
             axes_ticks=None,
             saturated_pixels=0,
             vmin=None,
             vmax=None,
             no_nans=False,
             centre_colormap="auto",
             **kwargs
             ):
        """%s
        %s
        %s

        """
        super(Signal2D, self).plot(
            colorbar=colorbar,
            scalebar=scalebar,
            scalebar_color=scalebar_color,
            axes_ticks=axes_ticks,
            saturated_pixels=saturated_pixels,
            vmin=vmin,
            vmax=vmax,
            no_nans=no_nans,
            centre_colormap=centre_colormap,
            **kwargs
        )
    plot.__doc__ %= BASE_PLOT_DOCSTRING, PLOT2D_DOCSTRING, KWARGS_DOCSTRING

    def create_model(self, dictionary=None):
        """Create a model for the current signal
        Parameters
        __________
        dictionary : {None, dict}, optional
            A dictionary to be used to recreate a model. Usually generated using
            :meth:`hyperspy.model.as_dictionary`
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
                         show_progressbar=None):
        """Estimate the shifts in a image using phase correlation
        This method can only estimate the shift by comparing
        bidimensional features that should not change position
        between frames. To decrease the memory usage, the time of
        computation and the accuracy of the results it is convenient
        to select a region of interest by setting the roi keyword.
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
            This parameter is only relevant when `reference` is 'stat'.
            If float, the shift estimations with a maximum correlation
            value lower than the given value are not used to compute
            the estimated shifts. If 'auto' the threshold is calculated
            automatically as the minimum maximum correlation value
            of the automatically selected reference image.
        chunk_size: {None, int}
            If int and `reference`=='stat' the number of images used
            as reference are limited to the given value.
        roi : tuple of ints or floats (left, right, top bottom)
             Define the region of interest. If int(float) the position
             is given axis index(value).
        sobel : bool
            apply a sobel filter for edge enhancement
        medfilter :  bool
            apply a median filter for noise reduction
        hanning : bool
            Apply a 2d hanning filter
        plot : bool
            If True plots the images after applying the filters and
            the phase correlation
        dtype : str or dtype
            Typecode or data-type in which the calculations must be
            performed.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.
        Returns
        -------
        list of applied shifts
        Notes
        -----
        The statistical analysis approach to the translation estimation
        when using `reference`='stat' roughly follows [1]_ . If you use
        it please cite their article.
        References
        ----------
        .. [1] Schaffer, Bernhard, Werner Grogger, and Gerald
        Kothleitner. “Automated Spatial Drift Correction for EFTEM
        Signal2D Series.”
        Ultramicroscopy 102, no. 1 (December 2004): 27–36.
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
                dtype=dtype)
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
                        shift = np.array([0, 0])
                    nshift, max_val = estimate_image_shift(
                        ref, im, roi=roi, sobel=sobel, medfilter=medfilter,
                        hanning=hanning, plot=plot,
                        normalize_corr=normalize_corr, dtype=dtype)
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
                                dtype=dtype)

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

    def align2D(self, crop=True, fill_value=np.nan, shifts=None, expand=False,
                roi=None,
                sobel=True,
                medfilter=True,
                hanning=True,
                plot=False,
                normalize_corr=False,
                reference='current',
                dtype='float',
                correlation_threshold=None,
                chunk_size=30,
                interpolation_order=1):
        """Align the images in place using user provided shifts or by
        estimating the shifts.
        Please, see `estimate_shift2D` docstring for details
        on the rest of the parameters not documented in the following
        section
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
            `estimate_shift2D`.
        expand : bool
            If True, the data will be expanded to fit all data after alignment.
            Overrides `crop`.
        interpolation_order: int, default 1.
            The order of the spline interpolation. Default is 1, linear
            interpolation.
        Returns
        -------
        shifts : np.array
            The shifts are returned only if `shifts` is None
        Notes
        -----
        The statistical analysis approach to the translation estimation
        when using `reference`='stat' roughly follows [1]_ . If you use
        it please cite their article.
        References
        ----------
        .. [1] Schaffer, Bernhard, Werner Grogger, and Gerald
        Kothleitner. “Automated Spatial Drift Correction for EFTEM
        Signal2D Series.”
        Ultramicroscopy 102, no. 1 (December 2004): 27–36.
        """
        self._check_signal_dimension_equals_two()
        if shifts is None:
            shifts = self.estimate_shift2D(
                roi=roi,
                sobel=sobel,
                medfilter=medfilter,
                hanning=hanning,
                plot=plot,
                reference=reference,
                dtype=dtype,
                correlation_threshold=correlation_threshold,
                normalize_corr=normalize_corr,
                chunk_size=chunk_size)
            return_shifts = True
        else:
            return_shifts = False
        if not np.any(shifts):
            # The shift array if filled with zeros, nothing to do.
            return

        if expand:
            # Expand to fit all valid data
            left, right = (int(np.floor(shifts[:, 1].min())) if
                           shifts[:, 1].min() < 0 else 0,
                           int(np.ceil(shifts[:, 1].max())) if
                           shifts[:, 1].max() > 0 else 0)
            top, bottom = (int(np.floor(shifts[:, 0].min())) if
                           shifts[:, 0].min() < 0 else 0,
                           int(np.ceil(shifts[:, 0].max())) if
                           shifts[:, 0].max() > 0 else 0)
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
            self.data = np.pad(self.data, padding, mode='constant',
                               constant_values=(fill_value,))
            if left < 0:
                xaxis.offset += left * xaxis.scale
            if np.any((left < 0, right > 0)):
                xaxis.size += right - left
            if top < 0:
                yaxis.offset += top * yaxis.scale
            if np.any((top < 0, bottom > 0)):
                yaxis.size += bottom - top

        # Translate with sub-pixel precision if necesary
        for im, shift in zip(self._iterate_signal(),
                             shifts):
            if np.any(shift):
                shift_image(im, -shift,
                            fill_value=fill_value,
                            interpolation_order=interpolation_order)
                del im

        if crop and not expand:
            # Crop the image to the valid size
            shifts = -shifts
            bottom, top = (int(np.floor(shifts[:, 0].min())) if
                           shifts[:, 0].min() < 0 else None,
                           int(np.ceil(shifts[:, 0].max())) if
                           shifts[:, 0].max() > 0 else 0)
            right, left = (int(np.floor(shifts[:, 1].min())) if
                           shifts[:, 1].min() < 0 else None,
                           int(np.ceil(shifts[:, 1].max())) if
                           shifts[:, 1].max() > 0 else 0)
            self.crop_image(top, bottom, left, right)
            shifts = -shifts

        self.events.data_changed.trigger(obj=self)
        if return_shifts:
            return shifts

    def crop_image(self, top=None, bottom=None,
                   left=None, right=None):
        """Crops an image in place.

        top, bottom, left, right : int or float

            If int the values are taken as indices. If float the values are
            converted to indices.

        See also:
        ---------
        crop

        """
        self._check_signal_dimension_equals_two()
        self.crop(self.axes_manager.signal_axes[1].index_in_axes_manager,
                  top,
                  bottom)
        self.crop(self.axes_manager.signal_axes[0].index_in_axes_manager,
                  left,
                  right)

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
            The fulcrum of the linear ramp is at the origin and the slopes are given in units of
            the axis with the according scale taken into account. Both are available via the
            `axes_manager` of the signal.

        """
        yy, xx = np.indices(self.axes_manager._signal_shape_in_array)
        ramp = offset * np.ones(self.data.shape, dtype=self.data.dtype)
        ramp += ramp_x * xx
        ramp += ramp_y * yy
        self.data += ramp



    def rotate(self, angle, reshape=False, crop=False, out=None,
               record=True, *args, **kwargs):
        """Rotates and interpolates the signal by an angle in degrees

        Parameters
        ----------
        angle : {int, float}
            In degrees, the angle by which the image shall be rotated anti-clockwise.
        reshape : bool [False]
            Increases the size of the signal (if necessary), to avoid cropping any of the signal.
        crop : bool [False]
            Crops the signal around its centre to its largest area without any black corners, based on
            a geometric calculation: http://stackoverflow.com/a/16778797/1018861
        out : To be filled by Dev
        record : To be filled by Dev (UI)

        See also
        --------
        Dev: Suggestions?

        Examples
        --------
        >>> # Rotate and crop an image to its largest area without black corners.
        >>> s = hs.signals.Signal2D(sc.misc.ascent())
        >>> s2 = s.rotate(angle=45, reshape=False, crop=True)
        >>> s2.plot()


        """
        import scipy.ndimage
        import math

        s2 = self.deepcopy()
        s2.map(scipy.ndimage.rotate, angle=angle, reshape=reshape)

        if crop == False:
            return s2
        elif crop == True:
            w, h = get_signal_width_height(s2)

            crop_w, crop_h = get_largest_rectangle_from_rotation(w, h, angle)
            crop_w, crop_h = math.floor(crop_w), math.floor(crop_h)
            center = (w / 2, h / 2)

            x1 = math.ceil(center[0] - crop_w / 2)
            x2 = math.floor(center[0] + crop_w / 2)
            y1 = math.ceil(center[1] - crop_h / 2)
            y2 = math.floor(center[1] + crop_h / 2)

            s2 = s2.isig[x1:x2, y1:y2]
            return s2