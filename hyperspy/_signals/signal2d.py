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

import copy
import numpy as np
import numpy.ma as ma
import scipy as sp
from scipy.fftpack import fftn, ifftn
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import warnings

from hyperspy.defaults_parser import preferences
from hyperspy.external.progressbar import progressbar
from hyperspy.misc.math_tools import symmetrize, antisymmetrize
from hyperspy.signal import BaseSignal


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

def find_peaks_minmax(z, separation, threshold, interpolation_order=3):
    """
    Method to locate the positive peaks in an image by comparing maximum
    and minimum filtered images.
    Parameters
    ----------
    z: image
    separation: expected distance between peaks
    threshold:
    interpolation order:
    Returns
    -------
    peaks: array with dimensions (npeaks, 2) that contains the x, y coordinates
           for each peak found in the image.
    """
    data_max = ndi.filters.maximum_filter(z, separation)
    maxima = (z == data_max)
    data_min = ndi.filters.minimum_filter(z, separation)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndi.label(maxima)
    peaks = np.array(ndi.center_of_mass(z, labeled, range(1, num_objects + 1)))

    return peaks


def find_peaks_max(z, alpha=3, size=10):
    """
    Method to locate positive peaks in an image by simple local maximum
    searching.
    """
    # preallocate lots of peak storage
    k_arr = np.zeros((10000, 2))
    # copy image
    image_temp = copy.deepcopy(z)
    peak_ct = 0
    # calculate standard deviation of image for thresholding
    sigma = np.std(z)
    while True:
        k = np.argmax(image_temp)
        j, i = np.unravel_index(k, image_temp.shape)
        if (image_temp[j, i] >= alpha * sigma):
            k_arr[peak_ct] = [j, i]
            # masks peaks already identified.
            x = np.arange(i - size, i + size)
            y = np.arange(j - size, j + size)
            xv, yv = np.meshgrid(x, y)
            # clip to handle peaks near image edge
            image_temp[yv.clip(0, image_temp.shape[0] - 1),
                       xv.clip(0, image_temp.shape[1] - 1)] = 0
            peak_ct += 1
        else:
            break
    # trim array to have shape (number of peaks, 2)
    peaks = k_arr[:peak_ct]
    return peaks

def find_peaks_zaefferer(z, grad_threshold=400, window_size=40,
                         distance_cutoff=50):
    """
    Method to locate positive peaks in an image based on gradient thresholding
    and subsequent refinement within masked regions.
    Parameters
    ----------
    z : ndarray
        Matrix of image intensities.
    grad_threshold : float
        The minimum gradient required to begin a peak search.
    window_size : int
        The size of the square window within which a peak search is
        conducted. If odd, will round down to even.
    distance_cutoff : int
        The maximum distance a peak may be from the initial high-gradient point.
    Returns
    -------
    peaks : numpy.ndarray
        (n_peaks, 2)
        Peak pixel coordinates.
    Notes
    -----
    Implemented as described in Zaefferer "New developments of computer-aided
    crystallographic analysis in transmission electron microscopy" J. Ap. Cryst.
    This version by Ben Martineau (2016)
    """

    def box(x, y, window_size, x_max, y_max):
        """Produces a list of coordinates in the box about (x, y)."""
        a = int(window_size / 2)
        x_min = max(0, x - a)
        x_max = min(x_max, x + a)
        y_min = max(0, y - a)
        y_max = min(y_max, y + a)
        return np.array(
            np.meshgrid(range(x_min, x_max), range(y_min, y_max))).T.reshape(-1,
                                                                             2)

    def get_max(image, box):
        """Finds the coordinates of the maximum of 'image' in 'box'."""
        vals = image[box[:, 1], box[:, 0]]
        max_position = box[np.argmax(vals)]
        return max_position

    def distance(x, y):
        """Calculates the distance between two points."""
        v = x - y
        return np.sqrt(np.sum(np.square(v)))

    def gradient(image):
        """Calculates the square of the 2-d partial gradient."""
        image_gradient = np.gradient(image)
        image_gradient = image_gradient[0] ** 2 + image_gradient[1] ** 2
        return image_gradient

    # Generate an ordered list of matrix coordinates.
    if len(z.shape) != 2:
        raise ValueError("'z' should be a 2-d image matrix.")
    coordinates = np.indices(z.data.shape).T.reshape(-1, 2)
    # Calculate the gradient at every point.
    image_gradient = gradient(z)
    # Boolean matrix of high-gradient points.
    gradient_is_above_threshold = image_gradient >= grad_threshold
    peaks = []
    for coordinate in coordinates[gradient_is_above_threshold.flatten()]:
        # Iterate over coordinates where the gradient is high enough.
        b = box(coordinate[0], coordinate[1], window_size, z.shape[0],
                z.shape[1])
        p_old = np.array([0, 0])
        p_new = get_max(z, b)
        while np.all(p_old != p_new):
            p_old = p_new
            b = box(p_old[0], p_old[1], window_size, z.shape[0], z.shape[1])
            p_new = get_max(z, b)
        if distance(coordinate, p_new) <= distance_cutoff:
            peaks.append(tuple(p_new))
    peaks = np.array([np.array(p) for p in set(peaks)])
    return peaks

def find_peaks_stat(z):
    """
    Method to locate positive peaks in an image based on statistical refinement
    and difference with respect to mean intensity.
    Parameters
    ----------
    z : ndarray
        Array of image intensities.
    Returns
    -------
    ndarray
        (n_peaks, 2)
        Array of peak coordinates.
    Notes
    -----
    Implemented as described in the PhD thesis of Thomas White (2009) the
    algorithm was developed by Gordon Ball during a summer project in
    Cambridge.
    This version by Ben Martineau (2016), with minor modifications to the
    original where methods were ambiguous or unclear.
    """
    from scipy.ndimage.filters import generic_filter
    from scipy.ndimage.filters import uniform_filter
    from sklearn.cluster import DBSCAN

    def normalize(image):
        """Scales the image to intensities between 0 and 1."""
        return image/np.max(image)

    def _local_stat(image, radius, func):
        """Calculates rolling method 'func' over a circular kernel."""
        x, y = np.ogrid[-radius:radius+1, -radius:radius+1]
        kernel = x**2 + y**2 <= radius**2
        stat = generic_filter(image, func, footprint=kernel)
        return stat

    def local_mean(image, radius):
        """Calculates rolling mean over a circular kernel."""
        return _local_stat(image, radius, np.mean)

    def local_std(image, radius):
        """Calculates rolling standard deviation over a circular kernel."""
        return _local_stat(image, radius, np.std)

    def single_pixel_desensitize(image):
        """Reduces single-pixel anomalies by nearest-neighbor smoothing."""
        kernel = np.array([[0.5, 1, 0.5], [1, 1, 1], [0.5, 1, 0.5]])
        smoothed_image = generic_filter(image, np.mean, footprint=kernel)
        return smoothed_image

    def stat_binarise(image):
        """Peaks more than one standard deviation from the mean set to one."""
        image_rolling_mean = local_mean(image, 10)
        image_rolling_std = local_std(image, 10)
        image = single_pixel_desensitize(image)
        binarised_image = np.zeros(image.shape)
        binarised_image[image > image_rolling_mean + image_rolling_std] = 1
        return binarised_image

    def smooth(image):
        """Image convolved twice using a uniform 3x3 kernel."""
        image = uniform_filter(image, size=3)
        image = uniform_filter(image, size=3)
        return image

    def half_binarise(image):
        """Image binarised about values of one-half intensity."""
        binarised_image = np.zeros(image.shape)
        binarised_image[image > 0.5] = 1
        return binarised_image

    def separate_peaks(binarised_image):
        """Identify adjacent 'on' coordinates via DBSCAN."""
        bi = binarised_image.astype('bool')
        coordinates = np.indices(bi.data.shape).T.reshape(-1, 2)[bi.flatten()]
        db = DBSCAN(2, 3)
        peaks = []
        labeled_points = db.fit_predict(coordinates)
        for peak_label in list(set(labeled_points)):
            peaks.append(coordinates[labeled_points==peak_label])
        return peaks

    def _peak_find_once(image):
        """Smooth, binarise, and find peaks according to main algorithm."""
        image = smooth(image)
        image = half_binarise(image)
        peaks = separate_peaks(image)
        return image, peaks

    def stat_peak_finder(image):
        """Find peaks in diffraction image. Algorithm stages in comments."""
        image = normalize(image)  # 1
        image = stat_binarise(image)  # 2, 3
        n_peaks = np.infty  # Initial number of peaks
        image, peaks = _peak_find_once(image)  # 4-6
        m_peaks = len(peaks)  # Actual number of peaks
        while (n_peaks - m_peaks)/n_peaks > 0.05:  # 8
            n_peaks = m_peaks
            image, peaks = _peak_find_once(image)
            m_peaks = len(peaks)
        peak_centers = np.array([np.mean(peak, axis=0) for peak in peaks])  # 7
        return peak_centers

    return stat_peak_finder(z)


def find_peaks_masiel(z, subpixel=False, peak_width=10, medfilt_radius=5,
                      maxpeakn=10000):
    """
    Method to locate peaks in an image by finding peaks in the  x-direction and
    y-direction separately and then determining common position.
    Parameters
    ----------
    arr : array
    2D input array, i.e. an image
    medfilt_radius : int (optional)
                     median filter window to apply to smooth the data
                     (see scipy.signal.medfilter). If 0, no filter applied.
                     Default value is 5.
    peak_width : int (optional)
                 expected peak width. Affects subpixel precision fitting window,
                 which takes the center of gravity of a box that has sides equal
                 to this parameter. If the value is too big other peaks will be
                 included.
                 Default value is 10.
    subpixel : bool (optional)
               Default is False.
    Returns
    -------
    peaks : array of shape (npeaks, 3)
            contains position and height of each peak
    Notes
    -----
    Based on matlab function from Dan Masiel and originally coded in python by
    Michael Sarahan (2011).
    Developed in to this version by Duncan Johnstone (2016)
    """
    mapX = np.zeros_like(z)
    mapY = np.zeros_like(z)
    peak_array = np.zeros((maxpeakn, 3))

    if medfilt_radius > 0:
        z = ndi.filters.median_filter(z, medfilt_radius)
    xc = [find_peaks_ohaver(z[i], medfilt_radius=None,
                            peakgroup=peak_width,
                            subchannel=False,
                            peak_array=peak_array).copy()[:, 0] for i in
          xrange(z.shape[0])]
    for row in xrange(len(xc)):
        for col in xrange(xc[row].shape[0]):
            mapX[row, int(xc[row][col])] = 1
    yc = [find_peaks_ohaver(z[:, i], medfilt_radius=None,
                            peakgroup=peak_width,
                            subchannel=False,
                            peak_array=peak_array).copy()[:, 0] for i in
          xrange(z.shape[1])]
    for col in xrange(len(yc)):
        for row in xrange(yc[col].shape[0]):
            mapY[int(yc[col][row]), col] = 1

    Fmap = mapX * mapY
    nonzeros = np.nonzero(Fmap)
    peaks = np.vstack((nonzeros[1], nonzeros[0])).T
    if subpixel:
        peaks = subpix_locate(z, peaks, peak_width)
    peaks = np.ma.fix_invalid(peaks, fill_value=-1)
    peaks = np.ma.masked_outside(peaks, peak_width / 2 + 1,
                                 z.shape[0] - peak_width / 2 - 1)
    peaks = np.ma.masked_less(peaks, 0)
    peaks = np.ma.compress_rows(peaks)
    # add the heights
    # heights = np.array([z[peaks[i, 1], peaks[i, 0]] for i in xrange(peaks.shape[0])]).reshape((-1, 1))
    # peaks = np.hstack((peaks, heights))
    return peaks


def find_peaks_blob(z, **kwargs):
    """
    Finds peaks via the difference of Gaussian Matrices method in scikit-image.

    Parameters
    ----------
    z : ndarray
        Array of image intensities.
    threshold : Minimum cut-off value for peak detection. May be considerably
        lower than minimum peak intensity.
    kwargs : Additional parameters to be passed to the algorithm. See 'blob_dog'
        documentation for details.

    Returns
    -------
    ndarray
        (n_peaks, 2)
        Array of peak coordinates.

    Notes
    -----
    While highly effective at finding even very faint peaks, this method is
        sensitive to fluctuations in intensity near the edges of the image.

    """
    from skimage.feature import blob_dog
    blobs = blob_dog(z, **kwargs)
    centers = blobs[:, :2]
    clean_centers = []
    for center in centers:
        if len(np.intersect1d(center, (0,) + z.shape + tuple(
                        c - 1 for c in z.shape))) > 0:
            continue
        clean_centers.append(center)
    return np.array(clean_centers)


def subpix_locate(z, peaks, peak_width, scale=None):
    top = left = peak_width / 2 + 1
    centers = np.array(peaks, dtype=np.float32)
    for i in xrange(peaks.shape[0]):
        pk = peaks[i]
        center = np.array(
            ndi.measurements.center_of_mass(z[(pk[0] - left):(pk[0] + left),
                                            (pk[1] - top):(pk[1] + top)]))
        center = center[0] - peak_width / 2, center[1] - peak_width / 2
        centers[i] = np.array([pk[0] + center[0], pk[1] + center[1]])
    if scale:
        centers = centers * scale
    return centers

class Signal2DTools(object):

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

    def find_peaks2D(self, method='skimage', *args, **kwargs):
        """Find peaks in a 2D signal/image.
        Function to locate the positive peaks in an image using various, user
        specified, methods. Returns a structured array containing the peak
        positions.
        Parameters
        ---------
        method : str
                 Select peak finding algorithm to implement. Available methods
                 are:
                     'max' - simple local maximum search
                     'skimage' - call the peak finder implemented in
                                 scikit-image which uses a maximum filter
                     'minmax' - finds peaks by comparing maximum filter results
                                with minimum filter, calculates centers of mass
                     'zaefferer' - based on gradient thresholding and refinement
                                   by local region of interest optimisation
                     'stat' - statistical approach requiring no free params.
                     'massiel' - finds peaks in each direction and compares the
                                 positions where these coincide.
                     'blob' - a blob finder implemented in scikit-image which
                              uses the difference of Gaussian matrices approach
        keywords : associated with above methods.
        Returns
        -------
        peaks: structured array of shape _navigation_shape_in_array in which
               each cell contains an array with dimensions (npeaks, 2) that
               contains the x, y coordinates of peaks found in each image.
        """
        arr_shape = (self.axes_manager._navigation_shape_in_array
                     if self.axes_manager.navigation_size > 0
                     else [1, ])
        peaks = np.zeros(arr_shape, dtype=object)
        for z, indices in zip(self._iterate_signal(),
                              self.axes_manager._array_indices_generator()):
            if method == 'skimage':
                peaks[indices] = peak_local_max(z, *args, **kwargs)
            if method == 'max':
                peaks[indices] = find_peaks_max(z, *args, **kwargs)
            if method == 'minmax':
                peaks[indices] = find_peaks_minmax(z, *args, **kwargs)
            if method == 'zaefferer':
                peaks[indices] = find_peaks_zaefferer(z, *args, **kwargs)
            if method == 'stat':
                peaks[indices] = find_peaks_stat(z, *args, **kwargs)
            if method == 'massiel':
                peaks[indices] = find_peaks_masiel(z, *args, **kwargs)
            if method == 'blob':
                peaks[indices] = find_peaks_blob(z, *args, **kwargs)

        return peaks

class Signal2D(BaseSignal,
               Signal2DTools,):

    """
    """
    _record_by = "image"

    def __init__(self, *args, **kw):
        super(Signal2D, self).__init__(*args, **kw)
        if self.metadata._HyperSpy.Folding.signal_unfolded:
            self.axes_manager.set_signal_dimension(1)
        else:
            self.axes_manager.set_signal_dimension(2)

    def to_signal1D(self):
        """Returns the image as a spectrum.

        See Also
        --------
        as_signal1D : a method for the same purpose with more options.
        signals.Signal1D.to_signal1D : performs the inverse operation on one
        dimensional signals.

        """
        return self.as_signal1D(0 + 3j)

    def plot(self,
             colorbar=True,
             scalebar=True,
             scalebar_color="white",
             axes_ticks=None,
             auto_contrast=True,
             saturated_pixels=0,
             vmin=None,
             vmax=None,
             no_nans=False,
             centre_colormap="auto",
             **kwargs
             ):
        """Plot image.

        For multidimensional datasets an optional figure,
        the "navigator", with a cursor to navigate that data is
        raised. In any case it is possible to navigate the data using
        the sliders. Currently only signals with signal_dimension equal to
        0, 1 and 2 can be plotted.

        Parameters
        ----------
        navigator : {"auto", None, "slider", "spectrum", Signal}
            If "auto", if navigation_dimension > 0, a navigator is
            provided to explore the data.
            If navigation_dimension is 1 and the signal is an image
            the navigator is a spectrum obtained by integrating
            over the signal axes (the image).
            If navigation_dimension is 1 and the signal is a spectrum
            the navigator is an image obtained by stacking horizontally
            all the spectra in the dataset.
            If navigation_dimension is > 1, the navigator is an image
            obtained by integrating the data over the signal axes.
            Additionaly, if navigation_dimension > 2 a window
            with one slider per axis is raised to navigate the data.
            For example,
            if the dataset consists of 3 navigation axes X, Y, Z and one
            signal axis, E, the default navigator will be an image
            obtained by integrating the data over E at the current Z
            index and a window with sliders for the X, Y and Z axes
            will be raised. Notice that changing the Z-axis index
            changes the navigator in this case.
            If "slider" and the navigation dimension > 0 a window
            with one slider per axis is raised to navigate the data.
            If "spectrum" and navigation_dimension > 0 the navigator
            is always a spectrum obtained by integrating the data
            over all other axes.
            If None, no navigator will be provided.
            Alternatively a Signal instance can be provided. The signal
            dimension must be 1 (for a spectrum navigator) or 2 (for a
            image navigator) and navigation_shape must be 0 (for a static
            navigator) or navigation_shape + signal_shape must be equal
            to the navigator_shape of the current object (for a dynamic
            navigator).
            If the signal dtype is RGB or RGBA this parameters has no
            effect and is always "slider".
        axes_manager : {None, axes_manager}
            If None `axes_manager` is used.
        colorbar : bool, optional
             If true, a colorbar is plotted for non-RGB images.
        scalebar : bool, optional
            If True and the units and scale of the x and y axes are the same a
            scale bar is plotted.
        scalebar_color : str, optional
            A valid MPL color string; will be used as the scalebar color.
        axes_ticks : {None, bool}, optional
            If True, plot the axes ticks. If None axes_ticks are only
            plotted when the scale bar is not plotted. If False the axes ticks
            are never plotted.
        auto_contrast : bool, optional
            If True, the contrast is stretched for each image using the
            `saturated_pixels` value. Default True.
        saturated_pixels: scalar
            The percentage of pixels that are left out of the bounds.
            For example, the low and high bounds of a value of 1 are the 0.5%
            and 99.5% percentiles. It must be in the [0, 100] range.
        vmin, vmax : scalar, optional
            `vmin` and `vmax` are used to normalize luminance data. If at
            least one of them is given `auto_contrast` is set to False and any
            missing values are calculated automatically.
        no_nans : bool, optional
            If True, set nans to zero for plotting.
        centre_colormap : {"auto", True, False}
            If True the centre of the color scheme is set to zero. This is
            specially useful when using diverging color schemes. If "auto"
            (default), diverging color schemes are automatically centred.
        **kwargs, optional
            Additional key word arguments passed to matplotlib.imshow()

        """
        super(Signal2D, self).plot(
            colorbar=colorbar,
            scalebar=scalebar,
            scalebar_color=scalebar_color,
            axes_ticks=axes_ticks,
            auto_contrast=auto_contrast,
            saturated_pixels=saturated_pixels,
            vmin=vmin,
            vmax=vmax,
            no_nans=no_nans,
            centre_colormap=centre_colormap,
            **kwargs
        )

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
