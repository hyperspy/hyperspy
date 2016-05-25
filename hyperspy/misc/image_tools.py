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

import numpy as np
import scipy as sp
import copy
from scipy.fftpack import fftn, ifftn
import scipy.ndimage as ndi
from hyperspy.misc.spectrum_tools import find_peaks_ohaver
import matplotlib.pyplot as plt


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
    sx = sp.ndimage.sobel(im, axis=0)
    sy = sp.ndimage.sobel(im, axis=1)
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
                         dtype='float', normalize_corr=False, ):
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
        if medfilter is True:
            im[:] = sp.signal.medfilt(im)
        if sobel is True:
            im[:] = sobel_filter(im)
        if hanning is True:
            im *= hanning2d(*im.shape)

    phase_correlation = fft_correlation(ref, image,
                                        normalize=normalize_corr)

    # Estimate the shift by getting the coordinates of the maximum
    argmax = np.unravel_index(np.argmax(phase_correlation),
                              phase_correlation.shape)
    threshold = (phase_correlation.shape[0] / 2 - 1,
                 phase_correlation.shape[1] / 2 - 1)
    shift0 = argmax[0] if argmax[0] < threshold[0] else \
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
        axarr[1].set_title('Image')
        axarr[2].set_title('Phase correlation')
        plt.show()
    # Liberate the memory. It is specially necessary if it is a
    # memory map
    del ref
    del image

    return -np.array((shift0, shift1)), max_val


def contrast_stretching(data, saturated_pixels):
    """Calculate bounds that leaves out a given percentage of the data.

    Parameters
    ----------
    data: numpy array
    saturated_pixels: scalar
        The percentage of pixels that are left out of the bounds.  For example,
        the low and high bounds of a value of 1 are the 0.5% and 99.5%
        percentiles. It must be in the [0, 100] range.

    Returns
    -------
    vmin, vmax: scalar
        The low and high bounds

    Raises
    ------
    ValueError if the value of `saturated_pixels` is out of the valid range.

    """
    # Sanity check
    if not 0 <= saturated_pixels <= 100:
        raise ValueError(
            "saturated_pixels must be a scalar in the range[0, 100]")
    nans = np.isnan(data)
    if nans.any():
        data = data[~nans]
    vmin = np.percentile(data, saturated_pixels / 2.)
    vmax = np.percentile(data, 100 - saturated_pixels / 2.)
    return vmin, vmax


MPL_DIVERGING_COLORMAPS = [
    "BrBG",
    "bwr",
    "coolwarm",
    "PiYG",
    "PRGn",
    "PuOr",
    "RdBu",
    "RdGy",
    "RdYIBu",
    "RdYIGn",
    "seismic",
    "Spectral", ]
# Add reversed colormaps
MPL_DIVERGING_COLORMAPS += [cmap + "_r" for cmap in MPL_DIVERGING_COLORMAPS]


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


def find_peaks_blob(z, threshold=5., **kwargs):
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
    blobs = blob_dog(z, threshold=threshold, **kwargs)
    return blobs[:, :2]


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


def centre_colormap_values(vmin, vmax):
    """Calculate vmin and vmax to set the colormap midpoint to zero.

    Parameters
    ----------
    vmin, vmax : scalar
        The range of data to display.

    Returns
    -------
    cvmin, cvmax : scalar
        The values to obtain a centre colormap.

    """

    absmax = max(abs(vmin), abs(vmax))
    return -absmax, absmax
