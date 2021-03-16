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

import copy

from numba import njit
import numpy as np
import scipy.ndimage as ndi
from skimage.feature import blob_dog, blob_log, match_template, peak_local_max

from hyperspy.misc.machine_learning import import_sklearn

NO_PEAKS = np.array([[np.nan, np.nan]])


@njit(cache=True)
def _fast_mean(X):  # pragma: no cover
    """JIT-compiled mean of array.

    Parameters
    ----------
    X : :py:class:`numpy.ndarray`
        Input array.

    Returns
    -------
    mean : float
        Mean of X.

    Notes
    -----
    Used by scipy.ndimage.generic_filter in the find_peaks_stat
    method to reduce overhead of repeated Python function calls.
    See https://github.com/scipy/scipy/issues/8916 for more details.
    """
    return np.mean(X)


@njit(cache=True)
def _fast_std(X):  # pragma: no cover
    """JIT-compiled standard deviation of array.

    Parameters
    ----------
    X : :py:class:`numpy.ndarray`
        Input array.

    Returns
    -------
    std : float
        Standard deviation of X.

    Notes
    -----
    Used by scipy.ndimage.generic_filter in the find_peaks_stat
    method to reduce overhead of repeated Python function calls.
    See https://github.com/scipy/scipy/issues/8916 for more details.
    """
    return np.std(X)


def clean_peaks(peaks):
    """Sort array of peaks and deal with no peaks being found.

    Parameters
    ----------
    peaks : :py:class:`numpy.ndarray`
        Array of found peaks.

    Returns
    -------
    peaks : :py:class:`numpy.ndarray`
        Sorted array, first by `peaks[:,1]` (y-coordinate) then by `peaks[:,0]`
        (x-coordinate), of found peaks.
    NO_PEAKS : str
        Flag indicating no peaks found.

    """
    if len(peaks) == 0:
        return NO_PEAKS
    else:
        ind = np.lexsort((peaks[:,0], peaks[:,1]))
        return peaks[ind]


def find_local_max(z, **kwargs):
    """Method to locate positive peaks in an image by local maximum searching.

    This function wraps :py:func:`skimage.feature.peak_local_max` function and
    sorts the results for consistency with other peak finding methods.

    Parameters
    ----------
    z : :py:class:`numpy.ndarray`
        Array of image intensities.
    **kwargs : dict
        Keyword arguments to be passed to the ``peak_local_max`` method of
        the ``scikit-image`` library. See its documentation for details
        http://scikit-image.org/docs/dev/api/skimage.feature.html#peak-local-max

    Returns
    -------
    peaks : :py:class:`numpy.ndarray` of shape (n_peaks, 2)
        Peak pixel coordinates.

    """
    peaks = peak_local_max(z, **kwargs)
    return clean_peaks(peaks)


def find_peaks_minmax(z, distance=5., threshold=10.):
    """Method to locate the positive peaks in an image by comparing maximum
    and minimum filtered images.

    Parameters
    ----------
    z : numpy.ndarray
        Matrix of image intensities.
    distance : float
        Expected distance between peaks.
    threshold : float
        Minimum difference between maximum and minimum filtered images.

    Returns
    -------
    peaks : :py:class:`numpy.ndarray` of shape (n_peaks, 2)
        Peak pixel coordinates.

    """
    data_max = ndi.filters.maximum_filter(z, distance)
    maxima = (z == data_max)
    data_min = ndi.filters.minimum_filter(z, distance)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndi.label(maxima)
    peaks = np.array(
        ndi.center_of_mass(z, labeled, range(1, num_objects + 1)))

    return clean_peaks(np.round(peaks).astype(int))


def find_peaks_max(z, alpha=3., distance=10):
    """Method to locate positive peaks in an image by local maximum searching.

    Parameters
    ----------
    alpha : float
        Only maxima above `alpha * sigma` are found, where `sigma` is the
        standard deviation of the image.
    distance : int
        When a peak is found, all pixels in a square region of side
        `2 * distance` are set to zero so that no further peaks can be found
        in that region.

    Returns
    -------
    peaks : :py:class:`numpy.ndarray` of shape (n_peaks, 2)
        Peak pixel coordinates.

    """
    # preallocate lots of peak storage
    k_arr = []
    # copy image
    image_temp = copy.deepcopy(z)
    peak_ct = 0
    # calculate standard deviation of image for thresholding
    sigma = np.std(z)
    while True:
        k = np.argmax(image_temp)
        j, i = np.unravel_index(k, image_temp.shape)
        if image_temp[j, i] >= alpha * sigma:
            k_arr.append([j, i])
            # masks peaks already identified.
            x = np.arange(i - distance, i + distance)
            y = np.arange(j - distance, j + distance)
            xv, yv = np.meshgrid(x, y)
            # clip to handle peaks near image edge
            image_temp[yv.clip(0, image_temp.shape[0] - 1),
                       xv.clip(0, image_temp.shape[1] - 1)] = 0
            peak_ct += 1
        else:
            break
    peaks = np.array(k_arr)
    return clean_peaks(peaks)


def find_peaks_zaefferer(z, grad_threshold=0.1, window_size=40,
                         distance_cutoff=50.):
    """Method to locate positive peaks in an image based on gradient
    thresholding and subsequent refinement within masked regions.

    Parameters
    ----------
    z : :py:class:`numpy.ndarray`
        Matrix of image intensities.
    grad_threshold : float
        The minimum gradient required to begin a peak search.
    window_size : int
        The size of the square window within which a peak search is
        conducted. If odd, will round down to even. The size must be larger
        than 2.
    distance_cutoff : float
        The maximum distance a peak may be from the initial
        high-gradient point.

    Returns
    -------
    peaks : :py:class:`numpy.ndarray` of shape (n_peaks, 2)
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
        return np.mgrid[x_min:x_max, y_min:y_max].reshape(2, -1, order="F")

    def get_max(image, box):
        """Finds the coordinates of the maximum of 'image' in 'box'."""
        vals = image[tuple(box)]
        ind = np.argmax(vals)
        return tuple(box[:, ind])

    def squared_distance(x, y):
        """Calculates the squared distance between two points."""
        return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2

    def gradient(image):
        """Calculates the square of the 2-d partial gradient.

        Parameters
        ----------
        image : :py:class:`numpy.ndarray`
            The image for which the gradient will be calculated.

        Returns
        -------
        gradient_of_image : :py:class:`numpy.ndarray`
            The gradient of the image.

        """
        gradient_of_image = np.gradient(image)
        gradient_of_image = gradient_of_image[0] ** 2 + gradient_of_image[
                                                            1] ** 2
        return gradient_of_image
    # Check window size is appropriate.
    if window_size < 2:
        raise ValueError("`window_size` must be >= 2.")
    # Generate an ordered list of matrix coordinates.
    if len(z.shape) != 2:
        raise ValueError("'z' should be a 2-d image matrix.")
    z = z / np.max(z)
    coordinates = np.indices(z.data.shape).reshape(2, -1).T
    # Calculate the gradient at every point.
    image_gradient = gradient(z)
    # Boolean matrix of high-gradient points.
    coordinates = coordinates[(image_gradient >= grad_threshold).flatten()]

    # Compare against squared distance (avoids repeated sqrt calls)
    distance_cutoff_sq = distance_cutoff ** 2

    peaks = []
    for coordinate in coordinates:
        # Iterate over coordinates where the gradient is high enough.
        b = box(coordinate[0], coordinate[1], window_size, z.shape[0],
                z.shape[1])
        p_old = (0, 0)
        p_new = get_max(z, b)

        while p_old[0] != p_new[0] and p_old[1] != p_new[1]:
            p_old = p_new
            b = box(p_old[0], p_old[1], window_size, z.shape[0], z.shape[1])
            p_new = get_max(z, b)
            if squared_distance(coordinate, p_new) > distance_cutoff_sq:
                break
            peaks.append(p_new)
    peaks = np.array([p for p in set(peaks)])
    return clean_peaks(peaks)


def find_peaks_stat(z, alpha=1.0, window_radius=10, convergence_ratio=0.05):
    """Method to locate positive peaks in an image based on statistical
    refinement and difference with respect to mean intensity.

    Parameters
    ----------
    z : :py:class:`numpy.ndarray`
        Array of image intensities.
    alpha : float
        Only maxima above `alpha * sigma` are found, where `sigma` is the
        local, rolling standard deviation of the image.
    window_radius : int
        The pixel radius of the circular window for the calculation of the
        rolling mean and standard deviation.
    convergence_ratio : float
        The algorithm will stop finding peaks when the proportion of new peaks
        being found is less than `convergence_ratio`.

    Returns
    -------
    peaks : :py:class:`numpy.ndarray` of shape (n_peaks, 2)
        Peak pixel coordinates.

    Notes
    -----
    Implemented as described in the PhD thesis of Thomas White, University of
    Cambridge, 2009, with minor modifications to resolve ambiguities.

    The algorithm is as follows:

    1. Adjust the contrast and intensity bias of the image so that all pixels
       have values between 0 and 1.
    2. For each pixel, determine the mean and standard deviation of all pixels
       inside a circle of radius 10 pixels centered on that pixel.
    3. If the value of the pixel is greater than the mean of the pixels in the
       circle by more than one standard deviation, set that pixel to have an
       intensity of 1. Otherwise, set the intensity to 0.
    4. Smooth the image by convovling it twice with a flat 3x3 kernel.
    5. Let k = (1/2 - mu)/sigma where mu and sigma are the mean and standard
       deviations of all the pixel intensities in the image.
    6. For each pixel in the image, if the value of the pixel is greater than
       mu + k*sigma set that pixel to have an intensity of 1. Otherwise, set the
       intensity to 0.
    7. Detect peaks in the image by locating the centers of gravity of regions
       of adjacent pixels with a value of 1.
    8. Repeat #4-7 until the number of peaks found in the previous step
       converges to within the user defined convergence_ratio.
    """
    if not import_sklearn.sklearn_installed:
        raise ImportError("This method requires scikit-learn.")

    def normalize(image):
        """Scales the image to intensities between 0 and 1."""
        return image / np.max(image)

    def _local_stat(image, radius, func):
        """Calculates rolling method 'func' over a circular kernel."""
        x, y = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        kernel = np.hypot(x, y) < radius
        stat = ndi.filters.generic_filter(image, func, footprint=kernel)
        return stat

    def local_mean(image, radius):
        """Calculates rolling mean over a circular kernel."""
        return _local_stat(image, radius, _fast_mean)

    def local_std(image, radius):
        """Calculates rolling standard deviation over a circular kernel."""
        return _local_stat(image, radius, _fast_std)

    def single_pixel_desensitize(image):
        """Reduces single-pixel anomalies by nearest-neighbor smoothing."""
        kernel = np.array([[0.5, 1, 0.5], [1, 1, 1], [0.5, 1, 0.5]])
        smoothed_image = ndi.filters.generic_filter(image, _fast_mean, footprint=kernel)
        return smoothed_image

    def stat_binarise(image):
        """Peaks more than one standard deviation from the mean set to one."""
        image_rolling_mean = local_mean(image, window_radius)
        image_rolling_std = local_std(image, window_radius)
        image = single_pixel_desensitize(image)
        binarised_image = np.zeros(image.shape)
        stat_mask = image > (image_rolling_mean + alpha * image_rolling_std)
        binarised_image[stat_mask] = 1
        return binarised_image

    def smooth(image):
        """Image convolved twice using a uniform 3x3 kernel."""
        image = ndi.filters.uniform_filter(image, size=3)
        image = ndi.filters.uniform_filter(image, size=3)
        return image

    def half_binarise(image):
        """Image binarised about values of one-half intensity."""
        binarised_image = np.where(image > 0.5, 1, 0)
        return binarised_image

    def separate_peaks(binarised_image):
        """Identify adjacent 'on' coordinates via DBSCAN."""
        bi = binarised_image.astype("bool")
        coordinates = np.indices(bi.shape).reshape(2, -1).T[bi.flatten()]
        db = import_sklearn.sklearn.cluster.DBSCAN(2, min_samples=3)
        peaks = []
        if coordinates.shape[0] > 0:  # we have at least some peaks
            labeled_points = db.fit_predict(coordinates)
            for peak_label in list(set(labeled_points)):
                peaks.append(coordinates[labeled_points == peak_label])
        return peaks

    def _peak_find_once(image):
        """Smooth, binarise, and find peaks according to main algorithm."""
        image = smooth(image)  # 4
        image = half_binarise(image)  # 5
        peaks = separate_peaks(image)  # 6
        centers = np.array([np.mean(peak, axis=0) for peak in peaks])  # 7
        return image, centers

    def stat_peak_finder(image, convergence_ratio):
        """Find peaks in image. Algorithm stages in comments."""
        # Image preparation
        image = normalize(image)  # 1
        image = stat_binarise(image)  # 2, 3
        # Perform first iteration of peak finding
        image, peaks_curr = _peak_find_once(image)  # 4-7
        n_peaks = len(peaks_curr)
        if n_peaks == 0:
            return peaks_curr

        m_peaks = 0
        # Repeat peak finding with more blurring to convergence
        while (n_peaks - m_peaks) / n_peaks > convergence_ratio:  # 8
            m_peaks = n_peaks
            peaks_old = np.copy(peaks_curr)
            image, peaks_curr = _peak_find_once(image)
            n_peaks = len(peaks_curr)
            if n_peaks == 0:
                return peaks_old

        return peaks_curr

    return clean_peaks(stat_peak_finder(z, convergence_ratio))


def find_peaks_dog(z, min_sigma=1., max_sigma=50., sigma_ratio=1.6,
                   threshold=0.2, overlap=0.5, exclude_border=False):
    """Method to locate peaks via the Difference of Gaussian Matrices method.

    This function wraps :py:func:`skimage.feature.blob_dog` function and
    sorts the results for consistency with other peak finding methods.

    Parameters
    ----------
    z : :py:class:`numpy.ndarray`
        2-d array of intensities
    min_sigma, max_sigma, sigma_ratio, threshold, overlap, exclude_border :
        Additional parameters to be passed to the algorithm. See `blob_dog`
        documentation for details:
        http://scikit-image.org/docs/dev/api/skimage.feature.html#blob-dog

    Returns
    -------
    peaks : :py:class:`numpy.ndarray` of shape (n_peaks, 2)
        Peak pixel coordinates.

    Notes
    -----
    While highly effective at finding even very faint peaks, this method is
        sensitive to fluctuations in intensity near the edges of the image.

    """
    z = z / np.max(z)
    blobs = blob_dog(z, min_sigma=min_sigma, max_sigma=max_sigma,
                     sigma_ratio=sigma_ratio, threshold=threshold,
                     overlap=overlap, exclude_border=exclude_border)
    try:
        centers = np.round(blobs[:, :2]).astype(int)
    except IndexError:
        return NO_PEAKS
    clean_centers = []
    for center in centers:
        if len(np.intersect1d(center, (0, 1) + z.shape + tuple(
                        c - 1 for c in z.shape))) > 0:
            continue
        clean_centers.append(center)
    return clean_peaks(np.array(clean_centers))


def find_peaks_log(z, min_sigma=1., max_sigma=50., num_sigma=10,
                   threshold=0.2, overlap=0.5, log_scale=False,
                   exclude_border=False):
    """Method to locate peaks via the Laplacian of Gaussian Matrices method.

    This function wraps :py:func:`skimage.feature.blob_log` function and
    sorts the results for consistency with other peak finding methods.

    Parameters
    ----------
    z : :py:class:`numpy.ndarray`
        Array of image intensities.
    min_sigma, max_sigma, num_sigma, threshold, overlap, log_scale, exclude_border :
        Additional parameters to be passed to the ``blob_log`` method of the
        ``scikit-image`` library. See its documentation for details:
        http://scikit-image.org/docs/dev/api/skimage.feature.html#blob-log

    Returns
    -------
    peaks : :py:class:`numpy.ndarray` of shape (n_peaks, 2)
        Peak pixel coordinates.

    """
    z = z / np.max(z)
    if isinstance(num_sigma, float):
        raise ValueError("`num_sigma` parameter should be an integer.")
    blobs = blob_log(z, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=num_sigma, threshold=threshold, overlap=overlap,
                     log_scale=log_scale, exclude_border=exclude_border)
    # Attempt to return only peak positions. If no peaks exist, return an
    # empty array.
    try:
        centers = np.round(blobs[:, :2]).astype(int)
        ind = np.lexsort((centers[:,0], centers[:,1]))
    except IndexError:
        return NO_PEAKS
    return centers[ind]


def find_peaks_xc(z, template, distance=5, threshold=0.5, **kwargs):
    """Find peaks in the cross correlation between the image and a template by
    using the :py:func:`~hyperspy.utils.peakfinders2D.find_peaks_minmax` function
    to find the peaks on the cross correlation result obtained using the
    :py:func:`skimage.feature.match_template` function.

    Parameters
    ----------
    z : :py:class:`numpy.ndarray`
        Array of image intensities.
    template : numpy.ndarray (square)
        Array containing a single bright disc, similar to those to detect.
    distance : float
        Expected distance between peaks.
    threshold : float
        Minimum difference between maximum and minimum filtered images.
    **kwargs : dict
        Keyword arguments to be passed to the
        :py:func:`skimage.feature.match_template` function.

    Returns
    -------
    peaks : :py:class:`numpy.ndarray` of shape (n_peaks, 2)
        Array of peak coordinates.
    """
    pad_input = kwargs.pop('pad_input', True)
    response_image = match_template(z, template, pad_input=pad_input, **kwargs)
    peaks = find_peaks_minmax(response_image,
                              distance=distance,
                              threshold=threshold)

    return clean_peaks(peaks)
