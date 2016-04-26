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
    peaks = np.array(ndi.center_of_mass(z, labeled, range(1, num_objects+1)))

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
        if(image_temp[j, i] >= alpha * sigma):
            k_arr[peak_ct] = [j, i]
            # masks peaks already identified.
            x = np.arange(i-size, i+size)
            y = np.arange(j-size, j+size)
            xv, yv = np.meshgrid(x, y)
            # clip to handle peaks near image edge
            image_temp[yv.clip(0, image_temp.shape[0]-1),
                       xv.clip(0, image_temp.shape[1]-1)] = 0
            peak_ct += 1
        else:
            break
    # trim array to have shape (number of peaks, 2)
    peaks = k_arr[:peak_ct]

    return peaks


def find_peaks_zaefferer(z, grad_threshold=10, separation):
    """
    Method to locate positive peaks in an image based on gradient thresholding
    and subsequent refinement within masked regions.

    Parameters
    ----------

    grad_threshold : int

    separation : int

    Returns
    -------

    peaks : array

    Notes
    -----
    Implemented as described in Zaefferer "New developments of computer-aided
    crystallographic analysis in transmission electron microscopy" J. Ap. Cryst.
    """


def find_peaks_stat(z):
    """
    Method to locate positive peaks in an image based on statistical refinement
    and difference with respect to mean intensity.

    Parameters
    ----------

    Returns
    -------

    Notes
    -----
    Implemented as described in the PhD thesis of Thomas White (2009) the
    algorithm was developed by Gordon Ball during a summer project in Cambridge.
    """


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
                            peak_array=peak_array).copy()[:, 0] for i in xrange(z.shape[0])]
    for row in xrange(len(xc)):
        for col in xrange(xc[row].shape[0]):
            mapX[row, int(xc[row][col])] = 1
    yc = [find_peaks_ohaver(z[:, i], medfilt_radius=None,
                            peakgroup=peak_width,
                            subchannel=False,
                            peak_array=peak_array).copy()[:, 0] for i in xrange(z.shape[1])]
    for col in xrange(len(yc)):
        for row in xrange(yc[col].shape[0]):
            mapY[int(yc[col][row]), col] = 1

    Fmap = mapX*mapY
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


def subpix_locate(z, peaks, peak_width, scale=None):
    top = left = peak_width / 2 + 1
    centers = np.array(peaks, dtype=np.float32)
    for i in xrange(peaks.shape[0]):
        pk = peaks[i]
        center = np.array(ndi.measurements.center_of_mass(z[(pk[0] - left):(pk[0] + left),
                                                            (pk[1] - top):(pk[1] + top)]))
        center = center[0] - peak_width / 2, center[1] - peak_width / 2
        centers[i] = np.array([pk[0] + center[0], pk[1]+center[1]])
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
