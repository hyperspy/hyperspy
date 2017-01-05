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
from scipy.fftpack import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
import logging

_logger = logging.getLogger(__name__)


def estimate_sideband_position(holo_data, holo_sampling, central_band_mask_radius=None, sb='lower'):
    """
    Finds the position of the sideband and returns its position.

    Parameters
    ----------
    holo_data: ndarray
        The data of the hologram.
    holo_sampling: tuple
        The sampling rate in both image directions.
    central_band_mask_radius: float, optional
        The aperture radius used to mask out the centerband.
    sb : str, optional
        Chooses which sideband is taken. 'lower' or 'upper'

    Returns
    -------
    Tuple of the sideband position (y, x), referred to the unshifted FFT.
    """

    sb_position = (0, 0)

    f_freq = freq_array(holo_data.shape, holo_sampling)

    # If aperture radius of centerband is not given, it will be set to 5 % of the Nyquist
    # frequency.
    if central_band_mask_radius is None:
        central_band_mask_radius = 1 / 20. * np.max(f_freq)

    # A small aperture masking out the centerband.
    aperture_central_band = np.subtract(1.0, aperture_function(f_freq, central_band_mask_radius, 1e-6))  # 1e-6
    # imitates 0

    fft_holo = fft2(holo_data) / np.prod(holo_data.shape)
    fft_filtered = fft_holo * aperture_central_band

    # Sideband position in pixels referred to unshifted FFT
    if sb == 'lower':
        fft_sb = fft_filtered[:int(fft_filtered.shape[0] / 2), :]
        sb_position = np.asarray(np.unravel_index(fft_sb.argmax(), fft_sb.shape))
    elif sb == 'upper':
        fft_sb = fft_filtered[int(fft_filtered.shape[0] / 2):, :]
        sb_position = (np.unravel_index(fft_sb.argmax(), fft_sb.shape))
        sb_position = np.asarray(np.add(sb_position, (int(fft_filtered.shape[0] / 2), 0)))

    return sb_position


def estimate_sideband_size(sb_position, holo_shape, sb_size_ratio=0.5):
    """
    Estimates the size of sideband filter

    Parameters
    ----------
    holo_shape : array_like
            Holographic data array
    sb_position : tuple
        The sideband position (y, x), referred to the non-shifted FFT.
    sb_size_ratio : float, optional
        Size of sideband as a fraction of the distance to central band

    Returns
    -------
    sb_size : float
        Size of sideband filter

    """

    h = np.array((np.asarray(sb_position) - np.asarray([0, 0]),
                  np.asarray(sb_position) - np.asarray([0, holo_shape[1]]),
                  np.asarray(sb_position) - np.asarray([holo_shape[0], 0]),
                  np.asarray(sb_position) - np.asarray(holo_shape))) * sb_size_ratio
    return np.min(np.linalg.norm(h, axis=1))


def reconstruct(holo_data, holo_sampling, sb_size, sb_position, sb_smoothness, output_shape=None,
                plotting=False):
    """Core function for holographic reconstruction.

    Parameters
    ----------
    holo_data : array_like
        Holographic data array
    holo_sampling : tuple
        Sampling rate of the hologram in y and x direction.
    sb_size : float
        Size of the sideband filter in pixel.
    sb_position : tuple
        Sideband position in pixel.
    sb_smoothness: float
        Smoothness of the aperture in pixel.
    output_shape: tuple, optional
        New output shape.
    plotting : boolean
        Plots the masked sideband used for reconstruction.

    Returns
    -------
        wav : nparray
            Reconstructed electron wave

    """

    holo_size = holo_data.shape
    f_sampling = np.divide(1, [a * b for a, b in zip(holo_size, holo_sampling)])

    fft_exp = fft2(holo_data) / np.prod(holo_size)

    f_freq = freq_array(holo_data.shape, holo_sampling)

    sb_size *= np.mean(f_sampling)
    sb_smoothness *= np.mean(f_sampling)
    aperture = aperture_function(f_freq, sb_size, sb_smoothness)

    fft_shifted = np.roll(fft_exp, sb_position[0], axis=0)
    fft_shifted = np.roll(fft_shifted, sb_position[1], axis=1)

    fft_aperture = fft_shifted * aperture

    if plotting:
        _, axs = plt.subplots(1, 1, figsize=(4, 4))
        axs.imshow(np.abs(fftshift(fft_aperture)), clim=(0, 0.1))
        axs.scatter(sb_position[1], sb_position[0], s=10, color='red', marker='x')
        axs.set_xlim(int(holo_size[0]/2) - sb_size/np.mean(f_sampling), int(holo_size[0]/2) +
                     sb_size/np.mean(f_sampling))
        axs.set_ylim(int(holo_size[1]/2) - sb_size/np.mean(f_sampling), int(holo_size[1]/2) +
                     sb_size/np.mean(f_sampling))
        plt.show()

    if output_shape is not None:
        y_min = int(holo_size[0] / 2 - output_shape[0] / 2)
        y_max = int(holo_size[0] / 2 + output_shape[0] / 2)
        x_min = int(holo_size[1] / 2 - output_shape[1] / 2)
        x_max = int(holo_size[1] / 2 + output_shape[1] / 2)

        fft_aperture = fftshift(fftshift(fft_aperture)[y_min:y_max, x_min:x_max])

    wav = ifft2(fft_aperture) * np.prod(holo_data.shape)

    return wav


def aperture_function(r, apradius, rsmooth):
    """
    A smooth aperture function that decays from apradius-rsmooth to apradius+rsmooth.

    Parameters
    ----------
    r : ndarray
        Array of input data (e.g. frequencies)
    apradius : float
        Radius (center) of the smooth aperture. Decay starts at apradius - rsmooth.
    rsmooth : float
        Smoothness in halfwidth. rsmooth = 1 will cause a decay from 1 to 0 over 2 pixel.
    """

    return 0.5 * (1. - np.tanh((np.absolute(r) - apradius) / (0.5 * rsmooth)))


def freq_array(shape, sampling):
    """
    Makes up a frequency array.

    Parameters
    ----------
    shape : tuple
        The shape of the array.
    sampling: tuple
        The sampling rates of the array.

    Returns
    -------
    Array of the frequencies.
    """
    f_freq_1d_y = np.fft.fftfreq(shape[0], sampling[0])
    f_freq_1d_x = np.fft.fftfreq(shape[1], sampling[1])
    f_freq_mesh = np.meshgrid(f_freq_1d_x, f_freq_1d_y)
    f_freq = np.hypot(f_freq_mesh[0], f_freq_mesh[1])

    return f_freq
