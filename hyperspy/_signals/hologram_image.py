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
from hyperspy.signals import Signal2D
from collections import OrderedDict
import logging

_logger = logging.getLogger(__name__)


class HologramImage(Signal2D):
    """Image subclass for holograms acquired via electron holography."""

    _signal_type = 'hologram'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling = (self.axes_manager[0].scale, self.axes_manager[1].scale)
        self.f_sampling = np.divide(1, [a * b for a, b in zip(self.data.shape, self.sampling)])

    def reconstruct_phase(self, reference=None, sb_size=None, sb_smooth=None, sb_unit=None,
                          sb='lower', sb_pos=None, output_shape=None, plotting=False):
        """Reconstruct electron holograms.

        Parameters
        ----------
        reference : ndarray, :class:`~hyperspy.signals.Signal2D
            Vacuum reference hologram.
        sb_size : float
            Sideband radius of the aperture in corresponding unit (see 'sb_unit'). If None,
            the radius of the aperture is set to 1/3 of the distance between sideband and
            centerband.
        sb_smooth : float, optional
            Smoothness of the aperture in the same unit as sb_size.
        sb_unit : str, optional
            Unit of the two sideband parameters 'sb_size' and 'sb_smoothness'.
            Default: None - Sideband size given in pixels
            'nm': Size and smoothness of the aperture are given in 1/nm.
            'mrad': Size and smoothness of the aperture are given in mrad.
        sb : str, optional
            Select which sideband is selected. 'upper' or 'lower'.
        sb_pos : tuple, optional
            Sideband position in pixel. If None, sideband is determined automatically from FFT.
        output_shape: tuple, optional
            Choose a new output shape. Default is the shape of the input hologram. The output
            shape should not be larger than the input shape.
        plotting : boolean
            Shows details of the reconstruction (i.e. SB selection).

        Returns
        -------
        wave : :class:`~hyperspy.signals.WaveImage
            Reconstructed electron wave. By default object wave is devided by reference wave

        Notes
        -----
        Use wave.rec_param to extract reconstruction parameters, which can be used for batch
        processing.
        """

        ref_data = None

        # Parsing reference input:
        if reference is not None:
            if isinstance(reference, Signal2D):
                ref_data = reference.data
            else:
                ref_data = reference

        fft_holo = fft2(self.data) / np.prod(self.data.shape)

        # Find sideband position
        if sb_pos is None:
            if reference is None:
                sb_pos = self.find_sideband_position(self.data, self.sampling, sb=sb)
            else:
                sb_pos = self.find_sideband_position(ref_data, self.sampling, sb=sb)
        else:
            sb_pos = sb_pos

        if sb_size is None:
            sb_size = np.linalg.norm(sb_pos) / 3  # in pixels

        # Convert sideband sie from 1/nm or mrad to pixels
        if sb_unit == 'nm':
            sb_size /= np.mean(self.f_sampling)
            sb_smooth /= np.mean(self.f_sampling)
        elif sb_unit == 'mrad':
            try:
                ht = self.metadata.Acquisition_instrument.TEM.beam_energy
            except AttributeError:
                ht = int(input('Enter beam energy in kV: '))
            wavelength = 1.239842447 / np.sqrt(ht * (1022 + ht))  # in nm
            sb_size /= (1000 * wavelength * np.mean(self.f_sampling))
            sb_smooth /= (1000 * wavelength * np.mean(self.f_sampling))

        # Standard edge smoothness of sideband aperture 5% of sb_size
        if sb_smooth is None:
            sb_smooth = 0.05 * sb_size

        # Reconstruction parameters are stored in rec_param
        rec_param = [sb_pos, sb_size, sb_smooth]

        # ???
        _logger.info('Sideband pos in pixels: {}'.format(sb_pos))
        _logger.info('Sideband aperture radius in pixels: {}'.format(sb_size))
        _logger.info('Sideband aperture smoothness in pixels: {}'.format(sb_smooth))

        # Shows the selected sideband and the position of the sideband
        if plotting:
            fig, axs = plt.subplots(1, 1, figsize=(4, 4))
            axs.imshow(np.abs(fft_holo), clim=(0, 2.2))
            axs.scatter(sb_pos[1], sb_pos[0], s=20, color='red', marker='x')
            axs.set_xlim(sb_pos[1]-sb_size, sb_pos[1]+sb_size)
            axs.set_ylim(sb_pos[0]-sb_size, sb_pos[0]+sb_size)
            plt.show()

        # Reconstruction
        if reference is None:
            w_ref = 1
        else:
            # reference electron wave
            w_ref = self._reconstruct(holo_data=ref_data, holo_sampling=self.sampling,
                                      sb_size=sb_size, sb_pos=sb_pos, sb_smoothness=sb_smooth,
                                      output_shape=output_shape, plotting=plotting)
        # object wave
        w_obj = self._reconstruct(holo_data=self.data, holo_sampling=self.sampling,
                                  sb_size=sb_size, sb_pos=sb_pos, sb_smoothness=sb_smooth,
                                  output_shape=output_shape, plotting=plotting)

        wave = w_obj / w_ref
        wave_image = self._deepcopy_with_new_data(wave)
        wave_image.set_signal_type('wave')  # New signal is a wave image!
        rec_param_dict = OrderedDict([('sb_pos', rec_param[0]), ('sb_size', rec_param[1]),
                                      ('sb_smoothness', rec_param[2])])

        wave_image.metadata.Signal.add_node('holo_rec_param')
        wave_image.metadata.Signal.holo_rec_param.add_dictionary(rec_param_dict)

        wave_image.axes_manager[0].size = wave.data.shape[0]
        wave_image.axes_manager[1].size = wave.data.shape[1]
        wave_image.axes_manager[0].scale = self.sampling[0] * self.data.shape[0] / \
                                           wave.data.shape[0]
        wave_image.axes_manager[1].scale = self.sampling[1] * self.data.shape[1] / \
                                           wave.data.shape[1]

        return wave_image

    @staticmethod
    def _reconstruct(holo_data, holo_sampling, sb_size, sb_pos, sb_smoothness, output_shape=None,
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
        sb_pos : tuple
            Sideband position in pixel.
        sb_smoothness: float
            Smoothness of the aperture in pixel.
        output_shape: tuple, optional
            New output shape.
        plotting : boolean
            Shows details of the reconstruction (i.e. SB selection).

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

        fft_shifted = np.roll(fft_exp, sb_pos[0], axis=0)
        fft_shifted = np.roll(fft_shifted, sb_pos[1], axis=1)

        if plotting:
            fig, axs = plt.subplots(1, 1, figsize=(4, 4))
            axs.imshow(np.abs(fftshift(fft_shifted * aperture)), clim=(0, 0.1))
            axs.scatter(sb_pos[1], sb_pos[0], s=10, color='red', marker='x')
            axs.set_xlim(int(holo_size[0]/2) - sb_size/np.mean(f_sampling), int(holo_size[0]/2) +
                         sb_size/np.mean(f_sampling))
            axs.set_ylim(int(holo_size[1]/2) - sb_size/np.mean(f_sampling), int(holo_size[1]/2) +
                         sb_size/np.mean(f_sampling))
            plt.show()

        fft_aperture = fft_shifted * aperture

        if output_shape is not None:
            y_min = int(holo_size[0] / 2 - output_shape[0] / 2)
            y_max = int(holo_size[0] / 2 + output_shape[0] / 2)
            x_min = int(holo_size[1] / 2 - output_shape[1] / 2)
            x_max = int(holo_size[1] / 2 + output_shape[1] / 2)

            fft_aperture = fftshift(fftshift(fft_aperture)[y_min:y_max, x_min:x_max])

        wav = ifft2(fft_aperture) * np.prod(holo_data.shape)

        return wav

    @staticmethod
    def find_sideband_position(holo_data, holo_sampling, ap_cb_radius=None, sb='lower'):
        """
        Finds the position of the sideband and returns its position.

        Parameters
        ----------
        holo_data: ndarray
            The data of the hologram.
        holo_sampling: tuple
            The sampling rate in both image directions.
        ap_cb_radius: float, optional
            The aperture radius used to mask out the centerband.
        sb : str, optional
            Chooses which sideband is taken. 'lower' or 'upper'

        Returns
        -------
        Tuple of the sideband position (y, x), referred to the unshifted FFT.
        """

        sb_pos = (0, 0)

        f_freq = freq_array(holo_data.shape, holo_sampling)

        # If aperture radius of centerband is not given, it will be set to 5 % of the Nyquist
        # frequency.
        if ap_cb_radius is None:
            ap_cb_radius = 1 / 20. * np.max(f_freq)

        # A small aperture masking out the centerband.
        ap_cb = np.subtract(1, aperture_function(f_freq, ap_cb_radius, 0))

        fft_holo = fft2(holo_data) / np.prod(holo_data.shape)
        fft_filtered = fft_holo * ap_cb

        # Sideband position in pixels referred to unshifted FFT
        if sb == 'lower':
            fft_sb = fft_filtered[:int(fft_filtered.shape[0] / 2), :]
            sb_pos = tuple(np.unravel_index(fft_sb.argmax(), fft_sb.shape))
        elif sb == 'upper':
            fft_sb = fft_filtered[int(fft_filtered.shape[0] / 2):, :]
            sb_pos = tuple(np.unravel_index(fft_sb.argmax(), fft_sb.shape))
            sb_pos = np.add(sb_pos, (int(fft_filtered.shape[0] / 2), 0))

        return sb_pos


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
