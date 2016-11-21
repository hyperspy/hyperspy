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
from hyperspy.signals import Signal2D, BaseSignal
from collections import OrderedDict
from hyperspy.misc.holography.reconstruct import reconstruct, find_sideband_position, find_sideband_size
import logging
import warnings

_logger = logging.getLogger(__name__)


class HologramImage(Signal2D):
    """Image subclass for holograms acquired via electron holography."""

    _signal_type = 'hologram'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling = (self.axes_manager[0].scale, self.axes_manager[1].scale)
        self.f_sampling = np.divide(1, [a * b for a, b in zip(self.data.shape, self.sampling)])

    def find_sideband_position(self, ap_cb_radius=None, sb='lower'):
        """
        Finds the position of the sideband and returns its position.

        Parameters
        ----------
        ap_cb_radius: float, optional
            The aperture radius used to mask out the centerband.
        sb : str, optional
            Chooses which sideband is taken. 'lower' or 'upper'

        Returns
        -------
        Tuple of the sideband position (y, x), referred to the unshifted FFT.
        """

        if self.axes_manager.navigation_size:
            self.unfold_navigation_space()
            sb_position = np.zeros((self.axes_manager.navigation_size, 2), dtype='int64')
            for i in range(self.axes_manager.navigation_size):
                sb_position[i] = find_sideband_position(self.inav[i].data, self.sampling, ap_cb_radius, sb)
            self.fold()
        else:
            sb_position = find_sideband_position(self.data, self.sampling, ap_cb_radius, sb)
        return sb_position

    def find_sideband_size(self, sb_position):
        """
        Finds the size of the sideband and returns its position.

        Parameters
        ----------
        sb_position : array, :class:`~hyperspy.signals.BaseSignal
            The sideband position (y, x), referred to the non-shifted FFT.

        Returns
        -------
        Sideband size (y, x), referred to the unshifted FFT.
        """
        if isinstance(sb_position, BaseSignal):
            sb_position = sb_position.data

        if self.axes_manager.navigation_size:
            sb_size = np.zeros(self.axes_manager.navigation_size)

            for i in range(self.axes_manager.navigation_size):
                sb_size[i] = find_sideband_size(self.axes_manager.signal_shape, sb_position[i])

        else:
            sb_size = find_sideband_size(self.axes_manager.signal_shape, sb_position)

        return sb_size

    def reconstruct_phase(self, reference=None, sb_size=None, sb_smoothness=None, sb_unit=None,
                          sb='lower', sb_position=None, output_shape=None, plotting=False):
        """Reconstruct electron holograms.

        Parameters
        ----------
        reference : ndarray, :class:`~hyperspy.signals.Signal2D
            Vacuum reference hologram.
        sb_size : float
            Sideband radius of the aperture in corresponding unit (see 'sb_unit'). If None,
            the radius of the aperture is set to 1/3 of the distance between sideband and
            centerband.
        sb_smoothness : float, optional
            Smoothness of the aperture in the same unit as sb_size.
        sb_unit : str, optional
            Unit of the two sideband parameters 'sb_size' and 'sb_smoothness'.
            Default: None - Sideband size given in pixels
            'nm': Size and smoothness of the aperture are given in 1/nm.
            'mrad': Size and smoothness of the aperture are given in mrad.
        sb : str, optional
            Select which sideband is selected. 'upper' or 'lower'.
        sb_position : tuple, optional
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

        # Unfolding self:

        # Parsing reference:
        if not isinstance(reference, HologramImage):
            if isinstance(reference, Signal2D):
                _logger.warning('The reference image signal type is not HologramImage. It will '
                                'be converted to HologramImage automatically.')
                reference.set_signal_type('hologram')
            elif reference is not None:
                reference = HologramImage(reference)

        # Parsing sideband position:
        if sb_position is None:
            warnings.warn('Sideband position is not specified. The sideband will be found automatically which may '
                          'cause wrong results.')
            if reference is None:
                sb_position = self.find_sideband_position(sb=sb)
            else:
                sb_position = reference.find_sideband_position(sb=sb)

        if self.axes_manager.navigation_size:
            if sb_position.ndim == 1:
                sb_position = np.stack([sb_position]*self.axes_manager.navigation_size)

            elif sb_position.ndim == 2 and sb_position.shape[0] != self.axes_manager.navigation_size:
                warnings.warn('The sideband position size does not match the size of the hologram image! '
                              'The first values of sideband position will be used to reconstruct all slices'
                              'of hologram image.')
                sb_position = np.stack([sb_position[0]]*self.axes_manager.navigation_size)

        if sb_size is None:  # Default value is 1/2 distance between sideband and central band
            sb_size = self.find_sideband_size(sb_position)

        folded = self.unfold_navigation_space()

        # Standard edge smoothness of sideband aperture 5% of sb_size
        if sb_smoothness is None:
            sb_smoothness = sb_size * 0.05

        # Convert sideband size from 1/nm or mrad to pixels
        if sb_unit == 'nm':
            sb_size /= np.mean(self.f_sampling)
            sb_smoothness /= np.mean(self.f_sampling)
        elif sb_unit == 'mrad':
            try:
                ht = self.metadata.Acquisition_instrument.TEM.beam_energy
            except AttributeError:
                ht = int(input('Enter beam energy in kV: '))
                self.metadata.add_node('Acquisition_instrument')
                self.metadata.Acquisition_instrument.add_node('TEM')
                self.metadata.Acquisition_instrument.TEM.add_node('beam_energy')
                self.metadata.Acquisition_instrument.TEM.beam_energy = ht
            wavelength = 1.239842447 / np.sqrt(ht * (1022 + ht))  # in nm
            sb_size /= (1000 * wavelength * np.mean(self.f_sampling))
            sb_smoothness /= (1000 * wavelength * np.mean(self.f_sampling))

        # Find output shape:
        if output_shape is None:
            output_shape = self.axes_manager.signal_shape

        # ???
        _logger.info('Sideband pos in pixels: {}'.format(sb_position))
        _logger.info('Sideband aperture radius in pixels: {}'.format(sb_size))
        _logger.info('Sideband aperture smoothness in pixels: {}'.format(sb_smoothness))

        # Shows the selected sideband and the position of the sideband
        # if plotting:
        #     fft_holo = fft2(self.data) / np.prod(self.data.shape)
        #     fig, axs = plt.subplots(1, 1, figsize=(4, 4))
        #     axs.imshow(np.abs(fft_holo), clim=(0, 2.2))
        #     axs.scatter(sb_position[1], sb_position[0], s=20, color='red', marker='x')
        #     axs.set_xlim(sb_position[1]-sb_size, sb_position[1]+sb_size)
        #     axs.set_ylim(sb_position[0]-sb_size, sb_position[0]+sb_size)
        #     plt.show()

        # object electron wave:
        if self.axes_manager.navigation_size:
            wave_object = np.zeros((self.axes_manager.navigation_size, ) + output_shape, dtype='complex')
            for i in range(self.axes_manager.navigation_size):
                wave_object[i] = reconstruct(self.inav[i].data, holo_sampling=self.sampling,
                                             sb_size=sb_size[i], sb_position=sb_position[i], sb_smoothness=sb_smoothness[i],
                                             output_shape=output_shape, plotting=plotting)
        else:
            wave_object = reconstruct(self.data, holo_sampling=self.sampling,
                                      sb_size=sb_size, sb_position=sb_position, sb_smoothness=sb_smoothness,
                                      output_shape=output_shape, plotting=plotting)
        # Reconstructing reference wave and applying it (division):
        folded_ref = False

        if reference is None:
            wave_reference = 1
        else:
            # reference electron wave
            if reference.axes_manager.navigation_size:
                folded_ref = reference.unfold_navigation_space()
                wave_reference = np.zeros((reference.axes_manager.navigation_size, ) + output_shape, dtype='complex')
                for i in range(reference.axes_manager.navigation_size):
                    wave_reference[i] = reconstruct(reference.inav[i].data, holo_sampling=self.sampling,
                                                    sb_size=sb_size[i], sb_position=sb_position[i], sb_smoothness=sb_smoothness[i],
                                                    output_shape=output_shape, plotting=plotting)

                if reference.axes_manager.navigation_size != self.axes_manager.navigation_size:  # case when navdim of
                    # reference is >0 and not equal to that of self:

                    warnings.warn('The navigation size of the reference and the hologram do not match! Reference wave '
                                  'will be averaged')
                    wave_reference = np.mean(wave_reference, axis=0)

            else:
                if self.axes_manager.navigation_size:  # use only one slice of reconstruction parameters for reference:
                    wave_reference = reconstruct(reference.data, holo_sampling=self.sampling,
                                                 sb_size=sb_size[0], sb_position=sb_position[0],
                                                 sb_smoothness=sb_smoothness[0],
                                                 output_shape=output_shape, plotting=plotting)
                else:
                    wave_reference = reconstruct(reference.data, holo_sampling=self.sampling,
                                                 sb_size=sb_size, sb_position=sb_position, sb_smoothness=sb_smoothness,
                                                 output_shape=output_shape, plotting=plotting)

        wave = wave_object / wave_reference
        wave_image = self._deepcopy_with_new_data(wave)
        wave_image.set_signal_type('electron_wave')  # New signal is a wave image!

        # Reconstruction parameters are stored in holo_reconstruction_parameters:
        rec_param_dict = OrderedDict([('sb_position', sb_position), ('sb_size', sb_size),
                                      ('sb_units', sb_unit), ('sb_smoothness', sb_smoothness)])

        wave_image.metadata.Signal.add_node('holo_reconstruction_parameters')
        wave_image.metadata.Signal.holo_reconstruction_parameters.add_dictionary(rec_param_dict)

        wave_image.axes_manager.signal_axes[0].size = output_shape[0]
        wave_image.axes_manager.signal_axes[1].size = output_shape[1]
        wave_image.axes_manager.signal_axes[0].scale = self.sampling[0] * self.axes_manager.signal_shape[0] / \
                                                       output_shape[0]
        wave_image.axes_manager.signal_axes[1].scale = self.sampling[1] * self.axes_manager.signal_shape[1] / \
                                                       output_shape[1]
        if folded:
            self.fold()
            wave_image.fold()
        if folded_ref:
            reference.fold()

        return wave_image
